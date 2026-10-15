"""原prepare_full --allow-full-preparation的观测外壳；不修改producer或算法。"""
import argparse,json,subprocess,threading,time,traceback,uuid
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import torch
import cache_io as io
import prepare_full
import irra_image_adapter as adapter
from phase25_audit import preflight

class CallCounter:
    """只保存整数，不保存张量参数/返回值或计算图。"""
    def __init__(self,function):
        self.function=function;self.call_count=0
    def __call__(self,*args,**kwargs):
        self.call_count+=1
        return self.function(*args,**kwargs)

def now():return datetime.now(timezone.utc).isoformat()

def module_sha(module):
    import hashlib
    h=hashlib.sha256()
    for name,t in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(str((t.dtype,tuple(t.shape))).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def run():
    baseline=preflight()
    out=io.safe(io.OUTPUTS/"phase25/preparation_attempts"/uuid.uuid4().hex)
    out.mkdir(parents=True)
    io.save_json(out/"preflight.json",baseline)
    io.save_json(out/"invocation.json",{"argv":["prepare_full.py","--allow-full-preparation"],
        "UTC_start":now(),"producer_sha256":prepare_full.file_contract(prepare_full.CACHE_FILES),
        "wrapper_sha256":{str(Path(__file__).resolve()):io.sha(__file__)}})
    stop=threading.Event();started=time.perf_counter();model_times=[];encoders=[];model_hashes=[]
    published_before=len(list((io.OUTPUTS/"conditions_full/samples").glob("*/metadata.json")))
    def monitor():
        with (out/"monitor.jsonl").open("x",encoding="utf-8") as f:
            while True:
                record={"UTC":now(),"seconds":time.perf_counter()-started,
                    "published_samples":len(list((io.OUTPUTS/"conditions_full/samples").glob("*/metadata.json")))}
                for kind,query in (("gpu",["--query-gpu=index,memory.used,memory.total,utilization.gpu"]),
                                   ("processes",["--query-compute-apps=pid,used_memory"])):
                    try:
                        result=subprocess.run(["nvidia-smi",*query,"--format=csv,noheader"],capture_output=True,text=True,timeout=5)
                        record[kind]=result.stdout.strip();record[kind+"_returncode"]=result.returncode
                    except Exception as exc:record[kind+"_error"]=repr(exc)
                f.write(json.dumps(record)+"\n");f.flush()
                if stop.wait(5):break
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    original_load=adapter.load_official
    def load(*args,**kwargs):
        t=time.perf_counter();encoder=original_load(*args,**kwargs)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        model_times.append(time.perf_counter()-t);encoders.append(encoder);model_hashes.append(module_sha(encoder))
        return encoder
    status="failed";failure=None;grad_calls=0;gradients=CallCounter(torch.autograd.grad)
    try:
        from image_only_generator import ImageOnlyGenerator
        import epoch_training,serial_training
        with patch.object(adapter,"load_official",side_effect=load), \
             patch.object(torch.Tensor,"backward",side_effect=AssertionError("scoring/training backward forbidden")), \
             patch.object(torch.autograd,"grad",new=gradients), \
             patch.object(ImageOnlyGenerator,"__init__",side_effect=AssertionError("G construction forbidden")), \
             patch.object(torch.optim,"Adam",side_effect=AssertionError("optimizer construction forbidden")), \
             patch.object(epoch_training,"run_epochs",side_effect=AssertionError("Epoch training forbidden")), \
             patch.object(serial_training,"train_single_sample",side_effect=AssertionError("G training forbidden")):
            prepare_full.main(["--allow-full-preparation"])
            grad_calls=gradients.call_count
        if encoders:
            torch.cuda.synchronize()
            for e,h in zip(encoders,model_hashes):
                if module_sha(e)!=h or any(p.requires_grad or p.grad is not None for p in e.parameters()):
                    raise ValueError("frozen official IRRA changed")
        io.verify(io.load(io.OUTPUTS/"phase25/protected_sources.json"))
        status="complete"
    except BaseException as exc:
        failure={"exception":repr(exc),"traceback":traceback.format_exc()}
        raise
    finally:
        stop.set();thread.join(timeout=12)
        metrics={"status":status,"UTC_end":now(),"wall_seconds":time.perf_counter()-started,
            "official_model_load_seconds":model_times,"published_before":published_before,
            "published_after":len(list((io.OUTPUTS/"conditions_full/samples").glob("*/metadata.json"))),
            "CUDA_initialized":torch.cuda.is_initialized(),
            "peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else None,
            "peak_reserved_MiB":torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else None,
            "actual_JVP_autograd_grad_calls":gradients.call_count if gradients is not None else grad_calls,"new_scoring_backwards":0,
            "G_forwards":0,"optimizer_steps":0,"IRRA_parameters_buffers_exact":status=="complete",
            "failure":failure,"G_training":False,"gallery_evaluation":False}
        io.save_json(out/"metrics.json",metrics)
        print("PHASE25_PREPARE_METRICS "+json.dumps(metrics),flush=True)
    return out

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--allow-full-preparation",action="store_true")
    a=p.parse_args(argv)
    if not a.allow_full_preparation:print(json.dumps(preflight()),flush=True)
    else:print("ATTEMPT",run(),flush=True)
if __name__=="__main__":main()
