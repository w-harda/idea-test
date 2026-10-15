"""受控恢复：只从Epoch1执行Epoch2–3，冻结总目标10；默认CPU预检。"""
import argparse,json,os,subprocess,threading,time,traceback
from pathlib import Path
from unittest.mock import patch
import torch
import cache_io as io
import epoch_training as et
import epoch_checkpoint as ck
import irra_image_adapter as adapter
import phase26_train as p26
from phase26_support import state_hash
from phase27_analysis import analyze,audit_epoch
from image_only_generator import ImageOnlyGenerator
from budget_projection import DEFAULT_EPSILON
from training_data import checked_cache,make_loader

CACHE=p26.CACHE;OUTPUT=p26.OUTPUT;STAGE=io.OUTPUTS/"phase27"
EPOCH1_SHA="876547ffeb827cbb4691bbf5b5d473da495590d15df2adba12c9e1415e52a27a"
CONFIG_SHA="7e87bd272abfee440afc13109d439507a34b140506a19a132a8f740afe093bc4"

def launch_two(runner,cache,output,config,**kwargs):
    if config["epochs"]!=10:raise ValueError("persistent total target must remain 10")
    return runner(cache,output,config,resume=True,stop_after_epochs=2,**kwargs)

def adam_steps(opt):
    return sorted({int(s["step"]) for s in opt.state.values()})

def assert_restored(g,opt,value,expected_epoch):
    if (value["completed_epoch"]!=expected_epoch or value["next_epoch"]!=expected_epoch+1
        or state_hash(g.state_dict())!=state_hash(value["generator"])
        or state_hash(opt.state_dict())!=state_hash(value["optimizer"])
        or state_hash(ck.rng_state())!=state_hash(value["rng"])
        or adam_steps(opt)!=[expected_epoch*8068]):
        raise ValueError("G/Adam/RNG boundary restore mismatch")

def preflight():
    config,m=p26.preflight()
    io.verify({str(p26.CONFIG):CONFIG_SHA,str(OUTPUT/"frozen_config.json"):CONFIG_SHA,
               str(OUTPUT/"checkpoints/epoch_0001/checkpoint.pt"):EPOCH1_SHA})
    contract=et.contract_for(m,CACHE,config)
    if io.digest(contract)!=io.digest(io.load(OUTPUT/"training_protocol.json")):
        raise ValueError("original formal training contract mismatch")
    io.verify(contract["training_source_sha256"]);io.verify(config["entry_source_sha256"])
    versions=sorted(p.name for p in (OUTPUT/"checkpoints").glob("epoch_*"))
    if versions!=["epoch_0001"]:raise ValueError("entry requires exactly completed Epoch1; never reinitializes/repeats")
    value=ck.verify_version(OUTPUT/"checkpoints/epoch_0001",contract)
    if value["next_epoch"]!=2 or sorted({int(s["step"]) for s in value["optimizer"]["state"].values()})!=[8068]:
        raise ValueError("Epoch1 next_epoch/Adam state mismatch")
    return config,m,value

def protected_sources(value):
    # 新增文件不会改变旧缓存/训练契约；本阶段另锁全部既有tracked源码和不可变Epoch1证据。
    repo=io.ROOT.parent.parent
    names=subprocess.check_output(["git","ls-files","-z"],cwd=repo).decode().split("\0")
    paths=[repo/n for n in names if n and (repo/n).is_file()]
    paths.extend((OUTPUT/"checkpoints/epoch_0001").iterdir())
    paths.extend(Path(p) for p in value["progress"]["supplementary_log_sha256"])
    paths.extend([Path(value["progress"]["sample_log_path"]),OUTPUT/"frozen_config.json",
                  OUTPUT/"phase26_summary.json",OUTPUT/"phase26_completed.json"])
    result=io.load(io.OUTPUTS/"phase26/protected_sources.json")
    result.update({str(p):io.sha(p) for p in paths})
    return result

def run():
    config,m,epoch1=preflight()
    if STAGE.exists():raise ValueError("Phase27 evidence directory exists; inspect evidence before any rerun")
    STAGE.mkdir()
    protected=protected_sources(epoch1);io.save_json(STAGE/"protected_sources.json",protected)
    io.save_json(STAGE/"execution_protocol.json",{
        "resume":True,"start_epoch":2,"stop_after_epochs":2,"last_authorized_epoch":3,"persistent_target":10,
        "original_epoch1_sha256":EPOCH1_SHA,"config_sha256":CONFIG_SHA,
        "new_source_sha256":et.file_contract(("phase27_resume.py","phase27_analysis.py")),
        "original_contract_sha256":io.digest(epoch1["contract"])})
    loader=make_loader(m,"cuda")
    handles=[];box={};context={};versions=None;phase="restore";epoch=1
    counts={};records=[];round_file=None;monitor_stop=None;monitor_thread=None;epoch_start=None
    completed=[];started=time.perf_counter()
    old_g=et.ImageOnlyGenerator;old_encoder=adapter.load_official;old_sample=et.train_single_sample
    old_train=et.train_epoch;old_validate=et.validate_epoch;old_save=et.save_epoch;old_restore=et.load_latest
    Adam=torch.optim.Adam
    def before_g(module,args):
        if phase=="train":
            counts["train_G"]+=1
            if (args[0].requires_grad or args[0].grad_fn is not None
                or not torch.equal(args[0],context["expected_image"])):
                raise ValueError("serial detach/original joint input mismatch")
            if versions is not None and versions!=tuple(p._version for p in module.parameters()):
                raise ValueError("immediate G update mismatch")
        elif phase=="validation":counts["validation_G"]+=1
        else:raise ValueError("unexpected G forward")
    def before_encoder(module,args):
        if phase=="train":
            counts["train_IRRA"]+=1
            if not args[0].requires_grad:raise ValueError("IRRA image gradient detached")
            def image_grad(grad):
                if not torch.isfinite(grad).all():raise ValueError("nonfinite IRRA image gradient")
                context["image_grad_norm"]=float(grad.double().norm())
            args[0].register_hook(image_grad)
        elif phase=="validation":
            counts["validation_IRRA"]+=1
            if torch.is_grad_enabled() or args[0].requires_grad:raise ValueError("validation gradients enabled")
        else:raise ValueError("unexpected IRRA forward")
    def g_factory(*args,**kwargs):
        g=old_g(*args,**kwargs);box["g"]=g;handles.append(g.register_forward_pre_hook(before_g));return g
    def load_encoder(*args,**kwargs):
        enc=old_encoder(*args,**kwargs);box["encoder"]=enc
        box["frozen_IRRA"]=state_hash(enc.state_dict());handles.append(enc.register_forward_pre_hook(before_encoder))
        return enc
    class AuditAdam(Adam):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);box["opt"]=self
            def backward(grad):
                if not torch.isfinite(grad).all():raise ValueError("nonfinite G gradient")
                counts["backward"]+=1
            handles.append(next(box["g"].parameters()).register_hook(backward))
        def zero_grad(self,*args,**kwargs):
            counts["zero_grad"]+=1;return super().zero_grad(*args,**kwargs)
        def step(self,*args,**kwargs):
            nonlocal versions
            with torch.no_grad():
                before=torch.cat([p.detach().reshape(-1) for p in box["g"].parameters()])
                result=super().step(*args,**kwargs)
                after=torch.cat([p.detach().reshape(-1) for p in box["g"].parameters()])
                if not torch.isfinite(after).all():raise ValueError("nonfinite updated G")
                context["changed"]=int(torch.count_nonzero(before!=after))
            counts["step"]+=1;versions=tuple(p._version for p in box["g"].parameters())
            return result
    def restore(*args,**kwargs):
        value=old_restore(*args,**kwargs)
        assert_restored(box["g"],box["opt"],value,1)
        io.save_json(STAGE/"initial_restore.json",{
            "G_Adam_RNG_exact":True,"next_epoch":2,"Adam_steps":adam_steps(box["opt"]),
            "G_sha256":state_hash(box["g"].state_dict()),"optimizer_sha256":state_hash(box["opt"].state_dict()),
            "rng_sha256":state_hash(ck.rng_state())})
        return value
    def monitor(path,stop):
        with path.open("x",encoding="utf-8") as f:
            while True:
                row={"epoch":epoch,"seconds":time.perf_counter()-epoch_start,
                     "samples":counts.get("samples",0),"steps":counts.get("step",0),
                     "allocated_MiB":torch.cuda.memory_allocated()/2**20,
                     "reserved_MiB":torch.cuda.memory_reserved()/2**20}
                try:
                    for name,query in (("GPU","--query-gpu=index,memory.used,memory.total,utilization.gpu"),
                                       ("processes","--query-compute-apps=pid,used_memory")):
                        p=subprocess.run(["nvidia-smi",query,"--format=csv,noheader"],capture_output=True,text=True,timeout=5)
                        if p.returncode:raise ValueError(p.stderr)
                        row[name]=p.stdout.strip()
                except Exception as exc:row["monitor_error"]=repr(exc)
                f.write(json.dumps(row,allow_nan=False)+"\n");f.flush()
                if stop.wait(10):break
            f.flush();os.fsync(f.fileno())
    def train(g,enc,opt,entries,loader,seed,current_epoch,weights,log_path=None):
        nonlocal epoch,phase,counts,records,round_file,monitor_stop,monitor_thread,epoch_start
        epoch=current_epoch
        if epoch not in (2,3) or adam_steps(opt)!=[(epoch-1)*8068]:raise ValueError("unauthorized epoch/start Adam")
        phase="train";counts={k:0 for k in ("train_G","train_IRRA","backward","step","zero_grad","validation_G","validation_IRRA","samples")}
        records=[];epoch_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        box["epoch_G_before"]=state_hash(g.state_dict())
        round_file=(STAGE/f"epoch_{epoch:04d}_round_audit.jsonl").open("x",encoding="utf-8")
        monitor_stop=threading.Event()
        monitor_thread=threading.Thread(target=monitor,args=(STAGE/f"epoch_{epoch:04d}_resources.jsonl",monitor_stop),daemon=True)
        monitor_thread.start()
        result=old_train(g,enc,opt,entries,loader,seed,epoch,weights,log_path)
        box["train_seconds"]=time.perf_counter()-epoch_start
        return result
    def sample(g,encoder,opt,image,c,**kwargs):
        position=counts["samples"]+1;k=c["K"];before=counts["step"]
        context.update(row_id=c["row_id"],K=k,expected_image=image.detach(),previous_feature=None)
        def observe(event):
            r=event["record"];idx=r["update_index"]-1
            if r["phase"]=="attribute":
                if idx:
                    v=c["attribute_jvps"][idx-1:idx].to(image.device)
                    ref=(context["previous_feature"]*v).sum(1)
                    if not torch.equal(event["previous_jvp"],v) or not torch.equal(event["previous_reference"],ref):
                        raise ValueError("previous-only Retain mismatch")
                context["expected_image"]=image.detach() if idx==k-1 else event["adversarial"]
            elif not torch.equal(event["input_image"],image):raise ValueError("joint original restart mismatch")
            context["previous_feature"]=event["feature"]
            record={**r,"epoch":epoch,"row_id":c["row_id"],"K":k,"sample_position":position,
                    "changed_parameter_count":context["changed"],"IRRA_input_gradient_l2":context["image_grad_norm"]}
            if record["cumulative_linf"]>DEFAULT_EPSILON:raise ValueError("budget violation")
            records.append(record);round_file.write(json.dumps(record,allow_nan=False)+"\n")
        result=old_sample(g,encoder,opt,image,c,observer=observe,**kwargs)
        if counts["step"]-before!=k+1:raise ValueError("K+1 step mismatch")
        counts["samples"]+=1;round_file.flush();os.fsync(round_file.fileno())
        if position%32==0:print("TRAIN_PROGRESS",epoch,position,"steps",counts["step"],"seconds",time.perf_counter()-epoch_start,flush=True)
        return result
    def validate(*args,**kwargs):
        nonlocal phase
        phase="validation";g=box["g"];opt=box["opt"]
        before=(state_hash(g.state_dict()),state_hash(opt.state_dict()),state_hash(ck.rng_state()),dict(counts))
        result=old_validate(*args,**kwargs)
        if before[:3]!=(state_hash(g.state_dict()),state_hash(opt.state_dict()),state_hash(ck.rng_state())):
            raise ValueError("validation mutated G/Adam/RNG")
        if counts["step"]!=before[3]["step"]:raise ValueError("validation optimizer step")
        io.save_json(STAGE/f"epoch_{epoch:04d}_validation.json",result);return result
    def save(out,g,opt,current_epoch,contract,progress):
        nonlocal phase
        expected={k:8068 for k in ("train_G","train_IRRA","backward","step","zero_grad")}
        expected.update(validation_G=128,validation_IRRA=128,samples=2048)
        if counts!=expected or current_epoch!=epoch or adam_steps(opt)!=[epoch*8068]:
            raise ValueError("Epoch counts/Adam mismatch")
        p26.finite_optimizer(opt)
        if state_hash(box["encoder"].state_dict())!=box["frozen_IRRA"]:raise ValueError("IRRA parameters/buffers changed")
        round_file.flush();os.fsync(round_file.fileno());round_file.close()
        monitor_stop.set();monitor_thread.join(timeout=12)
        if monitor_thread.is_alive():raise ValueError("resource log writer did not stop")
        io.verify(protected)
        result=audit_epoch(progress,m["entries"],records,epoch)
        progress["supplementary_log_sha256"]={str(STAGE/f"epoch_{epoch:04d}_{name}"):io.sha(STAGE/f"epoch_{epoch:04d}_{name}")
            for name in ("round_audit.jsonl","validation.json","resources.jsonl")}
        final=old_save(out,g,opt,epoch,contract,progress)
        # 独立新对象恢复；恢复保存时RNG，避免诊断建模消耗RNG影响下一Epoch。
        phase="boundary_restore"
        restored_g=ImageOnlyGenerator(**config["architecture"]).to(config["device"])
        restored_o=Adam(restored_g.parameters(),**config["optimizer"])
        value=ck.load_latest(out,contract,restored_g,restored_o)
        assert_restored(restored_g,restored_o,value,epoch)
        if (state_hash(g.state_dict())!=state_hash(restored_g.state_dict())
            or state_hash(opt.state_dict())!=state_hash(restored_o.state_dict())):raise ValueError("live/saved state mismatch")
        result.update(counts=dict(counts),checkpoint_sha256=io.sha(final/"checkpoint.pt"),
            sample_log_sha256=progress["sample_log_sha256"],Adam_steps=adam_steps(opt),
            G_Adam_RNG_restore_exact=True,IRRA_parameters_buffers_exact=True,
            G_changed=box["epoch_G_before"]!=state_hash(g.state_dict()),
            train_seconds=box["train_seconds"],epoch_seconds=time.perf_counter()-epoch_start,
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
        io.save_json(STAGE/f"epoch_{epoch:04d}_summary.json",result);completed.append(result)
        print("BOUNDARY_VERIFIED",epoch,result["checkpoint_sha256"],"Adam",result["Adam_steps"],flush=True)
        del restored_g,restored_o,value
        return final
    try:
        with patch.object(et,"ImageOnlyGenerator",new=g_factory),patch.object(adapter,"load_official",new=load_encoder), \
             patch.object(et.torch.optim,"Adam",new=AuditAdam),patch.object(et,"load_latest",new=restore), \
             patch.object(et,"train_epoch",new=train),patch.object(et,"train_single_sample",new=sample), \
             patch.object(et,"validate_epoch",new=validate),patch.object(et,"save_epoch",new=save), \
             patch.object(torch.autograd,"grad",side_effect=AssertionError("scoring/JVP forbidden")):
            g,opt,result=launch_two(et.run_epochs,CACHE,OUTPUT,config,loader=loader)
        if result["completed_epoch"]!=3 or len(completed)!=2 or adam_steps(opt)!=[24204]:
            raise ValueError("controlled stop mismatch")
        io.verify(protected)
        io.verify(io.load(STAGE/"execution_protocol.json")["new_source_sha256"])
        audit=analyze(CACHE,OUTPUT,STAGE)
        audit.update(epochs_this_run=[2,3],total_seconds=time.perf_counter()-started,
                     initial_restore_exact=True,boundary_restores_exact=True)
        io.save_json(STAGE/"summary.json",audit)
        io.save_json(STAGE/"completed.json",{"method_audit_passed":True,"summary_sha256":io.sha(STAGE/"summary.json")})
        print("PHASE27_COMPLETE Epoch3; Adam24204; Epoch4 not started",flush=True)
    except BaseException as exc:
        io.save_json(STAGE/"failure.json",{"error":repr(exc),"traceback":traceback.format_exc(),"epoch":epoch,
            "counts":counts,"sample":{k:v for k,v in context.items() if k in ("row_id","K")},"Epoch4_started":False})
        raise
    finally:
        if monitor_stop:monitor_stop.set()
        if monitor_thread:monitor_thread.join(timeout=12)
        if round_file and not round_file.closed:round_file.close()
        for handle in handles:handle.remove()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--allow-epochs2-3",action="store_true")
    a=p.parse_args(argv)
    if a.allow_epochs2_3:run()
    else:
        _,m,v=preflight()
        print("CPU_PREFLIGHT",m["counts"],"next_epoch",v["next_epoch"],"Adam8068; target10; limit2; no GPU/model")
if __name__=="__main__":main()
