"""CPU合成分数下的新旧生产Classifier入口微基准；不加载模型或真实Query。"""
from pathlib import Path
import sys,os,time,json,argparse,statistics,platform,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/framework')]
import torch
from tbps.baselines.cpgc_sep import sep_source as source
from sep_cpu_review_support import synthetic_case,exact,first_difference,bit_bytes,rng_signature
from sep_pid_integration_support import baseline_api,World,invoke,BASE

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();output=Path(a.output);assert not output.exists();assert not torch.cuda.is_initialized();forbidden=[0]
 def deny(*a,**k):forbidden[0]+=1;raise RuntimeError('CPU-only benchmark forbids CUDA/checkpoint/backward')
 torch.cuda._lazy_init=deny;torch.cuda.init=deny;torch.load=deny;torch.Tensor.cuda=deny;torch.Tensor.backward=deny;torch.autograd.grad=deny
 torch.set_num_threads(1);torch.set_num_interop_threads(1);cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu});reference=baseline_api();cases=[]
 def traced(fn):
  def target():return fn()
  def trace(frame,event,arg):return trace if frame.f_code is target.__code__ else None
  old=sys.gettrace();sys.settrace(trace)
  try:return target()
  finally:sys.settrace(old)
 def timed(fn):
  t=time.perf_counter_ns();value=fn();return(time.perf_counter_ns()-t)/1e6,value
 for name,G,P,trace,n in [('RSTP_shape',1000,200,False,7),('CUHK_shape',3074,1000,False,7),('ICFG_shape',19848,1000,False,5),('CUHK_trace_pattern',3074,1000,True,3)]:
  pids,matrix=synthetic_case(G,P,1);world=World(pids,lambda model,text:matrix[0]);old_obj=world.factory(reference.classifier);new_obj=world.factory(source.classifier);values=reference.pid_logits(matrix,pids)[0];durations={'old_entry':[],'new_entry':[]};before=rng_signature()
  for obj in(old_obj,new_obj):obj.cache.clear();assert exact(invoke(obj,['synthetic']),values)
  for rep in range(n):
   pairs=[('old_entry',old_obj),('new_entry',new_obj)]
   if rep%2:pairs.reverse()
   for label,obj in pairs:
    obj.cache.clear();world.clear_events();fn=lambda:invoke(obj,['synthetic']);elapsed,value=timed((lambda:traced(fn)) if trace else fn);assert exact(values,value),first_difference(values,value);durations[label].append(elapsed)
  assert rng_signature()==before
  cold=[]
  for _ in range(5):
   elapsed,obj=timed(lambda:world.factory(source.classifier));cold.append(elapsed)
  med={k:statistics.median(v) for k,v in durations.items()};row={'case':name,'Gallery':G,'PIDs':P,'batch':1,'dtype':'float32','trace_pattern':trace,'repeats':n,'milliseconds':durations,'median_ms':med,'new_classifier_creation_median_ms':statistics.median(cold),'measured_entry_speedup':med['old_entry']/med['new_entry'],'bitwise_exact':True,'RNG_unchanged':True,'input_scores_sha256':hashlib.sha256(bit_bytes(matrix)).hexdigest()};cases.append(row);print('PRODUCTION_ENTRY_CPU_BENCH',json.dumps(row),flush=True)
 result={'status':'PASS','reference_commit':BASE,'seed':20261103,'torch':torch.__version__,'python':platform.python_version(),'CPU':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'threads':1,'affinity_CPU':cpu,'CUDA_initialized':torch.cuda.is_initialized(),'forbidden_calls':forbidden[0],'model_checkpoints_loaded':0,'real_query_encodings':0,'history_Full_Tensor_reads':0,'cases':cases,'scope':'actual old/new production PIDClassifier.forward with fixed synthetic encoder/scorer; not real model or Full attack speed'}
 assert not result['CUDA_initialized'] and forbidden[0]==0;output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
