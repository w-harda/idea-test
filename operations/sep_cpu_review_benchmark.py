#!/usr/bin/env python3
"""独立固定合成输入CPU微基准；不导入执行器、不加载模型/历史Tensor。"""
from pathlib import Path
import argparse,sys,os,time,json,statistics,platform,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/framework')]
import torch
from sep_cpu_review_support import PIDPlan,synthetic_case,exact,first_difference,bit_bytes,observe_selector
from tbps.baselines.cpgc_sep.sep_bridge import pid_logits

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);assert not out.exists()
 assert not torch.cuda.is_initialized();cuda_calls=[0]
 def forbid(*a,**k):cuda_calls[0]+=1;raise RuntimeError('CPU benchmark forbids CUDA/model checkpoint/backward')
 torch.cuda._lazy_init=forbid;torch.cuda.init=forbid;torch.load=forbid;torch.Tensor.cuda=forbid;torch.Tensor.backward=forbid;torch.autograd.grad=forbid
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 affinity=None
 if hasattr(os,'sched_getaffinity'):
  allowed=os.sched_getaffinity(0);affinity=min(allowed);os.sched_setaffinity(0,{affinity})
 torch.manual_seed(20261103)
 def traced(fn):
  def invoke():return fn()
  def trace(frame,event,arg):return trace if frame.f_code is invoke.__code__ else None
  old=sys.gettrace();sys.settrace(trace)
  try:return invoke()
  finally:sys.settrace(old)
 def sample(fn):
  t=time.perf_counter_ns();v=fn();return (time.perf_counter_ns()-t)/1e6,v
 cases=[]
 for name,G,P,B,trace_mode,repeats in [('RSTP_shape',1000,200,1,False,7),('CUHK_shape',3074,1000,1,False,7),('CUHK_B8_fixture',3074,1000,8,False,7),('ICFG_shape',19848,1000,1,False,5),('CUHK_trace_pattern',3074,1000,1,True,3)]:
  pids,scores=synthetic_case(G,P,B);plan=PIDPlan.build(pids);reference=pid_logits(scores,pids)[0];warm={'reference':lambda:pid_logits(scores,pids),'precomputed':lambda:plan.aggregate(scores),'cold_precomputed':lambda:PIDPlan.build(pids).aggregate(scores)}
  wrapped={k:(lambda f=f:traced(f)) if trace_mode else f for k,f in warm.items()}
  for f in wrapped.values():assert exact(reference,f()[0])
  measurements={key:[] for key in wrapped};setup=[]
  for rep in range(repeats):
   keys=list(wrapped)
   if rep%2:keys.reverse()
   for key in keys:
    elapsed,(values,order)=sample(wrapped[key]);assert order==plan.order and exact(reference,values),first_difference(reference,values);measurements[key].append(elapsed)
   elapsed,_=sample(lambda:PIDPlan.build(pids));setup.append(elapsed)
  med={k:statistics.median(v) for k,v in measurements.items()};row={'case':name,'Gallery':G,'PID':P,'batch':B,'dtype':'float32','production_like_global_trace':trace_mode,'repeats':repeats,'milliseconds':measurements,'median_ms':med,'precompute_setup_median_ms':statistics.median(setup),'measured_warm_speedup':med['reference']/med['precomputed'],'measured_cold_speedup':med['reference']/med['cold_precomputed'],'output_bitwise_exact':True,'output_sha256':hashlib.sha256(bit_bytes(reference)).hexdigest()};cases.append(row);print('CPU_MICROBENCH',json.dumps(row),flush=True)
 # 限定合成观察器：仅测禁用深层trace的开销/捕获结果，不能据此升级生产修改为A类。
 pids,scores=synthetic_case(512,128,4);times={'original_scope':[],'paused_inside_reducer':[]};counts={}
 for rep in range(5):
  for key,pause in [('original_scope',False),('paused_inside_reducer',True)]:
   elapsed,value=sample(lambda:observe_selector(scores,pids,pause=pause));times[key].append(elapsed);counts[key]=value[2]
   if key=='original_scope':reference_capture=value[:2]
   else:assert value[:2]==reference_capture
 trace_review={'shape':[4,512],'PID':128,'milliseconds':times,'median_ms':{k:statistics.median(v) for k,v in times.items()},'callback_counts_last':counts,'synthetic_capture_exact':True,'classification':'B; single-thread synthetic observer only'}
 # 合成CPU矩阵演示：批次改变可能切换浮点归约；不是IRRA/RaSa真实编码证明。
 g=torch.Generator(device='cpu').manual_seed(20261103);q=torch.randn(8,512,generator=g);gallery=torch.randn(3074,512,generator=g)
 gallery[1::2]=gallery[::2]+torch.randn(1537,512,generator=g)*1e-7
 batched=q@gallery.T;single=torch.cat([q[i:i+1]@gallery.T for i in range(len(q))]);equal_bytes=exact(single,batched);value_diffs=int((single!=batched).sum());rank1=torch.argsort(single,dim=1,descending=True,stable=True);rank2=torch.argsort(batched,dim=1,descending=True,stable=True)
 batch_demo={'shape_query':[8,512],'shape_gallery':[3074,512],'dtype':'float32','seed':20261103,'near_tie_gallery_pairs':True,'bitwise_exact':equal_bytes,'different_score_values':value_diffs,'max_abs_difference':float((single-batched).abs().max()),'rows_with_any_rank_difference':int((rank1!=rank2).any(1).sum()),'meaning':'synthetic CPU GEMV/GEMM counterexample only; no target/model or attack input used'}
 result={'status':'PASS','seed':20261103,'torch':torch.__version__,'python':platform.python_version(),'host':platform.node(),'platform':platform.platform(),'CPU_model':next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),''),'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'process_affinity_CPU':affinity,'CUDA_initialized':torch.cuda.is_initialized(),'forbidden_runtime_calls':cuda_calls[0],'real_model_loads':0,'real_query_encodings':0,'history_Tensor_reads':0,'cases':cases,'trace_scope':trace_review,'batch_float_demo':batch_demo,'limitation':'CPU aggregation/observer microbenchmark, not end-to-end SEP speed or GPU/model numerical equivalence'}
 assert not result['CUDA_initialized'] and cuda_calls[0]==0
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('CPU_REVIEW_BENCHMARK_PASS',flush=True)
if __name__=='__main__':main()
