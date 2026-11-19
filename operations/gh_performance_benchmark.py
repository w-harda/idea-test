"""真实Train特征、已训练评价器及克隆H单步的性能/数值对照；不写训练产物。"""
from pathlib import Path
import sys,json,time,copy,types,statistics
from dataclasses import dataclass
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path("/home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116")
BASE=Path("/home/lzf/ldx/outputs/tbps-framework-gh-attribute-response-v1-20261116")
CTL=Path("/home/lzf/ldx/tmp/gh_performance_v1_20261119")
sys.path.insert(0,str(SOURCE/"src"))
import tbps
tbps.__path__.append(str(ROOT/"src/tbps"))
import torch
from tbps.artifacts import read_json,sha_file,checked,FileRef,ArtifactStore
from tbps.alternating.state import tree_sha
from tbps.attraction.objectives import ProfileKL,Settings
from tbps.attribute_response.settings import Settings as TeacherSettings
from tbps.attribute_response.teacher import ResponseTeacher
from tbps.performance.fast import candidate_rewards,fast_predict
from tbps.performance.runtime import verify_version,CONFIG,MANIFEST
from tbps.models.irra import fp32_runtime

def timed(fn,count=3):
 times=[];value=None
 for _ in range(count):
  torch.cuda.synchronize();start=time.perf_counter();value=fn()
  torch.cuda.synchronize();times.append(time.perf_counter()-start)
 return value,statistics.median(times)

def main():
 spec=verify_version();CTL.mkdir(exist_ok=True)
 output=CTL/"benchmark.json"
 if output.exists():raise RuntimeError("immutable benchmark already exists")
 torch.set_num_threads(2)
 from tbps.attribute_response.sources import preflight
 f,c,bundle,gate,proof=preflight(CONFIG,"train")
 gate.initialize_cuda()
 print("BENCH_STAGE CPU_source_preflight_PASS",flush=True)
 ledger=BASE/"train/ledgers/round_0030"
 summary=read_json(ledger/"ledger.json")
 shard=summary["sharded_arrays"]["H"]["chunks"][0]["file"]
 h=max((x for x in read_json(ledger/shard)["items"] if not x.get("fixed_NOOP") and x["attribute_response"]["credits"]), key=lambda x:x["attribute_response"]["counterfactual_unique_texts"])
 cf=h["attribute_response"]["counterfactual_features"]
 path=ledger/cf["feature_file"];checked(FileRef(str(path),cf["feature_sha256"]))
 features=torch.load(path,map_location="cpu",weights_only=True)
 anchorpath=BASE/"train/attraction_reference/anchors.pt"
 ArtifactStore(BASE/"train",new_base=BASE).verify("attraction_reference")
 anchors=torch.load(anchorpath,map_location="cpu",weights_only=True)
 clean=torch.tensor(h["attribute_response"]["original_features"])[:1]
 partners=torch.tensor(h["attraction"]["partner_view_features"])
 cfg=read_json(CONFIG);profile=ProfileKL(anchors,Settings(**cfg["attraction"]))
 result={"schema":"gh-performance-benchmark-1","manifest_sha256":sha_file(MANIFEST),
  "dataset_role":"official Train only","selection":"largest legal counterfactual support among first 32 prior Train records","historical_JVP_recomputed":0,
  "historical_weights_or_ledgers_written":False,"not_attack_metrics":True}
 with fp32_runtime():
  def reference():
   return [float(profile.terms(f[None].expand(3,-1),clean.expand(3,-1),partners)["reward"].mean()) for f in features]
  reference_value,oldtime=timed(reference)
  candidate_rewards(profile,features,clean,partners)
  value,newtime=timed(lambda:candidate_rewards(profile,features,clean,partners))
  a=torch.tensor(reference_value);b=torch.tensor(value)
  torch.testing.assert_close(a,b,atol=5e-5,rtol=2e-4)
  result["candidate_KL"]={"candidates":len(features),"max_abs_difference":float((a-b).abs().max()),
    "original_seconds":oldtime,"optimized_seconds":newtime,"speedup":oldtime/newtime}
  ArtifactStore(BASE/"train/checkpoints",new_base=BASE).verify("teacher_30")
  teacherstate=torch.load(BASE/"train/checkpoints/teacher_30/teacher.pt",map_location="cpu",weights_only=False)
  teacher=ResponseTeacher(TeacherSettings(**teacherstate["settings"]));teacher.load_state_dict(teacherstate)
  d=h["attribute_response"]["credits"][0]
  size=max(64,len(features))
  batch={"keys":[d["attribute"]]*size,"states":[d["states"]]*size,
    "actions":[[.1]*6]*size,"histories":[d["teacher_context"]["history"]]*size,"versions":[30]*size}
  ta,told=timed(lambda:teacher.predict(batch));tb,tnew=timed(lambda:fast_predict(teacher,batch))
  assert torch.equal(ta,tb),"teacher prediction must be bitwise identical"
  print("BENCH_STAGE candidate_KL_PASS",result["candidate_KL"],flush=True)
  result["teacher_history"]={"rows":size,"bitwise_equal":True,"original_seconds":told,
      "optimized_seconds":tnew,"speedup":told/tnew}

  print("BENCH_STAGE teacher_history_PASS",result["teacher_history"],flush=True)
  # 实际H策略/Adam、真实Train文本和IRRA，仅在克隆状态中执行一对完整H更新。
  from tbps.alternating.native_sources import native_session
  from tbps.models.irra import IRRAAdapter
  from tbps.models.irra_source import IRRASource
  from tbps.alternating.native_backend import TextVictim
  source=IRRASource.from_dict(read_json(c["sources"]["dataset_config"]["path"])["irra"])
  adapter=IRRAAdapter.from_source(source,device="cuda:0",allow_model_load=True)
  model=adapter._backend.eval().requires_grad_(False);modelsha=tree_sha(model.state_dict())
  victim=TextVictim(model,bundle.probe)
  class Runtime:
   probe=bundle.probe
   def encode_text(self,texts):
    with torch.inference_mode():return victim.encode_texts(texts).detach().cpu().clone()
   def assert_frozen(self):
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
  runtime=Runtime()
  @dataclass
  class Diagnostic:
   total:float=0.
  class Reward:
   def __init__(self):self.runtime=runtime
   def evaluate_many(self,original,variants):
    self.features=runtime.encode_text([original,*variants])
    return (Diagnostic(),Diagnostic())
   def diagnostics(self):return {"benchmark_diagnostic_fixture_only":True}
  parentpath=BASE/"recovery_evidence/20261119/committed_round30/boundaries/round_0030/state.pt"
  parentmanifest=read_json(parentpath.parent/"manifest.json")
  checked(FileRef(str(parentpath),parentmanifest["files_sha256"]["state.pt"]))
  parent=torch.load(parentpath,map_location="cpu",weights_only=False)
  qual=read_json(Path(c["prototype"]["history_root"])/"qualification/train.json")["entries"]
  entry=next(e for e in qual if e["identity"]["row_id"]==h["row_id"])
  assert entry["eligible"]
  with native_session(c,reuse=bundle.modules) as session:
   phase=session.import_module("phase4_training");policym=session.import_module("policy")
   constraints=session.import_module("constraints");data=session.import_module("formal_training.data")
   example=data.example_from(entry)
   original_step=__import__("tbps.attribute_response.h_step",fromlist=["train_step"]).train_step
   fast_step=__import__("tbps.performance.h_step",fromlist=["train_step"]).train_step
   policies=[];optimizers=[];teachers=[]
   for i in range(2):
    p=copy.deepcopy(bundle.h).requires_grad_(True);p.load_state_dict(parent["H"])
    opt=torch.optim.Adam(p.parameters(),lr=1e-4);opt.load_state_dict(copy.deepcopy(parent["H_Adam"]))
    t=ResponseTeacher(TeacherSettings(**teacherstate["settings"]));t.load_state_dict(copy.deepcopy(teacherstate))
    if i:t.predict=types.MethodType(fast_predict,t)
    policies.append(p);optimizers.append(opt);teachers.append(t)
   # 只按合法动作选首个包含实际编辑的诊断seed，不看奖励或Test结果。
   diagnostic=copy.deepcopy(policies[0])
   selected_seed=None
   for candidate_seed in range(731,1000):
    torch.manual_seed(candidate_seed)
    with torch.no_grad():
     diagnostic.train();diagnostic.rollout(example,"greedy")
     trial=diagnostic.rollout(example,"sample",generator=torch.Generator().manual_seed(candidate_seed))
    if trial.edits:
     selected_seed=candidate_seed;break
   assert selected_seed is not None,"need a representative actual-edit path"
   for text in (example.caption,):runtime.encode_text([text])
   outputs=[None,None];durations=[[],[]]
   for repeat in range(3):
    for i in ((0,1) if repeat%2==0 else (1,0)):
     step=(original_step,fast_step)[i]
     policies[i].load_state_dict(parent["H"])
     optimizers[i].load_state_dict(copy.deepcopy(parent["H_Adam"]))
     teachers[i].load_state_dict(copy.deepcopy(teacherstate))
     torch.manual_seed(selected_seed);sampler=torch.Generator().manual_seed(selected_seed)
     torch.cuda.synchronize();start=time.perf_counter()
     log,_=step(phase,policym,constraints,policies[i],optimizers[i],example,sampler,lambda k:Reward(),
       1,h["row_id"],profile,partners,teachers[i],[[0.]*12 for _ in example.attributes],
       version=30,response_probe=None,feature_sink=None)
     torch.cuda.synchronize();durations[i].append(time.perf_counter()-start);outputs[i]=log
   durations=[statistics.median(x) for x in durations]
   for name in ("sampled","greedy"):
    assert outputs[0][name]==outputs[1][name],"identical sampled/greedy legal actions"
   assert outputs[0]["K"]==outputs[1]["K"]
   assert [x["optimizer_steps_this_step"] for x in outputs]==[1,1]
   assert outputs[0]["attribute_response"]["teacher_after"]==outputs[1]["attribute_response"]["teacher_after"]
   maxgrad=0.;maxparam=0.
   for (name,a),(other,b) in zip(policies[0].named_parameters(),policies[1].named_parameters()):
    assert name==other and (a.grad is None)==(b.grad is None)
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    maxparam=max(maxparam,float((a-b).abs().max()))
    if a.grad is not None:
     torch.testing.assert_close(a.grad,b.grad,atol=5e-5,rtol=2e-4)
     maxgrad=max(maxgrad,float((a.grad-b.grad).abs().max()))
   result["real_H_step"]={"row":h["row_id"],"diagnostic_seed":selected_seed,"actual_edits":len(outputs[0]["sampled"]["edits"]),"counterfactual_texts":outputs[0]["attribute_response"]["counterfactual_unique_texts"],"timing_repeats":3,"original_seconds":durations[0],
      "optimized_seconds":durations[1],"speedup":durations[0]/durations[1],
      "sampled_and_greedy_actions_exact":True,"same_K_and_one_optimizer_update":True,
      "teacher_state_exact":True,"max_gradient_abs_difference":maxgrad,
      "max_parameter_abs_difference":maxparam,"G_updates":0,
      "auxiliary_old_R_sim_R_rank_diagnostics":"zero fixture; neither optimized objective uses these",
      "lookahead_probe":"not invoked in cloned H equivalence test"}
  assert tree_sha(model.state_dict())==modelsha,"IRRA frozen state changed"
 result["status"]="GH_PERFORMANCE_BENCHMARK_PASS"
 result["GPU_peak_allocated"]=torch.cuda.max_memory_allocated()
 with output.open("x") as out:json.dump(result,out,ensure_ascii=False,indent=2)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
