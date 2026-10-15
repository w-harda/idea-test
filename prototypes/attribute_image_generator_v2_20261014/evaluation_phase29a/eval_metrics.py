"""直接调用官方rank；同PID多正例，无camera过滤。"""
import importlib.util
import torch
from eval_common import OFFICIAL,METRICS,FEATURE_ATOL,CLEAN_METRIC_ATOL_PP
def official_rank():
 spec=importlib.util.spec_from_file_location("gv2_official_rank",OFFICIAL/"utils/metrics.py")
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.rank
def normalized(x,n):
 if (x.shape!=(n,512) or x.dtype!=torch.float32 or x.device.type!="cpu"
     or not torch.isfinite(x).all() or not torch.allclose(x.norm(dim=1),torch.ones(n),atol=FEATURE_ATOL,rtol=0)):
  raise ValueError("invalid finite normalized CPU [N,512] features")
def metrics(query,gallery,qids,gids):
 normalized(query,len(qids));normalized(gallery,len(gids))
 if len(gids)<10 or not set(qids)<=set(gids):raise ValueError("missing PID positive or Gallery<10")
 sim=torch.cat([query[i:i+128]@gallery.T for i in range(0,len(query),128)])
 cmc,ap,inp,order=official_rank()(sim,torch.tensor(qids),torch.tensor(gids),max_rank=10,get_mAP=True)
 result={"mAP":float(ap),"R@1":float(cmc[0]),"R@5":float(cmc[4]),"R@10":float(cmc[9]),"mINP":float(inp)}
 if any(not torch.isfinite(torch.tensor(v)) for v in result.values()):raise ValueError("nonfinite metrics")
 return result,sim
def drops(clean,adv):
 return {k:((clean[k]-adv[k])/clean[k]*100 if clean[k]!=0 else None) for k in METRICS}
def check_clean(actual,expected):
 errors={k:abs(actual[k]-expected[k]) for k in METRICS}
 if any(v>CLEAN_METRIC_ATOL_PP for v in errors.values()):raise ValueError("clean protocol mismatch: "+str(errors))
 return errors
