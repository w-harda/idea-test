"""显式授权入口。默认仅CPU预检；不会从import或参数缺省启动GPU。"""
import argparse,json,os
from pathlib import Path
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--config",required=True)
 p.add_argument("--action",choices=["preflight","cpgc-source","sep-source","target","audit","report"],default="preflight")
 p.add_argument("--dataset",choices=["cuhk","rstp","icfg"]);p.add_argument("--victim",choices=["irra","rasa"])
 p.add_argument("--method",choices=["cpgc","sep"]);p.add_argument("--scope",choices=["smoke","full"],default="smoke")
 p.add_argument("--allow-gpu-smoke",action="store_true");p.add_argument("--allow-full",action="store_true");p.add_argument("--resume",action="store_true")
 a=p.parse_args()
 from .preflight import run
 role="source" if a.action=="cpgc-source" else "victim"
 c,report=run(a.config,role=role)
 if a.action=="preflight":return 0
 if a.action in ("cpgc-source","sep-source","target"):
  if a.scope=="full" and not a.allow_full:p.error("Full explicit authorization required")
  if a.scope=="smoke" and not a.allow_gpu_smoke:p.error("GPU Smoke explicit authorization required")
  if a.scope=="full":
   # 每个方法的所有数据域/六个Victim先通过真实Smoke和独立CPU审计。
   from .audit import smoke_ready
   smoke_ready(c,a.method or ("cpgc" if a.action=="cpgc-source" else "sep"))
 if a.action=="cpgc-source":
  from .cpgc_source import run
  return run(c,a.dataset,a.scope,resume=a.resume)
 if a.action=="sep-source":
  from .sep_source import run
  return run(c,a.dataset,a.victim,a.scope,resume=a.resume)
 if a.action=="target":
  from .target import evaluate
  return evaluate(c,a.method,a.dataset,a.victim,a.scope,resume=a.resume)
 if a.action=="audit":
  from .audit import audit
  return audit(c,a.method,a.dataset,a.victim,a.scope)
 if a.action=="report":
  from .report import generate
  return generate(c)
 return 0
if __name__=="__main__":raise SystemExit(main())
