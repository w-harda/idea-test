"""只汇总实际完成结果，不用论文或Smoke指标填充正式表。"""
from pathlib import Path
import csv,io,json
from tbps.artifacts import ArtifactStore,json_bytes
from .runtime import ref
from .target import METRICS
METHODS={"cpgc":"CPGC_OFFICIAL_UAP_TBPS_JOINT","sep":"SEP_TBPS_IRRA_RASA_ENSEMBLE"}
def generate(c,report_dir=None):
 O=Path(report_dir) if report_dir else Path(c["output_base"])/"derived_report"
 O.mkdir(parents=True,exist_ok=True)
 rows=[];clean=[];evidence=[]
 for method in ("cpgc","sep"):
  for ds in ("cuhk","rstp","icfg"):
   for victim in ("irra","rasa"):
    root=Path(c["output_base"])/"full"/method/ds/victim
    a=Path(c["output_base"])/"full/audit"/method/ds/victim
    row={"Method":METHODS[method],"Dataset":ds,"Victim":victim}
    for pref in ("Clean","Attack","Matched Clean"):
     for k in METRICS:row[pref+" "+k]=""
    row["Status"]="PENDING"
    if (root/"completed/manifest.json").exists() and (a/"completed/manifest.json").exists():
     ArtifactStore(root,new_base=c["output_base"]).verify("completed")
     ArtifactStore(a,new_base=c["output_base"]).verify("completed")
     r=json.loads((root/"completed/summary.json").read_text())
     ar=json.loads((a/"completed/audit.json").read_text())
     assert r["scope"]==ar["scope"]=="full" and r["status"]==ar["status"]=="PASS"
     attack="Joint" if method=="cpgc" else "Text-only"
     for pref,mode in [("Clean","Clean"),("Attack",attack),("Matched Clean","Matched Clean")]:
      for k in METRICS:row[pref+" "+k]=r["groups"][mode][k]
     row["Status"]="FULL_AND_CPU_AUDIT_PASS"
     evidence.append({"result":ref(root/"completed/manifest.json").path,"result_sha256":ref(root/"completed/manifest.json").sha256,"audit":ref(a/"completed/manifest.json").path,"audit_sha256":ref(a/"completed/manifest.json").sha256})
     for mode in ("Clean","Matched Clean"):
      clean.append({"Method":METHODS[method],"Dataset":ds,"Victim":victim,"Control":mode,**r["groups"][mode]})
    elif (root/"FAIL.json").exists():row["Status"]="FAIL"
    elif (root/"started.json").exists():row["Status"]="RUNNING"
    elif method=="sep" and c["sep_protocol"].get("status","").startswith("BLOCKED"):row["Status"]=c["sep_protocol"]["status"]
    rows.append(row)
 def write_csv(path,values,fields):
  with path.open("w",newline="",encoding="utf-8-sig") as f:
   w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(values)
 write_csv(O/"RESULTS.csv",rows,list(rows[0]))
 display=[{k:format(v,".2f") if isinstance(v,float) else v for k,v in r.items()} for r in rows]
 write_csv(O/"RESULTS_2DP.csv",display,list(rows[0]))
 write_csv(O/"CLEAN_BASELINES.csv",clean,["Method","Dataset","Victim","Control",*METRICS])
 (O/"RUN_MANIFEST.json").write_text(json.dumps({"config":c["path"],"config_sha256":c["sha256"],"rows":rows,"evidence":evidence,"Full_pass_count":sum(r["Status"]=="FULL_AND_CPU_AUDIT_PASS" for r in rows)},ensure_ascii=False,indent=2)+"\n")
 print("RESULTS_SUMMARY",json.dumps({"Full_pass_count":sum(r["Status"]=="FULL_AND_CPU_AUDIT_PASS" for r in rows),"rows":[{k:r[k] for k in ["Method","Dataset","Victim","Status"]} for r in rows]}),flush=True)
 return 0
