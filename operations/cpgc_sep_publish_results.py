"""全部Full与审计完成后的轻量文档发布；不运行模型、不修改执行树或原产物。"""
from pathlib import Path
import json,subprocess,shutil,hashlib
def main():
 root=Path(__file__).resolve().parents[1]
 c=json.loads((root/"configs/framework/cpgc_sep_tbps.shared.server.json").read_text())
 derived=Path(c["output_base"])/"derived_report"
 m=json.loads((derived/"RUN_MANIFEST.json").read_text())
 assert m["Full_pass_count"]==12 and len(m["rows"])==12
 assert all(r["Status"]=="FULL_AND_CPU_AUDIT_PASS" for r in m["rows"])
 commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
 checkout=Path("/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-full-results-20261029")
 assert not checkout.exists(),"publication worktree exists; do not overwrite"
 branch="codex/cpgc-sep-full-results"
 subprocess.run(["git","-C",str(root),"worktree","add","-b",branch,str(checkout),commit],check=True)
 d=checkout/"docs/framework/results/cpgc_sep_tbps_full";d.mkdir()
 for name in ["RESULTS.csv","RESULTS_2DP.csv","CLEAN_BASELINES.csv","RUN_MANIFEST.json"]:
  shutil.copyfile(derived/name,d/name)
 shutil.copyfile(root/"configs/framework/cpgc_sep_tbps.shared.server.json",d/"CONFIGS.json")
 for name in ["CPGC_OFFICIAL_UAP_AUDIT.md","SEP_TBPS_ADAPTATION_AUDIT.md","THREAT_MODEL.md","FAILURES.md"]:
  shutil.copyfile(root/"docs/framework/results/cpgc_sep_tbps"/name,d/name)
 lines=["# C-PGC与SEP三数据集正式Full结果","",
 "12/12组原生Full及独立CPU审计PASS。指标均来自真实Full，未使用Smoke或论文结果填充。","",
 "执行代码Commit："+commit,"",
 "C-PGC仅使用作者ALBEF/Flickr30K epoch39发布UAP，图文Joint；SEP代理为IRRA+RaSa且包含Target，文本检索适配及语义限制见审计文档。保留Raw Clean和Matched Clean，不将前处理差异当作攻击收益。","",
 "| Method | Dataset | Victim | Clean R@1 | Clean R@5 | Clean R@10 | Clean mAP | Attack R@1 | Attack R@5 | Attack R@10 | Attack mAP |",
 "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
 for r in m["rows"]:
  values=[r["Method"],r["Dataset"],r["Victim"]]+[format(r[p+" "+k],".2f") for p in ["Clean","Attack"] for k in ["R@1","R@5","R@10","mAP"]]
  lines.append("| "+" | ".join(values)+" |")
 (d/"REPORT.md").write_text("\n".join(lines)+"\n")
 source={"execution_commit":commit,"config_sha256":hashlib.sha256((root/"configs/framework/cpgc_sep_tbps.shared.server.json").read_bytes()).hexdigest(),"evidence":m["evidence"],"output_base":c["output_base"]}
 (d/"SOURCE_EVIDENCE.json").write_text(json.dumps(source,ensure_ascii=False,indent=2)+"\n")
 for name in ["PROJECT_STATE.md","EXPERIMENT_REGISTRY.md"]:
  with (checkout/"docs"/name).open("a") as f:f.write("\n\n## C-PGC与SEP正式Full完成\n\n12/12原生Full与独立CPU审计PASS，报告与原精度结果见[正式结果](framework/results/cpgc_sep_tbps_full/REPORT.md)。执行版本"+commit+"；发布在独立工作树，不修改原始产物。\n")
 subprocess.run(["git","-C",str(checkout),"add","--","docs/framework/results/cpgc_sep_tbps_full","docs/PROJECT_STATE.md","docs/EXPERIMENT_REGISTRY.md"],check=True)
 subprocess.run(["git","-C",str(checkout),"diff","--cached","--check"],check=True)
 subprocess.run(["git","-C",str(checkout),"-c","user.name=w-harda","-c","user.email=wharda@163.com","commit","-m","Report twelve official C-PGC and SEP TBPS Full evaluations"],check=True)
 subprocess.run(["git","-C",str(checkout),"push","-u","origin",branch],check=True,timeout=120)
 head=subprocess.check_output(["git","-C",str(checkout),"rev-parse","HEAD"],text=True).strip()
 remote=subprocess.check_output(["git","-C",str(checkout),"ls-remote","origin","refs/heads/"+branch],text=True).split()[0]
 assert head==remote
 print("FULL_RESULTS_PUBLISHED",head,flush=True)
if __name__=="__main__":main()
