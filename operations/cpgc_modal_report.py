"""两Victim均通过独立审计后生成论文表；可在独立文档树发布，不修改执行源码树。"""
from pathlib import Path
import argparse,csv,io,json,os,subprocess
from dataclasses import asdict
from tbps.artifacts import FileRef,checked,sha_file,json_bytes
from tbps.baselines.cpgc_sep.modal_ablation import settings,MODES,METRICS
from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore
def collect(config_path):
 c,old=settings(config_path);rows=[];evidence={}
 for victim in ("irra","rasa"):
  root=Path(c["output_base"])/victim;ar=Path(c["output_base"])/"audit"/victim
  s=VerifiedArtifactStore(root,new_base=c["output_base"]);a=VerifiedArtifactStore(ar,new_base=c["output_base"])
  s.verify("completed");a.verify("completed")
  summary=json.loads((root/"completed/summary.json").read_text());audit=json.loads((ar/"completed/audit.json").read_text())
  if audit["status"]!="PASS" or audit["groups"]!=summary["groups"] or audit["result"]!={"path":str(root/"completed/manifest.json"),"sha256":sha_file(root/"completed/manifest.json")}:raise RuntimeError("independent audit binding")
  for mode in MODES:rows.append({"Victim":victim.upper() if victim=="irra" else "RaSa","Mode":mode,**summary["groups"][mode]})
  evidence[victim]={"Target":asdict(FileRef(str(root/"completed/manifest.json"),sha_file(root/"completed/manifest.json"))),
   "Audit":asdict(FileRef(str(ar/"completed/manifest.json"),sha_file(ar/"completed/manifest.json"))),
   "historical_Target":c["targets"][victim]["manifest"],"historical_Audit":c["audits"][victim]["manifest"],
   "elapsed_seconds":summary["elapsed_seconds"],"peak_GPU_reserved_bytes":summary["peak_GPU_reserved_bytes"],
   "native_pairs":summary["native_pairs"],"neural_counts":summary["neural_counts"],"history_unchanged":summary["history_unchanged"]}
 return c,rows,evidence
def write_report(dest,c,rows,evidence,code_commit):
 dest.mkdir(parents=True,exist_ok=True)
 for rounded,name in ((False,"METRICS.csv"),(True,"METRICS_2DP.csv")):
  f=io.StringIO();w=csv.writer(f,lineterminator="\n");w.writerow(("Victim","Mode",*METRICS))
  for r in rows:w.writerow((r["Victim"],r["Mode"],*(format(r[k],".2f") if rounded else r[k] for k in METRICS)))
  (dest/name).write_text(f.getvalue())
 table="| Victim | Mode | R@1 | R@5 | R@10 | mAP |\n| --- | --- | ---: | ---: | ---: | ---: |\n"
 for r in rows:table+="| "+r["Victim"]+" | "+r["Mode"]+" | "+" | ".join(format(r[k],".2f")for k in METRICS)+" |\n"
 rasa={r["Mode"]:r for r in rows if r["Victim"]=="RaSa"}
 lower=min(("Image-only","Text-only"),key=lambda m:rasa[m]["mAP"])
 joint_all=all(rasa["Joint"][k]<min(rasa["Image-only"][k],rasa["Text-only"][k]) for k in ("R@1","mAP"))
 text=f"""# C-PGC CUHK-PEDES单模态消融正式结果

两Victim新增Image-only、Text-only完整评测及各自独立CPU审计均PASS；3074 Gallery、6156 Query。
Matched Clean与Joint为历史Full只读引用，已从历史完整评分矩阵重新计算指标并与原CSV、summary、审计逐值核对。

{table}

## 实验解释

RaSa单模态中mAP更低的是{lower}。Joint在R@1与mAP上均低于两种单模态：{joint_all}。
这里只比较冻结攻击产物的模态组合；不把该观察解释为未经验证的训练或机制因果。

## 输入和评分协议

- Matched Clean：clean_gallery + matched_text。
- Image-only：adv_gallery + matched_text，明确不使用Raw Caption的clean_text。
- Text-only：clean_gallery + adv_text。
- Joint：adv_gallery + adv_text。
- C-PGC官方ALBEF/Flickr30K Epoch39 UAP、FP32 12/255原始语义、原单词替换产物保持。
- 原Victim Checkpoint、Tokenizer、完整Gallery、Query顺序和PID保持；新攻击生成、图文编码、训练均0。
- IRRA原生评分，RaSa FP32、全3074 Gallery全局Top128、全部128个候选原生ITM融合，fusion_batch16、query_chunk32。
- 原Matched Clean与Joint各前32条Query额外重放评分，逐元素exact后才执行新增两模式；不重跑原Full。
- 新两模式完整矩阵和逐块分数均保存。独立CPU审计验证SHA、Manifest、身份、完整排名、指标和历史保护。

## 来源

执行Commit：{code_commit}；配置SHA：{c['sha256']}。
原Full根目录：/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v2/full/cpgc/cuhk/。
新结果：{c['output_base']}。
来源及资源详见SOURCE_EVIDENCE.json。旧Full与SEP执行链未修改。
"""
 (dest/"REPORT.md").write_text(text)
 (dest/"SOURCE_EVIDENCE.json").write_bytes(json_bytes({"status":"PASS","code_commit":code_commit,
  "config":{"path":c["path"],"sha256":c["sha256"]},"parent_config":c["parent_config"],"victims":evidence,
  "Dataset":{"Gallery":3074,"Query":6156,"split":"CUHK-PEDES official Test"},"modes":c["modes"]}))
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--publish",action="store_true");a=p.parse_args()
 c,rows,evidence=collect(a.config);root=Path(__file__).resolve().parents[1]
 commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
 if a.publish:
  tree=Path("/home/lzf/ldx/projects/idea-test-tbps-cpgc-cuhk-modal-results-20261101")
  subprocess.run(["git","-C",str(root),"worktree","add","-b","codex/cpgc-cuhk-modal-results",str(tree),commit],check=True)
  dest=tree/"docs/framework/results/cpgc_cuhk_modal_ablation"
  write_report(dest,c,rows,evidence,commit)
  subprocess.run(["git","diff","--check"],cwd=tree,check=True)
  subprocess.run(["git","add",str(dest)],cwd=tree,check=True)
  subprocess.run(["git","-c","user.name=w-harda","-c","user.email=wharda@163.com","commit","-m","Report audited CPGC CUHK image and text modality ablations"],cwd=tree,check=True)
  subprocess.run(["git","push","-u","origin","codex/cpgc-cuhk-modal-results"],cwd=tree,check=True)
 else:write_report(Path(c["output_base"])/"report",c,rows,evidence,commit)
 print("CPGC_MODAL_REPORT_PASS",json.dumps(rows),flush=True)
if __name__=="__main__":main()
