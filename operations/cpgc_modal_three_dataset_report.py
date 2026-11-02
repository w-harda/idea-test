"""C-PGC三数据集消融的只读来源核验、24行汇总和隔离发布。"""
from pathlib import Path
from dataclasses import asdict
import json,csv,io,subprocess
from tbps.artifacts import FileRef,checked,sha_file,json_bytes
from tbps.baselines.cpgc_sep.modal_ablation import MODES,METRICS,settings,check_store,csv_bytes,dimensions
from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore
from tbps.baselines.cpgc_sep.dataset import require
LABELS={"cuhk":"CUHK-PEDES","rstp":"RSTPReid","icfg":"ICFG-PEDES"}
ORDER=(("rstp","irra"),("rstp","rasa"),("icfg","irra"),("icfg","rasa"))
def file_ref(path):return {"path":str(Path(path).resolve()),"sha256":sha_file(path)}
def load_plan(path):
 p=Path(path).resolve();c=json.loads(p.read_text())
 require(c["schema"]=="tbps-cpgc-three-dataset-modality-1","plan schema")
 require(tuple((x["dataset"],x["victim"])for x in c["cases"])==ORDER,"fixed four-case order")
 for ref in c["producers"]:checked(FileRef(**ref))
 for case in c["cases"]:
  checked(FileRef(**case["config"]));v,old=settings(case["config"]["path"])
  require(v["dataset"]==case["dataset"] and tuple(v["targets"])==(case["victim"],),"case identity")
  require(v["output_base"]==c["output_base"],"shared isolated ablation output root")
 c.update(path=str(p),sha256=sha_file(p));return c

def verify_cuhk(plan):
 records={};root=Path(plan["cuhk"]["output_base"])
 own=json.loads(Path(plan["cuhk"]["config"]["path"]).read_text());checked(FileRef(**own["parent_config"]))
 parent=json.loads(Path(own["parent_config"]["path"]).read_text())
 for victim in ("irra","rasa"):
  info=plan["cuhk"][victim]
  checked(FileRef(**info["target"]));checked(FileRef(**info["audit"]))
  target=VerifiedArtifactStore(root/victim,new_base=root);audit=VerifiedArtifactStore(root/"audit"/victim,new_base=root)
  n=check_store(target);check_store(audit)
  s=json.loads((target.root/"completed/summary.json").read_text());a=json.loads((audit.root/"completed/audit.json").read_text())
  require(s["status"]==a["status"]=="PASS" and (s["Gallery"],s["Query"])==(3074,6156),"completed CUHK full")
  require(s["config_sha256"]==plan["cuhk"]["config"]["sha256"] and a["config"]==plan["cuhk"]["config"],"CUHK frozen config identity")
  require(a["CUDA"]is False and a["neural_forward"]==0 and a["groups"]==s["groups"] and a["result"]==info["target"],"CUHK independent audit")
  require(s["model_before"]==s["model_after"] and s["generation"]==s["optimizer"]==s["backward"]==0,"CUHK frozen models")
  require(set(s["groups"])==set(MODES),"CUHK four groups")
  for rounded,name in ((False,"metrics.csv"),(True,"metrics_2dp.csv")):
   require((target.root/"completed"/name).read_bytes()==csv_bytes(victim,s["groups"],rounded),"CUHK CSV summary")
  ck=parent["victims"][victim]["datasets"]["cuhk"]["checkpoint"];checked(FileRef(**ck))
  records[victim]={"checkpoint":ck,"UAP":parent["resources"]["cpgc_noise"],"status":"PASS","groups":s["groups"],"target":info["target"],"audit":info["audit"],"units_verified":n,"recomputed":False}
 return records

def inventory(plan_path,output):
 import torch
 require(not torch.cuda.is_initialized(),"CPU inventory only")
 c=load_plan(plan_path);checked(FileRef(**c["cuhk"]["config"]))
 existing=verify_cuhk(c);cases=[]
 for item in c["cases"]:
  v,old=settings(item["config"]["path"]);victim=item["victim"]
  for typ in ("targets","audits"):
   info=v[typ][victim];checked(FileRef(**info["manifest"]))
   store=VerifiedArtifactStore(info["path"],new_base=old["output_base"]);store.verify("completed")
  t=json.loads((Path(v["targets"][victim]["path"])/"completed/summary.json").read_text())
  a=json.loads((Path(v["audits"][victim]["path"])/"completed/audit.json").read_text())
  require(t["status"]==a["status"]=="PASS" and t["groups"]==a["groups"],"historical Full and audit receipts")
  require((t["Gallery"],t["Query"])==dimensions(v),"historical official dimensions")
  require(a["result"]==v["targets"][victim]["manifest"],"historical audit exact target")
  cases.append({"dataset":item["dataset"],"victim":victim,"parent_config":v["parent_config"],"target":v["targets"][victim],"audit":v["audits"][victim]})
  print("CPGC_MODAL_HISTORY_RECEIPT_PASS",item["dataset"],victim,flush=True)
 p=Path(output);require(not p.exists(),"inventory receipt already exists")
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(json_bytes({"status":"PASS","CUDA":False,"plan":file_ref(plan_path),"cuhk":existing,"cases":cases,
  "scope":"CUHK complete immutable verification + four historical completion receipts; full per-case payload/SHA preflight mandatory before GPU"}))
 print("CPGC_MODAL_THREE_DATASET_INVENTORY_PASS",flush=True)

def collect(plan_path):
 plan=load_plan(plan_path);cuhk=verify_cuhk(plan);rows=[];evidence={"cuhk":cuhk,"new":{}}
 for v in ("irra","rasa"):
  for mode in MODES:rows.append({"Dataset":LABELS["cuhk"],"Victim":v.upper()if v=="irra"else"RaSa","Mode":mode,**cuhk[v]["groups"][mode]})
 for item in plan["cases"]:
  c,old=settings(item["config"]["path"]);ds,v=item["dataset"],item["victim"]
  base=Path(c["output_base"]);target=VerifiedArtifactStore(base/ds/v,new_base=base);aud=VerifiedArtifactStore(base/"audit"/ds/v,new_base=base)
  check_store(target);check_store(aud)
  s=json.loads((target.root/"completed/summary.json").read_text());a=json.loads((aud.root/"completed/audit.json").read_text())
  require(s["status"]==a["status"]=="PASS" and a["groups"]==s["groups"] and a["result"]==file_ref(target.root/"completed/manifest.json"),"final independent CPU audit")
  require(s["dataset"]==ds and s["victim"]==v and (s["Gallery"],s["Query"])==dimensions(c),"final case identity")
  require(a["config"]==item["config"] and a["CUDA"]is False and a["neural_forward"]==0,"audit config and CPU scope")
  for mode in MODES:rows.append({"Dataset":LABELS[ds],"Victim":v.upper()if v=="irra"else"RaSa","Mode":mode,**s["groups"][mode]})
  evidence["new"][ds+"_"+v]={"config":item["config"],"parent_config":c["parent_config"],"historical_target":c["targets"][v],"historical_audit":c["audits"][v],
    "target":file_ref(target.root/"completed/manifest.json"),"audit":file_ref(aud.root/"completed/manifest.json"),
    "checkpoint":old["victims"][v]["datasets"][ds]["checkpoint"],"UAP":old["resources"]["cpgc_noise"],
    "elapsed_seconds":s["elapsed_seconds"],"peak_GPU_reserved_bytes":s["peak_GPU_reserved_bytes"],"peak_RSS_bytes":s["peak_RSS_bytes"],
    "native_TopK":s["native_TopK"],"native_pairs":s["native_pairs"],"neural_counts":s["neural_counts"],"historical_score_parity":s["historical_score_parity"]}
 require(len(rows)==24,"24 complete rows required")
 return plan,rows,evidence

def write_report(dest,plan,rows,evidence,commit):
 dest.mkdir(parents=True,exist_ok=True)
 for rounded,name in ((False,"METRICS.csv"),(True,"METRICS_2DP.csv")):
  f=io.StringIO();w=csv.writer(f,lineterminator="\n");w.writerow(("Dataset","Victim","Mode",*METRICS))
  for r in rows:w.writerow((r["Dataset"],r["Victim"],r["Mode"],*(format(r[k],".2f")if rounded else r[k]for k in METRICS)))
  (dest/name).write_text(f.getvalue())
 table="| Dataset | Victim | Mode | R@1 | R@5 | R@10 | mAP |\n| --- | --- | --- | ---: | ---: | ---: | ---: |\n"
 for r in rows:table+="| "+" | ".join([r["Dataset"],r["Victim"],r["Mode"]]+[format(r[k],".2f")for k in METRICS])+" |\n"
 analysis=[]
 for ds in LABELS.values():
  for v in ("IRRA","RaSa"):
   groups={r["Mode"]:r for r in rows if r["Dataset"]==ds and r["Victim"]==v}
   low_r=min(("Image-only","Text-only"),key=lambda m:groups[m]["R@1"]);low_m=min(("Image-only","Text-only"),key=lambda m:groups[m]["mAP"])
   joint=all(groups["Joint"][k]<min(groups["Image-only"][k],groups["Text-only"][k])for k in("R@1","mAP"))
   analysis.append(f"- {ds} / {v}：单模态R@1较低为{low_r}，mAP较低为{low_m}；Joint在两项上均低于两种单模态：{joint}。")
 text=f"""# C-PGC三个TBPS数据集两Victim单模态消融正式报告

全部24行来自真实Full：CUHK两组已审计历史结果只读复用；RSTPReid与ICFG-PEDES四组新增Full及独立CPU审计全部PASS。
单位为%，原始精度保存在METRICS.csv，展示保留两位小数。

{table}

## 模态表现

{chr(10).join(analysis)}

以上只描述冻结攻击产物的模态组合，不据此推断未经验证的训练机制或因果。

## 协议

- Matched Clean=clean_gallery+matched_text；Image-only=adv_gallery+matched_text；Text-only=clean_gallery+adv_text；Joint=adv_gallery+adv_text。
- 沿用成功CUHK消融的Matched Manifest、原Canonical输入身份及缓存绑定，不使用clean_text代替matched_text。
- 官方ALBEF/Flickr30K Epoch39 UAP及12/255原始浮点语义、单词替换、历史缩放/裁剪/预处理完全不变。
- 不生成攻击图像、文本，不重新编码已有图文特征，Backward与Optimizer均0。
- IRRA原始权重/CLIP Tokenizer/精度/评分；RaSa原Checkpoint/Tokenizer/FP32，全Gallery全局Top128，再全部128候选官方ITM正类logit。未选候选为原协议-100，不宣称其已执行ITM。
- Query Chunk32、Fusion Batch16不变。仅复用已有分块特征与有界8GiB CPU Token缓存，局部Tokens按Top128懒读取，不全量放GPU或RAM。
- RSTP 1000 Gallery/2000 Query；ICFG 19848/19848；CUHK 3074/6156。完整排名和多正例mAP定义不变。
- Matched Clean与Joint仅重读原分数核对历史指标；各组前32 Query的小范围原生评分exact重放仅作接线验收，不重跑原Full。
- 每组新两模式Full后独立CPU进程核验完整矩阵、Chunk拼接、Top128成员、排序、四指标、来源SHA、CSV和历史元数据。通过审计才推进下一组。

## 来源与执行

CUHK优先参考数值源码：7973128ec1fc73ab8e2428dfbdea600a565421a9；已完成CUHK结果报告Commit：{plan['cuhk']['report_commit']}。
本轮执行Commit：{commit}；计划配置SHA：{plan['sha256']}。
四组历史来源严格分离：RSTP/IRRA、RSTP/RaSa、ICFG/IRRA=v2；ICFG/RaSa=v3。Checkpoint、原配置、来源SHA与独立审计见SOURCE_EVIDENCE.json。
新增产物根目录：{plan['output_base']}。

用户授权共享GPU。原SEP进程、代码和输出不修改，不设置固定最低空闲显存阈值；保留主机资源和完整性门禁。真实OOM/来源/数值/审计错误首错停止，不修改参数重跑。

所有结果仅针对C-PGC模态组合，不新增其他攻击或消融。
"""
 (dest/"REPORT.md").write_text(text)
 (dest/"SOURCE_EVIDENCE.json").write_bytes(json_bytes({"status":"PASS","execution_commit":commit,"plan":file_ref(plan["path"]),"protocol":"exact CUHK matched modality extension","evidence":evidence}))
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--plan",required=True);p.add_argument("--inventory");p.add_argument("--publish",action="store_true");a=p.parse_args()
 if a.inventory:return inventory(a.plan,a.inventory)
 plan,rows,evidence=collect(a.plan);root=Path(__file__).resolve().parents[1];commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
 if a.publish:
  tree=Path("/home/lzf/ldx/projects/idea-test-tbps-cpgc-modal-three-datasets-results-20261102")
  subprocess.run(["git","worktree","add","-b","codex/cpgc-modal-three-datasets-results",str(tree),commit],cwd=root,check=True)
  dest=tree/"docs/framework/results/cpgc_modal_three_datasets";write_report(dest,plan,rows,evidence,commit)
  subprocess.run(["git","diff","--check"],cwd=tree,check=True);subprocess.run(["git","add",str(dest)],cwd=tree,check=True)
  subprocess.run(["git","-c","user.name=w-harda","-c","user.email=wharda@163.com","commit","-m","Report audited CPGC modality ablations across three TBPS datasets"],cwd=tree,check=True)
  subprocess.run(["git","push","-u","origin","codex/cpgc-modal-three-datasets-results"],cwd=tree,check=True)
 else:write_report(Path(plan["output_base"])/"report",plan,rows,evidence,commit)
 print("CPGC_THREE_DATASET_MODAL_REPORT_PASS",flush=True)
if __name__=="__main__":main()
