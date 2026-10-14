"""完整CUHK test的选择价值评测。默认只preflight；full必须显式授权，支持逐图恢复。"""
import argparse,sys,time,copy,csv,json,contextlib,os
from pathlib import Path
import core as c
def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--stage",choices=("preflight","prepare","attack","evaluate","all"),default="preflight")
 p.add_argument("--scope",choices=("smoke","full"),default="smoke")
 p.add_argument("--allow-full",action="store_true")
 p.add_argument("--output",type=Path)
 return p
def dataset():
 m=c.manifest(c.load(c.RAW))
 if (m["gallery_count"],m["query_count"])!=(3074,6156):raise ValueError("official test counts differ")
 wanted={r["row_id"]:r for r in m["queries"]};labels={}
 for line in c.LABELS.open(encoding="utf-8"):
  a=json.loads(line)
  if a["row_id"] not in wanted:continue
  q=wanted[a["row_id"]]
  if a["row_id"] in labels or a["caption"]!=q["caption"] or a["image"]!=m["gallery"][q["paired_index"]]["image"] or a["split"]!="test":raise ValueError("test provenance/coverage mismatch")
  labels[a["row_id"]]=a
 if set(labels)!=set(wanted):raise ValueError("Stage01 test labels incomplete")
 return m,labels
def scope_manifest(m,scope):
 if scope=="full":return m
 # 固定annotation前两张，不按score或攻击效果选smoke。
 small=copy.deepcopy(m);small["gallery"]=m["gallery"][:2]
 small["queries"]=[q for q in m["queries"] if q["paired_index"]<2]
 small.update(gallery_count=2,query_count=len(small["queries"]),all_test_images_included=False,all_test_captions_included=False)
 return small
def enter(out,protocol):
 c.safe(out).mkdir(parents=True,exist_ok=True)
 file=out/"protocol.json"
 if file.exists():
  if c.load(file)!=protocol:raise ValueError("output protocol/source changed; use new output")
 else:c.save(file,protocol)
def finish(out,stage,m,scope,extra):
 c.save(out/(stage+"_completed.json"),{"method_audit_passed":True,"scope":scope,"gallery_count":m["gallery_count"],"query_count":m["query_count"],**extra})
def prepare(out,m,labels,protocol,rt):
 torch=rt.torch;cache=out/"clean_features.pt"
 if (out/"prepare_completed.json").exists():
  verify_prepared(out,m,protocol);print("PREPARE_ALREADY_COMPLETE",flush=True);return
 victim=rt.setup();versions={n:p._version for n,p in victim.named_parameters()}
 if cache.exists():features=torch.load(cache,map_location="cpu",weights_only=True)
 else:
  gf=[];qf=[]
  with torch.no_grad():
   # 每图独立forward，与TTA clean输入相同批大小；缓存仅此新test来源。
   for i,row in enumerate(m["gallery"]):
    gf.append(victim.encode_images(rt.pixels(victim,row)).cpu())
    if i%100==0:print("CLEAN_IMAGE",i,m["gallery_count"],flush=True)
   for start in range(0,m["query_count"],64):
    qf.append(victim.encode_texts([q["caption"] for q in m["queries"][start:start+64]]).cpu())
  features={"image":torch.cat(gf),"text":torch.cat(qf)}
  rt.tensor_save(cache,features)
 rt.normalized(features["image"],m["gallery_count"]);rt.normalized(features["text"],m["query_count"])
 reader,bases,localizer=rt.fresh_localizer(victim);rs={k:v.detach().cpu().clone() for k,v in reader.state_dict().items()}
 gallery=features["image"].cuda();items=[];new=0;reused=0;t=time.time()
 for row in m["gallery"]:
  path=out/"scores"/f'{row["index"]:06d}.json';ih=c.sha(c.IMAGES/row["image"])
  if path.exists():
   saved=c.load(path);record=saved["record"];payload=saved["payload"]
   c.validate_score(record,payload,row,rt.selection)
   if saved["protocol_sha256"]!=c.identity(protocol) or saved["image_sha256"]!=ih:raise ValueError("resume score provenance")
   reused+=1
  else:
   record,payload=rt.score_one(victim,reader,localizer,bases,gallery,row,labels[row["row_id"]])
   c.save(path,{"protocol_sha256":c.identity(protocol),"image_sha256":ih,"record":record,"payload":payload});new+=1
  items.append({"index":row["index"],"row_id":row["row_id"],"image":row["image"],"pid":row["pid"],
   "natural_k":payload["natural_k"],"candidate_count":payload["candidate_count"],"selection_ids":payload["selection_ids"],
   "set_overlap":payload["set_overlap"],"score_path":str(path),"score_sha256":c.sha(path),"image_sha256":ih})
  print("SCORE_READY",row["row_id"],"K",payload["natural_k"],"m",payload["candidate_count"],flush=True)
 rt.assert_frozen(victim,versions,reader,rs);c.verify(protocol["frozen_sha256"])
 manifest={**m,"rows":items,"inventory":c.inventory(items),"clean_features_sha256":c.sha(cache),
 "protocol_sha256":c.identity(protocol),"source_annotation_sha256":c.sha(c.RAW),
 "estimated_attack_hours_at_0_8_seconds_per_step":c.estimated_hours(c.inventory(items))}
 c.save(out/"selection_manifest.json",manifest)
 finish(out,"prepare",m,protocol["scope"],{"score_count":len(items),"new_backwards_this_invocation":new,
 "reused_scores":reused,"each_conditioning_caption_one_clean_backward":True,"IRRA_reader_frozen":True,
 "selection_manifest_sha256":c.sha(out/"selection_manifest.json"),"seconds":time.time()-t})
 print("PREPARE_COMPLETE",json.dumps(manifest["inventory"]),flush=True)
def verify_prepared(out,m,protocol):
 done=c.load(out/"prepare_completed.json");c.require_complete(done,m,protocol["scope"])
 manifest=c.load(out/"selection_manifest.json")
 if done["selection_manifest_sha256"]!=c.sha(out/"selection_manifest.json") or manifest["protocol_sha256"]!=c.identity(protocol):raise ValueError("prepare manifest changed")
 if manifest["gallery"]!=m["gallery"] or manifest["queries"]!=m["queries"]:raise ValueError("dataset changed")
 if len(manifest["rows"])!=m["gallery_count"] or [r["index"] for r in manifest["rows"]]!=list(range(m["gallery_count"])):raise ValueError("incomplete score cache")
 if c.sha(out/"clean_features.pt")!=manifest["clean_features_sha256"]:raise ValueError("clean cache changed")
 c.verify({row["score_path"]:row["score_sha256"] for row in manifest["rows"]})
 return manifest
def attack(out,m,protocol,rt):
 torch=rt.torch;manifest=verify_prepared(out,m,protocol)
 if (out/"attack_completed.json").exists():
  verify_attacked(out,m,protocol);print("ATTACK_ALREADY_COMPLETE",flush=True);return
 victim=rt.setup();source=rt.IRRASurrogate(victim);versions={n:p._version for n,p in victim.named_parameters()}
 cache=torch.load(out/"clean_features.pt",map_location="cpu",weights_only=True);gallery=cache["image"].cuda()
 bases=c.load(rt.score.BASIS);indices=[];t=time.time();steps=0;rounds=0
 for row,entry in zip(m["gallery"],manifest["rows"]):
  file=out/"attacks"/f'{row["index"]:06d}.json'
  if c.sha(c.IMAGES/row["image"])!=entry["image_sha256"]:raise ValueError("source image changed")
  if file.exists():
   result=c.load(file);validate_attack(result,row,entry,out,rt)
   indices.append({"index":row["index"],"path":str(file),"sha256":c.sha(file)});print("ATTACK_RESUME",row["row_id"],flush=True);continue
  saved=c.load(entry["score_path"]);record=saved["record"];payload=saved["payload"]
  if payload["selection_ids"]!=entry["selection_ids"] or payload["natural_k"]!=entry["natural_k"]:raise ValueError("cached selection changed")
  # 只读取prepare固定的K/序列/S/w/e；此阶段不调用scorer/selector。
  image=rt.pixels(victim,row)
  with torch.enable_grad():
   results,images,checks=rt.run_three(victim,source,image,record,payload,bases,gallery,1 if protocol["scope"]=="smoke" else None)
  archives={}
  with torch.no_grad():
   for mode in c.MODES:
    final=images[mode];cpu=final.cpu()
    path=out/"gallery"/mode/f'{row["index"]:06d}.pt'
    feature=source.encode_images(final).cpu()
    # float32归一化前像素精确保留，不做PNG/JPEG量化。
    rt.tensor_save(path,{"pixels":cpu,"IRRA_feature":feature})
    restored=torch.load(path,map_location="cpu",weights_only=True)
    if not torch.equal(restored["pixels"],cpu) or not torch.equal(restored["IRRA_feature"],feature):raise ValueError("pixel archive roundtrip")
    if payload["natural_k"] and rt.tensor_hash(cpu)!=results[mode]["rounds"][-1]["post_projection_image_sha256"]:raise ValueError("final image capture differs from actual TTA endpoint")
    archives[mode]={"path":str(path),"sha256":c.sha(path),"pixel_sha256":rt.tensor_hash(cpu),
                   "linf":float((final-image).abs().max()),"feature_shape":list(feature.shape)}
  result={"index":row["index"],"row_id":row["row_id"],"image":row["image"],"pid":row["pid"],
    "natural_k":payload["natural_k"],"candidate_count":payload["candidate_count"],"selection_ids":payload["selection_ids"],
    "score_sha256":entry["score_sha256"],"protocol_sha256":c.identity(protocol),
    "modes":results,"archives":archives,"checks":checks}
  validate_attack(result,row,entry,out,rt);c.save(file,result)
  indices.append({"index":row["index"],"path":str(file),"sha256":c.sha(file)})
  for mode in c.MODES:
   if results[mode]["execution_reuse_of"] is None:
    rounds+=len(results[mode]["rounds"]);steps+=sum(len(e["steps"]) for e in results[mode]["rounds"])
  rt.assert_frozen(victim,versions)
  print("ATTACK_DONE",row["index"],row["row_id"],"K",payload["natural_k"],"elapsed",time.time()-t,flush=True)
 rt.assert_frozen(victim,versions);c.verify(protocol["frozen_sha256"])
 c.save(out/"gallery_index.json",{"rows":indices,"shared":True,"gallery_count":len(indices),"manifest_sha256":c.sha(out/"selection_manifest.json")})
 finish(out,"attack",m,protocol["scope"],{"IRRA_frozen":True,"score_backwards_during_attack":0,"caption_loss_terms":0,
  "same_K":True,"source_SHA_unchanged":True,"shared_adversarial_gallery":True,
  "index_sha256":c.sha(out/"gallery_index.json"),"physical_rounds_this_invocation":rounds,"actual_steps_this_invocation":steps,
  "seconds_this_invocation":time.time()-t})
 print("ATTACK_COMPLETE",flush=True)
def validate_attack(result,row,entry,out,rt):
 if (result["index"],result["row_id"],result["image"],result["pid"])!=(row["index"],row["row_id"],row["image"],row["pid"]):raise ValueError("attack identity mismatch")
 if result["score_sha256"]!=entry["score_sha256"] or result["selection_ids"]!=entry["selection_ids"]:raise ValueError("attack score/sequence mismatch")
 if result["protocol_sha256"]!=c.identity(c.load(out/"protocol.json")):raise ValueError("attack protocol mismatch")
 limit=1 if c.load(out/"protocol.json")["scope"]=="smoke" else 10
 for mode in c.MODES:
  a=result["archives"][mode];p=Path(a["path"])
  if p.resolve()!= (out/"gallery"/mode/f'{row["index"]:06d}.pt').resolve() or c.sha(p)!=a["sha256"]:raise ValueError("archive path/SHA")
  if a["linf"]>8/255+1e-6:raise ValueError("saved budget")
  r=result["modes"][mode]
  if r["fixed_sequence"]!=entry["selection_ids"][mode] or len(r["rounds"])!=entry["natural_k"]:raise ValueError("saved K")
  if any(len(e["steps"])!=limit for e in r["rounds"]):raise ValueError("saved steps")
 if entry["natural_k"] and not result["checks"]["noise_transform_RNG_exact"]:raise ValueError("random control")
def verify_attacked(out,m,protocol):
 manifest=verify_prepared(out,m,protocol);done=c.load(out/"attack_completed.json");c.require_complete(done,m,protocol["scope"])
 if c.sha(out/"gallery_index.json")!=done["index_sha256"]:raise ValueError("gallery index changed")
 idx=c.load(out/"gallery_index.json")
 if idx["manifest_sha256"]!=c.sha(out/"selection_manifest.json") or len(idx["rows"])!=m["gallery_count"] or [r["index"] for r in idx["rows"]]!=list(range(m["gallery_count"])):raise ValueError("not full unique gallery")
 c.verify({r["path"]:r["sha256"] for r in idx["rows"]})
 return manifest,idx
def evaluate(out,m,protocol,rt):
 torch=rt.torch;manifest,idx=verify_attacked(out,m,protocol)
 # 必须实际加载每张共享对抗图的真实编码特征；不得回填paired cosine。
 galleries={mode:[] for mode in c.MODES};max_budget=0.
 for row,entry,ref in zip(m["gallery"],manifest["rows"],idx["rows"]):
  data=c.load(ref["path"]);validate_attack(data,row,entry,out,rt)
  for mode in c.MODES:
   archive=data["archives"][mode];value=torch.load(archive["path"],map_location="cpu",weights_only=True)
   px=value["pixels"];feature=value["IRRA_feature"]
   if px.shape!=(1,3,384,128) or px.dtype!=torch.float32 or not torch.isfinite(px).all() or px.min()<0 or px.max()>1 or rt.tensor_hash(px)!=archive["pixel_sha256"]:raise ValueError("saved pixels invalid")
   rt.normalized(feature,1);galleries[mode].append(feature);max_budget=max(max_budget,archive["linf"])
 clean=torch.load(out/"clean_features.pt",map_location="cpu",weights_only=True)
 galleries={mode:torch.cat(v) for mode,v in galleries.items()}
 galleries["CLEAN"]=clean["image"];query=clean["text"]
 rt.normalized(query,m["query_count"])
 gids=[g["pid"] for g in m["gallery"]];qids=[q["pid"] for q in m["queries"]]
 allrows={}
 for mode,features in galleries.items():
  rt.normalized(features,m["gallery_count"]);rows=[]
  for start in range(0,m["query_count"],128):
   scores=query[start:start+128]@features.T
   _,part=rt.text_to_image_metrics(scores,qids[start:start+128],gids)
   rows.extend({**q,**metric} for q,metric in zip(m["queries"][start:start+128],part))
  allrows[mode]=rows
 metrics={mode:rt.aggregate_rows(rows) for mode,rows in allrows.items()}
 contract={"task":"TBPS text-to-image","dataset":"CUHK-PEDES","query_scope":"entire official test" if protocol["scope"]=="full" else "two-image smoke only",
 "query_set_sha256":c.identity(m["queries"]),"gallery_layout_sha256":c.identity(m["gallery"]),
 "attack_set_id":c.sha(out/"gallery_index.json"),"protocol_kind":"shared_adversarial_gallery",
 "conditioning_rule":m["conditioning_rule"],"shared_adversarial_gallery":True,"modes":list(c.MODES),
 "relevance":"all same-PID gallery images; no camera exclusion"}
 report={"victim_id":"IRRA-official-CUHK-fp32","aggregation_contract":contract,"clean":metrics["CLEAN"],
 "adversarial":{mode:metrics[mode] for mode in c.MODES}}
 aggregation=rt.aggregate_victims([report])
 summary={"scope":protocol["scope"],"formal_test_evaluation":protocol["scope"]=="full","shared_adversarial_gallery":True,
  "query_count":m["query_count"],"gallery_count":m["gallery_count"],"victim_count":1,
  "clean":metrics["CLEAN"],"adversarial":report["adversarial"],"aggregate":aggregation,"inventory":manifest["inventory"]}
 c.save(out/"victim_summary.json",report);c.save(out/"summary.json",summary)
 with c.safe(out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
  writer=csv.writer(f);writer.writerow(["mode","clean_mAP","adversarial_mAP","aAP","mDR","R@1","R@5","R@10","mINP"])
  for mode in c.MODES:
   a=aggregation["modes"][mode];v=metrics[mode]
   writer.writerow([mode,metrics["CLEAN"]["mAP"],v["mAP"],a["aAP_adversarial"],a["mDR"],v["R@1"],v["R@5"],v["R@10"],v["mINP"]])
 with c.safe(out/"per_query_metrics.jsonl").open("w",encoding="utf-8") as f:
  for i,q in enumerate(m["queries"]):f.write(json.dumps({**q,"clean":allrows["CLEAN"][i],"adversarial":{mode:allrows[mode][i] for mode in c.MODES}},ensure_ascii=False)+"\n")
 lines=["# 完整test共享对抗图库评测" if protocol["scope"]=="full" else "# 两图smoke（不是正式结果）","",
 f'queries={m["query_count"]}; gallery={m["gallery_count"]}; one victim IRRA fp32; aAP=mAP。',"",
 "| mode | clean mAP | adv mAP / aAP | mDR |","|---|---:|---:|---:|"]
 for mode in c.MODES:
  a=aggregation["modes"][mode];lines.append(f'| {mode} | {metrics["CLEAN"]["mAP"]:.6f} | {a["aAP_adversarial"]:.6f} | {a["mDR"]:.6f} |')
 lines+=["","每张图库图像使用固定第一条caption作为攻击条件；所有查询使用同一完整图库。单固定随机种子，不声称跨种子稳定。无属性候选图像保持clean；全选集合相同保留。没有G训练或文本攻击。",
 "full的scoring沿用冻结paired-soft-rank公式，竞争图库由旧train34054改为test3074；只在准备阶段计算。不能把旧100句的S缓存视为test评分。"]
 c.safe(out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 c.verify(protocol["frozen_sha256"]);rt.frozen.assert_formal_clean()
 finish(out,"evaluate",m,protocol["scope"],{"shared_adversarial_gallery":True,"maximum_linf":max_budget,"formal_test_evaluation":protocol["scope"]=="full"})
 finish(out,"completed",m,protocol["scope"],{"shared_adversarial_gallery":True,"formal_test_evaluation":protocol["scope"]=="full",
 "maximum_linf":max_budget,"score_backward_during_attack":0,"caption_loss_terms":0,"source_SHA_unchanged":True})
 # finish生成completed_completed，通用完成文件另存明确命名。
 os.replace(out/"completed_completed.json",out/"completed.json")
 print("COMPLETE",json.dumps(summary),flush=True)
def main(argv=None):
 args=parser().parse_args(argv)
 if args.scope=="full" and args.stage!="preflight" and not args.allow_full:raise SystemExit("--allow-full required; user-run only")
 m,labels=dataset();print("TEST_PREFLIGHT",m["gallery_count"],m["query_count"],"Stage01 span source complete",flush=True)
 if args.stage=="preflight":return
 import runtime as rt
 rt.frozen.assert_formal_clean();locked=rt.locks()
 if args.scope=="full":
  smoke=c.ROOT/"outputs/smoke"
  done=c.load(smoke/"completed.json")
  if done.get("scope")!="smoke" or not done.get("method_audit_passed"):raise ValueError("successful two-image smoke required")
  c.verify(c.load(smoke/"protocol.json")["frozen_sha256"])
 m=scope_manifest(m,args.scope);out=c.safe(args.output or c.ROOT/"outputs"/args.scope)
 protocol={"schema":"full-test-shared-gallery-v1","scope":args.scope,"dataset":"CUHK-PEDES","split":"test",
 "query_count":m["query_count"],"gallery_count":m["gallery_count"],"conditioning_rule":m["conditioning_rule"],
 "AttributeInstance":"frozen v5.5","score":"frozen Step5.1 S_raw=||Q^T w^T G||_2",
 "score_gallery":"clean official test gallery; paired index of conditioned source image",
 "score_gallery_changed_from_old_train_diagnostic":True,"gamma":3,"beta":1,"kmax":None,"tau":.02,
 "scheduler":"unchanged equal-weight PREFIX_BASELINE","modes":list(c.MODES),"steps_per_round":10,
 "scales":[.5,.75,1.25,1.5],"transforms_per_scale":6,"momentum":.9,"step_size":2/255,
 "epsilon":8/255,"precision":"fp32","caption_loss_terms":0,"score_weighting":False,"priority_projection":False,
 "random_seed":42,"random_selection":"same per-caption SHA256 seed as Step7.6; one draw without replacement",
 "score_backward_during_attack":0,"no_training":True,"pixel_archive":"float32 model-input pixels; no lossy quantization",
 "smoke_observed_steps_per_round":1 if args.scope=="smoke" else None,
 "torch_version":str(rt.torch.__version__),"frozen_sha256":locked}
 enter(out,protocol)
 # 防止两个进程同时准备/攻击同一输出；退出后可安全续跑。
 import fcntl
 with c.safe(out/".run.lock").open("a") as f:
  fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  if args.stage in ("all","prepare"):prepare(out,m,labels,protocol,rt)
  if args.stage in ("all","attack"):attack(out,m,protocol,rt)
  if args.stage in ("all","evaluate"):evaluate(out,m,protocol,rt)
if __name__=="__main__":main()
