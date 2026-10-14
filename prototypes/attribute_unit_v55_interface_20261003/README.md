# Stage03 v5.5 AttributeInstance prototype

仅服务器独立接口实现，不修改正式Stage01/Stage03/Top-K/G。固定12 semantic axes，source-only依据现有Stage01 provenance，不做字符串重新定位/raw-only提取。正文见outputs/smoke/summary.md。

提供convert_record(rec)->List[AttributeInstance]、CachedReader.attach、text_candidates、dynamic_topk([(instance,S)], beta)、attack_payload。score(a)明确NotImplementedError。实例semantic representation为结构化e，分别保存512D projected_context与已有value_centroid、basis/rank/reader-source metadata；S和攻击方向尚未定义。原正式Top-K接受旧slot/int；prototype封装其相同entropy/effective-count规则，整数回归通过，浮点与多实例输入仅在adapter处理，不修改正式函数。

已运行28tests及16固定validation smoke；原9句仅source-span检查（不在feature cache）。没有full。full须用户手动：

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_unit_v55_interface_20261003
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python run_interface.py --mode full --allow-full-validation > outputs/full.log 2>&1
```

输出outputs/full/。CLI默认smoke，full显式flag才允许运行；已完成output目录拒绝覆盖。仅13700 validation既有features，不训练/IRRA重提/retrieval。train basis仅从已验证cached centroid充分统计量按value归并，value centroid等权中心化SVD，无learned/random projection。原输入/checkpoint/cache hashes前后检查；prototype源码变更不能静默替换出处锁。

Full输出coverage.json、roundtrip.json、semantic_bases.json、semantic_diagnostic.json、instances.jsonl、representations.fp32.npy、heldout_source_span_audit.json、reader_regression.json、protocol.json、final_audit.json、summary.md、completed.json。

instances.jsonl表示数组通过feature_index引用representations.fp32.npy，shape[N,2,512]，第二维依次projected_context/value_centroid。raw source spans保存Python字符[start,end)，原文未规范化。unresolved不得进入文本候选；语义缺失/截断另记representation_status。多个实例可共享同一个source span，文本消费者需考虑共享编辑；不自动重新绑定或修改原caption。

全量结果审阅前不冻结正式Unit；停止ontology扩展是本轮边界，不代表六轴语义验证已自动完成。
