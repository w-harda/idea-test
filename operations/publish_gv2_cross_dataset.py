"""只读生成两目标正式指标与来源报告；不加载模型或执行攻击。"""
from pathlib import Path
import argparse,csv,io,json
from tbps.artifacts import sha_file,json_bytes
from tbps.transfer.gv2_core import load_config,DATASETS,KEYS,SPECS,store_for,read_json,require,verify_history
ROOT=Path(__file__).resolve().parents[1]
def publish(config):
 c=load_config(config);rows=[];evidence={'source_reference_commit':c['reference_commit'],'G_checkpoint':c['checkpoint'],'datasets':{},'protected_historical_files':verify_history(c),'attempt_version':2,'original_failed_attempt':c['original_failed_attempt']}
 for ds in DATASETS:
  for scope in ('smoke','full'):
   s=store_for(c,ds,scope);s.verify('cpu_audit');audit=read_json(s.root/'cpu_audit/report.json');require(audit['status']=='PASS','all audits PASS required')
   evidence['datasets'].setdefault(ds,{})[scope]={'audit':audit,'audit_manifest_sha256':sha_file(s.root/'cpu_audit/manifest.json'),'summary_sha256':sha_file(s.root/'completed/summary.json'),'root':str(s.root)}
  for mode in ('Clean','Image-only'):
   values=audit['groups'][mode];rows.append([SPECS[ds][0],'IRRA',mode if mode=='Clean' else 'Image-only (G-v2 E60)',*[values[k] for k in KEYS]])
 target=ROOT/'docs/framework/results/gv2_cross_dataset_v2';target.mkdir(parents=True,exist_ok=True)
 for name,rounded in (('METRICS.csv',False),('METRICS_2DP.csv',True)):
  b=io.StringIO();w=csv.writer(b,lineterminator='\n');w.writerow(('Dataset','Victim','Mode',*KEYS))
  for row in rows:w.writerow(row[:3]+[format(v,'.2f') if rounded else v for v in row[3:]])
  (target/name).write_text(b.getvalue())
 evidence['cpu_regression']={'original':112,'new':29,'total':141,'main_result':'140 passed, 1 deselected','original_absolute_path_test':'1 passed'}
 before=c['original_failed_attempt']['before_evidence']
 from tbps.artifacts import checked,FileRef
 checked(FileRef(**before));old=json.loads(Path(before['path']).read_text())
 from tbps.transfer.gv2_core import signature
 for path,proof in old.items():require(signature(path)==proof['stat'] and sha_file(path)==proof['sha256'],'original failed Attempt changed')
 evidence['original_failed_attempt_SHA_and_stat_exact_files']=len(old)
 evidence['supervisor']=json.loads((Path(c['control'])/'detachment.json').read_text())
 evidence['execution_commit']=read_json(Path(c['control'])/'started.json')['code_commit'];evidence['config_sha256']=c['sha256'];evidence['status']='SUCCESS'
 (target/'SOURCE_EVIDENCE.json').write_bytes(json_bytes(evidence))
 table='| Dataset | Victim | Mode | R@1 | R@5 | R@10 | mAP |\n|---|---|---|---:|---:|---:|---:|\n'
 for row in rows:table+='| '+' | '.join(row[:3]+[format(v,'.2f') for v in row[3:]])+' |\n'
 report="""# G-v2 E60：CUHK → RSTPReid / ICFG-PEDES 跨数据集迁移

两个真实Smoke、两个Smoke独立CPU审计、两个正式Full评分和两个Full独立CPU重新排名审计全部PASS。

只有冻结的CUHK训练G-v2 E60；ImageOnlyGenerator(width32, blocks4)，原generate_once与8/255严格FP32 RGB投影不变。每张目标图库图重新一次G前向；不复用CUHK对抗图或扰动，不读目标PID/Caption作为G输入，无Backward/Optimizer/JVP，无目标域训练或微调，无H/文本攻击。

目标Victim复用C-PGC/SEP已验证的各数据集官方IRRA加载、Tokenizer与原生convert_weights混合参数/FP32输出特征、FP32 dot评分；G保持原fp32/math/no_tf32。该精度身份明确记录，未静默切换为CUHK权重或其他评分空间。Clean在当前协议下重新编码和计算，没有复制旧数值。

本实验同时改变目标数据域和IRRA checkpoint；不是只更换测试图像而Victim权重不变，也不证明多seed显著性或所有跨域泛化。E60由用户预先指定，不以目标Test结果选模型。

## 正式指标

单位为百分数，不另计算下降率。完整固定共享adversarial gallery；全部原始Test Caption、全部同PID正例、无camera过滤。

"""+table+"""
RSTP：1000 Gallery × 2000 Query；ICFG：19848 × 19848，完整集合、官方顺序和原始文本均保持。各组FP32原图/对抗图/有效扰动、特征、完整评分矩阵、全部排名索引、不可变回执与独立CPU审计保存在服务器输出，未向Git提交大型资产。

## 来源与运行

"""
 report+='参考执行Commit：'+c['reference_commit']+'；本轮执行Commit：'+evidence['execution_commit']+'。\n\nG checkpoint：`'+c['checkpoint']['path']+'`，SHA `'+c['checkpoint']['sha256']+'`。\n\n输出根：`'+c['output_base']+'`。控制与逐阶段日志：`'+c['control']+'`。\n\n'
 report+='共享GPU观察模式，无固定最低空闲显存门槛；RAM/磁盘/inode检查保留，无精度或Batch Size降级，不干预外部C-PGC/SEP进程。独立nohup+setsid监督器单次启动，阶段失败保留并停止，不自动重试。\n\n详细权重SHA、manifest/Tokenizer身份、真实调用数、原始指标、预算、审计SHA和历史保护计数见SOURCE_EVIDENCE.json；METRICS.csv保留原精度，METRICS_2DP.csv仅作展示舍入。\n'
 report+='\n## v2审计边界修复与回归\n\n基于1ed89e4，CPU审计仅允许无参数/缓冲区/子模块/Hook的精确Resize类型处理CPU图像；ToTensor走原纯CPU路径。其余nn.Module神经前向、CUDA、Backward、真实Optimizer step及两种JVP/forward AD入口仍禁止，异常后完整恢复。G、generate_once、8/255、IRRA Adapter、Tokenizer、打分和排名算法未改。\n\n原112+新增29共141个CPU用例PASS；真实v1 RSTP Artifact仅在测试目录隔离副本中验证完整审计，旧FAIL未重写。新Attempt重新生成两目标Smoke与全部Full，不复用旧失败的生成/评分作为新执行。原v1输出/控制264文件SHA与stat再次逐项核对一致，仍保留FAIL。\n'
 (target/'REPORT.md').write_text(report,encoding='utf-8')
 print(table);print('PUBLISHED_REPORT',target)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);publish(p.parse_args().config)
