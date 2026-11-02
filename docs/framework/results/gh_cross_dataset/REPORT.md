# CUHK冻结G60 + H7跨数据集迁移执行协议

CURRENT：独立`codex/gh-cross-dataset`已实现目标Caption冻结H推理及Text-only/Joint串行链。真实CPU预检、两目标Smoke/四份独立审计、Full执行状态只以本Attempt回执为准，不提前宣布完成。不等待Full完成；后台全部审计PASS后自动替换本报告并发布8行正式CSV和来源。

## 冻结边界

G为CUHK-PEDES训练的Epoch60，SHA `9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049`；H为CUHK-PEDES训练的E7/visit14336，SHA `487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df`。只恢复H policy，不恢复Optimizer；原Byte Transformer、CaptionFrontend、Dynamic-K、Unicode、BPE约束、NOOP及TextAttacker逐字节复用，H保持原CPU执行和原fp32/math设置。

H仅接收目标原Caption字符串及Caption-only Canonical属性，不接收PID、配对图、目标模型特征或反馈。RSTP生成2000条，ICFG生成19848条，绝不复用CUHK 6156条对抗文本。无训练、微调、Backward、Optimizer或JVP。

目标使用各自原生IRRA Checkpoint，与旧v2 ModelSpace完全一致；IRRA原生精度、Tokenizer、text_batch8、query_chunk128、评分/完整PID排名不变。无图像/G前向，8/255及原图像攻击协议不改。

## 输入与只读依赖

| Mode | Gallery | Query |
|---|---|---|
| Clean | clean_gallery | clean_text |
| Image-only | adv_gallery | clean_text |
| Text-only | clean_gallery | adv_text |
| Joint | adv_gallery | adv_text |

Clean/Image-only及其图像/特征完全只读引用`tbps-framework-gv2-cross-dataset-v2-20261102`。仅对应scope独立CPU审计完整原子发布且SHA/Manifest/Checkpoint/ModelSpace一致才允许读取；未提交缓存不消费。旧ICFG Full未完成时H可生成，新增评分等待已审计图库。新Text-only一次编码编辑后的目标Caption；合法NOOP复用原Clean特征；Joint只读复用同一份H文本和IRRA文本特征，不重推H或重编码。

RSTP Full为1000 Gallery/2000 Query，ICFG为19848/19848。两个Smoke分别严格10 Gallery/10原Query；两目标两模式四份Smoke CPU审计PASS才放行Full。审计新进程禁模型/CUDA/梯度，重放原文本约束、验证所有共享特征SHA/绑定，重算完整PID排名与指标；Joint还核验Text-only回执和共享文本身份。Smoke数值不填Full。

## 隔离与后台操作

- 工作树：`/home/lzf/ldx/projects/idea-test-tbps-gh-cross-dataset-20261102`
- 配置：该树`configs/framework/gh_cross_dataset.server.json`
- 输出：`/home/lzf/ldx/outputs/tbps-framework-gh-cross-dataset-20261102`
- 控制：`/home/lzf/ldx/tmp/gh_cross_dataset_20261102`
- 查看：`/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-cross-dataset-20261102/operations/run_gh_cross_dataset.py --status`

单次Supervisor采用flock、不可重用started Claim、逐阶段独立子进程和日志；nohup+setsid脱离SSH，核对PPID=1、PGID/SID独立和无TTY。共享GPU不设任何空闲显存门槛，资源只观察；不停止/重启989202、SEP、C-PGC或其他进程。原失败264文件、原H/G源码/Checkpoint等静态证据SHA/stat受保护；动态旧Full只通过已发布审计依赖消费，不改写。

真实错误立即FAIL、保留现场、不自动重试；后台全组PASS才生成`final/REPORT.md`、`METRICS.csv`、`METRICS_2DP.csv`、`SOURCE_EVIDENCE.json`并只发布小型报告/CSV/来源文件至本独立分支。发布失败保留FAIL，不伪装SUCCESS。

## CPU回归

最终184项CPU回归PASS：35项新增，149项相关历史回归；24+29+14+43+23+16+35，七个独立进程全部exit0。完整日志及配置SHA绑定在控制目录`cpu_tests.json`；逐模块独立进程运行，覆盖原G审计护栏、缓存/来源/排名和无GPU门槛，并新增冻结H原推理一致性、NOOP、Caption-only输入、禁训练/CUDA、四Smoke门禁、未发布缓存拒绝和完整Artifact路径。CPU端到端夹具使用真实冻结H与合成IRRA特征，不称真实Smoke。

开发首轮夹具命名空间、预先绑定Optimizer方法测试与历史legacy测试合并/原绝对路径不兼容的日志完整保留；修正夹具及测试进程隔离后重新验证，未因此重跑真实G或原Attempt。最终计数见CPU回执，不用旧测试结果冒充本轮通过。
