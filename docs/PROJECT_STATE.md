# 项目当前状态（唯一权威入口）

状态来自Git、实际代码与不可变证据；历史报告为当时快照。

## 当前任务：RaSa Full一次性GPU等待器（按用户明确授权部署）

基准b51d9e55190e96b6422f0c87952471ad6ac12bed，独立分支codex/tbps-rasa-materialized-evaluation，工作树/home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017。RaSa实现Commit：5869fa59cbdc005f6bbf58276c96070c5e1548b4，发布HEAD d2fbdfbe30b1356746743b2bc1c7be168f652534。

用户上传官方RaSa CUHK权重，SHA bc85da09c2991d5de503c2c2ac4f032e20cc536fec5094f0acff45e93d0104d2；官方代码固定bd16aa1a15f149548a90d196fcf10a27d7ab7c66。只读复用generation_v3固定3074图/6156文，不重新生成/优化G/H，不复用IRRA Embedding。复用原Dataset/Engine/指标/存储，仅补RaSa原生局部表示、Top-128 ITM、结构化缓存与薄入口。原生模型加载保持strict849项，旧Transformers兼容仅在Adapter初始化边界。

CPU127项通过：125项必要回归＋原树1项绝对路径测试＋独立官方RaSa itm_eval对照1项。一次真实10 Gallery/10 Query Smoke PASS，RaSa加载1次，Hook核实20图像/20文本/400融合pair，G/H/IRRA及优化0，模型冻结exact。峰值allocated1900.05859375MiB、reserved2082MiB，RSS约4.069GiB；执行28.283271秒、含预检总121.416593秒，GPU退出、空闲3865MiB。没有OOM或重试。

四组Smoke mAP为100.0/95.88888549804688/99.16666412353516/91.44444274902344，仅链路验证。10候选Smoke不冒充Full Top-128。目标384×384 Tensor BICUBIC来自固定384×128 FP32物化输入，不宣称与原始图片PIL直接预处理的论文基线逐值相同。

## 当前正式产物与证据

- generation_v3 Joint全量攻击数据已封存并审计：Plan7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5；3074 Gallery/6156 Query，6116编辑/40 NOOP/13537替换。
- **IRRA Materialized Full已由用户完成并独立CPU审计PASS**：Clean/Image-only/Text-only/Joint mAP分别66.09744262695312/32.28228759765625/51.471561431884766/23.553220748901367；与历史Joint Full四组20项指标exact。
- IRRA证据：/home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation/full/；summary SHA4530f3f97156d6e9f65c07b0775588324c39679feb0f16ae66b356692b017eb0；用户日志SUCCESS | IRRA_FULL_AND_CPU_AUDIT。本轮只读引用，不重新评测。
- RaSa Smoke：/home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/smoke/；summary SHA036ca6df1ffc793dfb39e36cc7b38294b12655f0f23975dc005822829c04aa61；report SHA8a9b7bcc0b73017f3cc85a8809b530c4a21e55ca071ab0242291f39d7c989995。
- 历史生成/审计26文件SHA/mtime、61555物化文件成员/size/mtime、IRRA正式结果3关键文件SHA及冻结工作树保持；没有重复7991文件/旧Full重审。
- **RaSa Full PENDING**；本轮启动前确认full目录未创建。用户新授权独立Bash后台等待器，每60秒采样，连续3次空闲≥8192MiB/利用率≤10%/无计算进程后单次启动原Full，成功后CPU审计；不干预Ollama、失败不重试。
- 等待器10项CPU测试及bash -n通过，未在测试中访问真实GPU。等待器代码Commit 8e4a4ab3d53df1fa940a509e6937b346e5cda91a；2026-10-17 23:45:13+0800通过nohup独立session启动，PID3297416。一次性确认PPID=1、SID=3297416、无TTY、stdin=/dev/null，当前WAITING 0/3；首条快照空闲3865MiB/利用率100%/Ollama PID2436556，Full目录及triggered均不存在。运行证据在/home/lzf/ldx/tmp/rasa_full_wait/；详见framework/RASA_GPU_WAIT.md。以上为部署时快照，后续状态以运行日志为准。

具体代码、限制和命令见[RaSa执行说明](framework/RASA_MATERIALIZED_EVALUATION.md)及[真实Smoke报告](framework/RASA_SMOKE_REPORT.md)。本轮不进入APTM、其他算法或训练；Codex在启动等待器并确认脱离会话后结束，不持续监控。

## 历史索引

- generation_v2：原FAIL结果Manifest SHA9b2dae20980ce9e1943fcaada80e2d73539fd09fb9ea3d4ada106e077d79efa9；gpu-result/report SHA176ae52ffc1851a3d710c445268569a987baaed912d95696941ec7e73c052d13。
- 原首图Claim Manifest SHA21ed14c935fdde887cfc6afcbdb67b6e6c6687085f6b8a46e8c3aed5ad5ea1e0，unit_id50b19770bb01c5c7e52237a4af73e4186342e77ccafe06204a8c3c91ecbfe3b7。
- Stage4D2R：dd0a213f2ac1b787412b755372c378ad49f31f9c，启动顺序CPU修复/626测试通过，报告是当时v2尚未执行的历史快照。
- generation_v1：整体FAIL、G前向0，原报告STAGE4D2_EXECUTION_REPORT.md及旧原树未提交文档保持。
- Stage4D1：82fe88a5a74955838fc6341c1ed9a46921c94d7c，6156历史文本物化/独立审计通过。
- Stage4C3：a5a6f6ae8c9200482ae0c70d228e1873d3ca4dd2，固定两图两文IRRA物化端到端等价PASS；不代表Full或跨模型。
