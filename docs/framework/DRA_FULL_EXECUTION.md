# VLPTAttack（DRA）官方IRRA Source Full执行

开发基准889a57c963ec850473e67543043df9391c0451aa，分支codex/tbps-vlpt-official-full。
官方Commit91f78325e7cf0c497c9bbb5be2215be652d8cd9d，
RAttacker.py SHA-256：
75518cb9465d9f8e994ea2f59c86b89fd4c39e3fe74f66b7a1d0a090eec2d313。

## 新增模块和复用边界

- dra_full.py：385批外层调度、预检、Claim/像素/文本/RNG证据与完整性检查。
- dra_full_target.py：三模型完整原生评分、四指标导出及独立CPU审计。
- run_dra_full.sh：独立后台监督器、阶段日志、单次启动锁及失败停止。
- dra_irra_full.server.json：冻结配置、模型/官方/运行来源SHA和Smoke完成证据。
- test_dra_full.py：385批实际Runner CPU Mock、失败注入、RNG/Storage/排名/Bash流程。

旧Smoke、Wrapper、官方源码、模型和历史产物均不改。
直接复用dra_irra的configuration/save_pixels/resources、dra_official的数据和被动追踪，
IRRA/RaSa/APTM原生Adapter、FeatureCache/NativeFeatureCache、
CachedIRRA/ChunkCachedScorer/APTMSetScorer及EvaluationEngine。

## 冻结协议

CUHK-PEDES官方Test3074 Gallery、6156 Query；385批（384×8+2），全部原始配对Caption，
包括cuhk:38920:0的10 Caption，不扩展为同PID其他图像Caption。
Source IRRA与MLM各加载一次；只在整个Source开始设置一次seed42，不按Batch重设。
直接调用官方RAttacker.Attacker.attack()；完整文本→图像→文本，8/255、10轮、2/255、
5候选/整数三角采样/五尺度、原MLM候选和0.6/0.2/0.2不变。
一次替换按官方原词槽位计数，保留多WordPiece候选展开短语的行为，不过滤候选。
Source eval；官方候选backward需要参数梯度，但无Optimizer、无参数更新，终态权重/buffers exact核验。

每批先Claim，再官方攻击；保存紧凑FP32原RGB、对抗RGB、effective_delta、原文/攻击文、
T1、最终动作、完整阶段记录；原子提交、文件SHA和Tensor bitwise读回通过后计入进度。
保存Python/NumPy/Torch CPU/CUDA前后完整RNG状态和稳定摘要，校验跨批连续。
RNG证据仅用于追溯，不提供自动恢复/重试/跳过；中断后保留现场，另行审查授权。

全部Source完成并逐批验证后，才顺序进入IRRA→RaSa→APTM，各自原生Tokenizer/预处理/融合评分，
共享同一冻结产物，无Target梯度或反馈。四模式Clean→Text-only→Image-only→Joint。
只发布R@1、R@5、R@10、mAP百分数，原精度CSV和两位小数CSV；不发布mINP或DR表。
Matched Clean是保存的IRRA PIL RGB bilinear384×128原图Tensor和官方pre_caption(...,30)文本，
不使用历史Raw Clean。官方FP32投影约2.793e-8实数边界舍入保留，无二次投影，
不宣称严格数学8/255上界PASS。

## CPU验证

42项DRA/Full测试通过，其中16项新增Full测试；148项适用公共回归通过。
两项旧绝对路径配置测试显式排除：
test_full_config_SHA_scope_and_no_real_model_load在原889a57c也失败；
test_final_APTM_config_dry_no_model_or_cuda已在前阶段记录基准失败，未改旧文件。
开发初次把旧官方Tokenizer测试放在较新的Target Transformers环境，出现vocab兼容错误；
日志保留，改用已冻结Source环境通过，没有修改Tokenizer或算法。

Source环境dra-irra-source：Torch2.1.0+cu121、Transformers4.28.1；
Target和审计使用tbps-image。
真实Full CPU预检3074/6156/385、最后2图、特殊10 Caption、原图SHA、
官方与模型/MLM/目标来源通过，模型加载/CUDA均0。另检查bash -n、git diff --check和历史保护。
CPU Mock不是Full容量/攻击效果证明。

## 唯一正式启动

工作树：/home/lzf/ldx/projects/idea-test-tbps-vlpt-full-20261019
控制：/home/lzf/ldx/tmp/vlpt_official_irra_full_v1/
输出：/home/lzf/ldx/outputs/tbps-framework-vlpt-official-irra/full/

~~~bash
mkdir -p /home/lzf/ldx/tmp/vlpt_official_irra_full_v1
nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-vlpt-full-20261019/src/tbps/baselines/run_dra_full.sh --allow-full >> /home/lzf/ldx/tmp/vlpt_official_irra_full_v1/launcher.log 2>&1 < /dev/null & echo "launcher PID=$!"
~~~

默认Python只CPU预检；Source/Target必须显式--allow-full。
started、Full输出或独占锁已存在则拒绝重入；不覆盖，不轮询，不自动重试。
资源保持空闲GPU≥20480MiB、可用RAM≥48GiB、磁盘≥35GiB，shared_observe_only，
其他用户进程只观察。实际不足/OOM停止，不降低参数。

阶段：CPU_PREFLIGHT→RUNNING_SOURCE→RUNNING_IRRA→RUNNING_RASA→RUNNING_APTM→RUNNING_CPU_AUDIT→SUCCESS/FAIL。
控制目录记录supervisor.pid/.pgid/.sid、code.commit、phase、status.log、launcher.log及资源日志。
Full/source.log中DRA_SOURCE_PROGRESS记录已提交Batch/Gallery/Caption数量、资源和耗时。
Target日志为irra.log/rasa.log/aptm.log，审计audit.log；模型会话独立，审计GPU0次。
每个Target完成目录包含summary.json、metrics.csv、metrics_2dp.csv、scores.pt；
最终审计为full/audit/completed/audit.json。

~~~bash
C=/home/lzf/ldx/tmp/vlpt_official_irra_full_v1; O=/home/lzf/ldx/outputs/tbps-framework-vlpt-official-irra/full; cat "$C/phase"; ps -o pid,ppid,pgid,sid,tty,etime,args -p "$(cat "$C/supervisor.pid")"; tail -n 8 "$C/status.log"; tail -n 5 "$O/source.log"; tail -n 3 "$O"/{irra,rasa,aptm,audit}.log 2>/dev/null
~~~

需要人工停止时仅向本监督器发TERM（转发给当前子进程、保留现场）：
~~~bash
kill -TERM "$(cat /home/lzf/ldx/tmp/vlpt_official_irra_full_v1/supervisor.pid)"
~~~
停止后不得直接重跑，无自动恢复承诺，不操作其他用户进程。

最终成功需要各阶段及独立CPU审计全部PASS，并出现：
SUCCESS | VLPTATTACK_OFFICIAL_IRRA_SOURCE_THREE_TARGET_FULL_AND_CPU_AUDIT
启动成功不等于Full PASS。预计整链18–22小时，以真实运行记录为准；
10-Caption峰值、全量Token边界和长期资源趋势待运行验证。
本轮无32图/100图中间实验、无重新训练。启动真实状态在后续登记文件记录。
