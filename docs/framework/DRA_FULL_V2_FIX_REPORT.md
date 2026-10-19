# VLPTAttack Full v1预检隔离最小修复与v2执行

## 修复范围

基准055fea880214c95f3b8eee6282bb9de91ea1b742。
独立分支codex/tbps-vlpt-full-v2，工作树：
/home/lzf/ldx/projects/idea-test-tbps-vlpt-full-v2-20261019。

只改变dra_full.py的CPU预检分阶段职责：
- preflight(stage="source")保留公共配置、全部SHA、IRRA、MLM文件、官方DRA、
  完整3074/6156数据和Smoke来源核验，并严格验证原Source依赖版本。
- preflight(stage="target")默认保持原IRRA/RaSa/APTM来源、原生依赖与导入上下文检查。
- source_run明确选择source阶段；其已有Source版本检查保留，未删除或伪造版本检查。
- CLI增加--preflight-stage source/target，仅选择预检职责，不绕过模型加载授权。

Source进程不再导入/运行RaSa/APTM的native runtime verify；仍核验这些来源描述文件的声明SHA。
Target始终使用tbps-image验证自身原生依赖。两个环境版本完全不改。

Source主函数AST与v1相比，除preflight调用的stage关键字外exact；
dra_full_target、DRA Wrapper、Smoke、官方算法、参数、模型、Loss和梯度行为字节不改。
RAttacker.py SHA-256保持：
75518cb9465d9f8e994ea2f59c86b89fd4c39e3fe74f66b7a1d0a090eec2d313。

## v2身份与不变协议

新配置dra_irra_full_v2.server.json仅更新output_base和明确的当前producer路径/SHA；
其余字段与v1配置逐项相同。v1配置及18份控制/失败文件SHA、mtime和size完整保留。

控制目录：/home/lzf/ldx/tmp/vlpt_official_irra_full_v2/
输出根：/home/lzf/ldx/outputs/tbps-framework-vlpt-official-irra-v2/
正式结果：/home/lzf/ldx/outputs/tbps-framework-vlpt-official-irra-v2/full/

启动器仅更新工作树、独立v2配置/目录，并在GPU前分别运行两套真实Python的CPU预检。
从Gallery0开始，原3074图/6156文/385批、Batch8/global seed42一次，
Source只加载IRRA/MLM各一次，官方文本→图像→文本、8/255/10轮/2/255/5区域/五尺度及MLM不变。
先完整Source产物，再顺序IRRA/RaSa/APTM原生评分和独立CPU审计。
不复用v1 started/PID/日志/输出，不自动重试，不提供中断后RNG等价恢复，不创建v3。

## 真实CPU验证

真实Source环境：
/home/lzf/ldx/envs/dra-irra-source/bin/python
Torch2.1.0+cu121、Transformers4.28.1、timm0.6.13。

真实Target环境：
/home/lzf/ldx/envs/tbps-image/bin/python
Torch2.5.1+cu124、Transformers4.44.2、timm0.9.8。

两个CLI预检均已实际执行并exit0：
DRA_FULL_SOURCE_CPU_PREFLIGHT_PASS、DRA_FULL_TARGET_CPU_PREFLIGHT_PASS；
各3074/6156/385、最后2图、10 Caption成员cuhk:38920:0；
禁止真实torch.load模型和CUDA初始化，model_loads=0、GPU=false。

新增回归直接执行真实preflight，而非Mock其返回值：
- Source实际预检期间拒绝任何RaSa/APTM模块导入。
- 实际source_run经过真实预检并到达Store入口后由CPU测试主动停止，不创建产物或加载模型。
- Source依赖版本不符仍拒绝。
- Target对原生verify做透传计数，保留真实SHA/版本/导入检查。
- Target的RaSa/APTM依赖异常必须传播，不能吞掉。
- 非法stage拒绝，Source调用必须显式选择source。
- 沿用385批/全局RNG连续/失败停止/Storage与真实Bash调度回归。

Source侧47项通过、3项Target专用用例按环境跳过；Target侧153项通过、3项Source专用用例跳过、2项历史路径测试显式排除。
两项已有历史路径配置测试仍显式排除，不改旧配置/测试来使其通过。
开发时新增Target spy漏转发require_checkpoint关键字导致一次CPU测试失败；
只修正spy的*a/**kw透传，实际运行代码未因此改动，原测试日志保留。

未重跑GPU Smoke、容量测试或任何中间规模实验。
Source模型及MLM加载只由正式v2授权执行。

## 直接执行命令（仅本轮授权的一次v2）

~~~bash
cd /home/lzf/ldx/projects/idea-test-tbps-vlpt-full-v2-20261019
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-vlpt-full-v2-20261019/src
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/dra-irra-source/bin/python -B -u -m tbps.baselines.dra_full --config /home/lzf/ldx/projects/idea-test-tbps-vlpt-full-v2-20261019/configs/framework/dra_irra_full_v2.server.json --preflight-stage source
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.baselines.dra_full --config /home/lzf/ldx/projects/idea-test-tbps-vlpt-full-v2-20261019/configs/framework/dra_irra_full_v2.server.json --preflight-stage target
~~~

后台命令：
~~~bash
mkdir -p /home/lzf/ldx/tmp/vlpt_official_irra_full_v2
nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-vlpt-full-v2-20261019/src/tbps/baselines/run_dra_full.sh --allow-full >> /home/lzf/ldx/tmp/vlpt_official_irra_full_v2/launcher.log 2>&1 < /dev/null & echo "launcher PID=$!"
~~~

正式启动后禁止重复运行。后台复核两环境预检，再做一次即时资源检查；
GPU≥20GiB、可用RAM≥48GiB、磁盘≥35GiB，shared_observe_only，不操作其他用户任务、不轮询。
任何失败保留现场、退出后续阶段，无自动修复重试。

## 查看进度

~~~bash
C=/home/lzf/ldx/tmp/vlpt_official_irra_full_v2; O=/home/lzf/ldx/outputs/tbps-framework-vlpt-official-irra-v2/full; cat "$C/phase"; ps -o pid,ppid,pgid,sid,tty,etime,args -p "$(cat "$C/supervisor.pid")"; tail -n 6 "$C/status.log"; tail -n 4 "$O/source.log"; tail -n 3 "$O"/{irra,rasa,aptm,audit}.log 2>/dev/null
~~~

Source日志DRA_SOURCE_PROGRESS是已提交batch/Gallery/Caption数量；DRA_SOURCE_BATCH_BEGIN只是批次开始。
新产物的protocol、claim、bridge_trace/official_flow及batch/RNG证据提供真实执行记录；
不能仅凭nohup PID或RUNNING_SOURCE阶段名断言攻击已执行。

监督器、stage日志和完整Source/Target输出位置沿用v2目录。完整成功仍必须所有阶段及独立CPU审计PASS，
最终标志SUCCESS | VLPTATTACK_OFFICIAL_IRRA_SOURCE_THREE_TARGET_FULL_AND_CPU_AUDIT。
预计18–22小时，启动确认不是Full PASS；本轮启动状态另以不可变启动记录登记。
