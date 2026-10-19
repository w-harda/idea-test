# VLPTAttack官方IRRA Source Full首次启动：FAIL记录

## 结论

本轮仅一次后台启动，已FAIL并停止，未自动重试或创建新Attempt。
正式Source模型加载、MLM加载、官方攻击调用、Claim和新图文产物均为0；
三个Target未开始。未消耗真实GPU攻击计算，不存在已完成批次需恢复。

代码Commit：fde8f6b23af09e18fc1ad2a8da9aacc241b89cb2。
分支：codex/tbps-vlpt-official-full。
开发基准：889a57c963ec850473e67543043df9391c0451aa。
原官方RAttacker SHA75518cb9465d9f8e994ea2f59c86b89fd4c39e3fe74f66b7a1d0a090eec2d313不变。

## 启动与实际状态

2026-10-19 13:54:14 +08:00后台启动，13:54:25 CPU_PREFLIGHT PASS，
13:54:26进入RUNNING_SOURCE，13:54:31 FAIL（exit1）。
监督器PID/PGID/SID均870320；最初实测PPID=1、TTY=?、独立会话正常。
复核时监督器已退出，phase=FAIL，不能将Shell成功启动当成实验正在运行。

控制目录：
/home/lzf/ldx/tmp/vlpt_official_irra_full_v1/
正式输出：
/home/lzf/ldx/outputs/tbps-framework-vlpt-official-irra/full/
原started、PID、phase、所有日志和SHA记录完整保留；输出仅source.log，source/目录尚未创建。

实际启动命令：
~~~bash
nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-vlpt-full-20261019/src/tbps/baselines/run_dra_full.sh --allow-full >> /home/lzf/ldx/tmp/vlpt_official_irra_full_v1/launcher.log 2>&1 < /dev/null &
~~~

## 已通过的检查

- Source固定环境42项DRA/Full CPU测试；Target环境148项适用公共回归。
- 两项旧绝对路径配置测试显式排除，详见DRA_FULL_EXECUTION.md；
  其中IRRA配置测试已在原基准Commit复现同样失败。
- 实际Target环境Full CPU preflight：3074 Gallery、6156 Query、385批、最后2图，
  cuhk:38920:0的10 Caption及原图/来源SHA通过，无模型加载或CUDA。
- bash -n、git diff --check通过；旧Smoke、模型/算法、配置、历史结果未改。
- 即时资源：RTX4090空闲24051MiB、利用率0%、无计算进程；
  可用RAM约113GiB、磁盘约5.8TiB，超过20/48/35GiB门限。
  后台自身资源检查亦通过。不是GPU容量/OOM故障。

## 具体原因和测试缺口

新增dra_full.py::source_run首先调用完整preflight；
preflight不仅验证Source/data，还执行RaSa/APTM的原生Target来源verify。

首次预检在tbps-image环境通过。但Source子进程使用原冻结的dra-irra-source环境，
再次执行完整预检时，RaSa.verify检查到依赖版本不匹配并拒绝：

RaSa runtime dependency version mismatch

对应代码位置：dra_full.py的preflight内src.verify()，由source_run内preflight(path)调用。
Source环境实际Torch2.1.0+cu121 / Transformers4.28.1 / timm0.6.13；
Target环境Torch2.5.1+cu124 / Transformers4.44.2 / timm0.9.8。
两个环境原本均正确、各自Smoke通过；错误是新增调度入口把Target环境检查带入Source。
没有改动环境，也没有发生Source模型或MLM加载。

这是新增Full控制代码的环境隔离错误，不是官方DRA算法失败。
CPU实际Runner Mock替换了preflight，真实预检只在Target环境执行，未覆盖Source环境预检，
因此现有PASS未能发现此问题；该覆盖缺口如实登记。

## 最小后续修复建议（本轮未实施、未重启）

只拆分公共来源/数据检查与各阶段运行环境检查：
- CPU启动阶段在Target环境验证三个受害者来源及Target运行依赖；
- Source进程只检查Source固定版本、官方/IRRA/MLM来源、数据和完整映射；
- 保留所有SHA与模型检查，不能伪造版本或关闭来源验证；
- 在Source与Target两个真实Python环境分别执行禁止模型加载/CUDA的CPU预检回归；
- 加入实际Source启动流程不能进入Target运行依赖检查的CPU测试。

不需要改官方RAttacker、Wrapper、Loss、参数或模型。
任何再次正式启动必须使用独立新控制身份并保留本次全部FAIL日志，
由用户审查修复和新启动方案后授权；不得重跑当前唯一入口或清理started/full/source.log。
本轮未执行这些后续操作。

## 证据

- DRA_FULL_PRELAUNCH_EVIDENCE.json：CPU测试、预检、官方SHA与启动前快照索引。
- DRA_FULL_LAUNCH_EVIDENCE.json：本次所有18个控制/输出文件SHA、size、mtime，
  PID/会话、环境版本、0次模型/攻击/产物及FAIL原因。
- 完整错误：full/source.log。
- 完整阶段：控制目录status.log、launcher.log、phase。
- 启动代码：控制目录code.commit、launch_sources.sha256。
- 完成状态：FAIL，无Full指标，不宣称Full PASS。
