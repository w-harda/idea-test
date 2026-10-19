# VLPTAttack Full v2正式后台启动核验

## 启动结论

STARTUP_VERIFIED：v2已真实进入官方IRRA Source攻击，首批提交通过，第二批已开始。
仅代表启动与第一批运行核验，**不代表Full或最终CPU审计PASS**。

核验时间：2026-10-19T16:42:11.805758+08:00。
修复及实际运行源码Commit：34097c338997561fd7556e43c472d2e5625c05f0。
开发基准：055fea880214c95f3b8eee6282bb9de91ea1b742。
分支：codex/tbps-vlpt-full-v2。当前提交仅登记启动，不修改运行代码或配置。

## 后台进程与阶段

- 监督器PID=1307026、PGID=1307026、SID=1307026。
- 实测PPID=1、TTY=?，独立后台会话存活，SSH断开后可继续。
- Source子进程PID=1307611，使用dra-irra-source专用Python。
- 16:38:15 +08:00开始后台CPU预检，16:38:26 Target预检PASS，
  16:38:31 Source预检PASS，16:38:32进入RUNNING_SOURCE。
- 核验时已提交1/385 Batch、8/3074 Gallery、16/6156 Caption；第二批正在处理。
- v1的18个失败文件SHA、mtime、size全部一致，未复用v1身份。
- 没有新Attempt重试或v3，没有再次执行GPU Smoke或中间规模实验。

## CPU与保真证据

真实Source、Target CLI CPU预检均exit0，3074/6156/385、特殊10 Caption成员与SHA核验完整，
模型加载和CUDA初始化均0。Source47项/Target153项测试通过；
各跳过3项另一环境专用用例，Target2项历史路径配置测试显式排除。
完整日志索引在DRA_FULL_V2_CPU_EVIDENCE.json。

Source仅执行自身运行依赖检查；Target原生依赖和导入检查保留。
运行Source日志中无RaSa/APTM版本错误。
Source主函数除显式预检stage外AST exact，Target评分、Wrapper、模型和全部攻击参数未改。
官方RAttacker.py SHA保持：
75518cb9465d9f8e994ea2f59c86b89fd4c39e3fe74f66b7a1d0a090eec2d313。

## 首批实际攻击与提交

首8图及全部16文。首批官方攻击耗时123.6774915009737秒，
截至首批提交Source会话132.8636492267251秒。
完整事件text1→image→text2；实际Attacker.attack调用1、
img_guided_attack调用2、txt_guided_attack调用1，
区域rand3Num采样45、选择9轮、图像backward100、梯度观测55。

首批Claim、RNG和像素/文本单元已原子提交，发布后立即bitwise读回由Runner通过；
本轮再只读核验protocol/claim/rng/batch四份Manifest及声明文件SHA。
首批完整动作、原始/攻击Caption、T1、图像和effective_delta保留在batch-000000。

官方FP32边界检查通过；实数最大L∞=0.03137257695198059，
高于8/255的舍入差2.793237274811844e-8，未做二次投影，
不宣称严格数学上界PASS。

首批日志Peak RSS=2443964416 bytes；
提交时GPU allocated=1248596480 bytes、reserved=9869197312 bytes。
这是首批/核验时记录，不是整个Full的最终显存峰值或模型冻结终态审计。
启动GPU空闲24051MiB、利用率0%、无计算进程，
可用RAM121058377728 bytes、磁盘6378438537216 bytes。
未终止或修改其他用户进程。最终资源/模型摘要仍由Full execution/audit产物确认。

MLM未使用seq_relationship头与torchvision/TypedStorage告警是既有环境提示，未出现异常Traceback，
没有更换权重、版本、精度、候选或算法以压制这些提示。

## 命令与输出

实际执行：
~~~bash
nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-vlpt-full-v2-20261019/src/tbps/baselines/run_dra_full.sh --allow-full >> /home/lzf/ldx/tmp/vlpt_official_irra_full_v2/launcher.log 2>&1 < /dev/null &
~~~

该唯一v2已经启动，禁止重复执行。
控制：/home/lzf/ldx/tmp/vlpt_official_irra_full_v2/
正式输出：/home/lzf/ldx/outputs/tbps-framework-vlpt-official-irra-v2/full/

只读查看：
~~~bash
C=/home/lzf/ldx/tmp/vlpt_official_irra_full_v2; O=/home/lzf/ldx/outputs/tbps-framework-vlpt-official-irra-v2/full; cat "$C/phase"; ps -o pid,ppid,pgid,sid,tty,etime,args -p "$(cat "$C/supervisor.pid")"; tail -n 6 "$C/status.log"; tail -n 4 "$O/source.log"; tail -n 3 "$O"/{irra,rasa,aptm,audit}.log 2>/dev/null
~~~

Source全部完成后后台自动顺序进入IRRA、RaSa、APTM，再独立CPU审计。
预计整链18–22小时，以实际耗时为准。失败即停、不自动修复或重试、不创建v3。
最终成功需独立审计通过与：
SUCCESS | VLPTATTACK_OFFICIAL_IRRA_SOURCE_THREE_TARGET_FULL_AND_CPU_AUDIT

动态日志不会在本次文档中伪称为已封存；启动快照和首批不可变Manifest来源见
DRA_FULL_V2_STARTUP_EVIDENCE.json。本轮完成后不再持续监控或等待Full。
