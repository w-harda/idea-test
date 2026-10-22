# ADR-0016：真实后端的来源、字节Span及分级执行门禁

日期：2026-10-22。状态：CPU实现采用，真实数值待授权。继承ADR-0015。

## 背景

原G/H接口的contracts、utils等模块名存在冲突；必须在不改历史源码的情况下保持同一类型身份。
Unicode同形字符会改变UTF-8/BPE长度，原mapper可能因utf8_character_split拒绝。
CPU恢复通过不等于GPU Adam/RNG、新reader/JVP或双向更新已经通过。

## 决定

- 旧源码按SHA编译，CPU及真实执行复用同一模块对象；会话退出恢复路径、模块、函数包装及锁。
- IRRA原生加载在legacy会话外，CPUStartupGate在隔离退出后才允许CUDA。
- CPU完整读回G60/H10和双Adam/RNG；G设备迁移后重新绑定Adam对象并读回全部父状态，不能丢动量。
- 原Canonical与字符动作保持；仅对可证明的UTF-8字节拆片建立Span接口，BPE ID不改，原诊断保留。
- 新条件空间绑定reader/basis/字节接口/producer；非NOOP调用原定位与JVP/FD，NOOP显式复用，不改S_raw/K。
- 默认CPU。数值探针只计算原第一步Loss梯度、禁止Optimizer.step；两轮Smoke还必须验证同来源探针PASS。
- G更新使完整2048 bank失效；重新建原RewardBank，原H SCST使用最新图库。
- 同文本旧/新bank的Reward差只诊断；流水线通过与双向更新证据分别记录，不能靠新Reward制造通过。
- 状态、动作、条件、bank及Claim不可变发布；未提交状态不自动重播。

## 影响与限制

坐标桥是明确的新输入接线契约，不能伪称旧mapper对Unicode拆片已经验证。
其他歧义、截断、跨Span、FD/冻结/预算失败仍拒绝；不得回退旧JVP。
保留G(I)/H(T)、原G K+1/Loss与原H K/Reward；共享属性和新交互目标仍是独立研究变量。
本轮没有GPU、真实模型前向、JVP或更新，不能标记双向训练VERIFIED。
完整说明见framework/results/gh_bidirectional_backend/REPORT.md与EXECUTION_PROTOCOL.md。
