# 独立审计明细与证据边界

原60轮审计结果：**PASS**。覆盖60轮、122880条G/H/条件数据，G 484080更新、H 122880访问/122220更新。耗时157.251830秒，实际进程峰值RSS 586027008字节（558.879MiB）。

原训练计算已完成；原产物SHA/结构完整；最初失败仅为read_json自身64MiB限制。未重新训练、未重算JVP、未重建历史Bank，也未将原FAIL改写成PASS。

| 项目 | 证据/结论 |
|---|---|
| 账本大小 | 总3890547998字节；60份范围63102008–68178636字节 |
| 首次超限 | Round2，68178636字节（65.020MiB）；超限轮2/3/4/5/7/9/10 |
| Round2字段 | G 10934439字节(16.038%)；H 19537913字节(28.657%)；conditions 37613964字节(55.170%) |
| 纯解析进程实测 | 整份421588992字节RSS；流式20951040字节RSS，均为同一Round2文件 |
| 实际最大解析缓冲 | 103522字符；没有整份账本常驻 |
| 文本动作 | 共246477份输出：244440份训练sampled/greedy + 2037份最新合法rollout；最多6词 |
| 文本规则 | 重新计算原候选合法性与gamma3/beta1 Natural-K，原Unicode回放，属性无放回，每原词位最多1字符 |
| 图像训练账本 | 每个G更新的已记录最大L∞=0.03137254901960784，≤8/255 |
| 最新Bank Tensor | 2048张，实测最大L∞=0.0313725471496582 |
| 更新/状态 | 每行原K+1；所有步数/完整父状态声明连续，G/H轮边界链及H原生逐步链连续，双Adam最终step匹配 |
| 条件/来源 | 原S_raw、原选择ID与顺序不改；当前H语境、新JVP/原FD回执和冻结隔离核对 |
| 权重身份 | 六对现有G/H都与对应轮ledgerSHA、配置SHA、边界状态内容摘要精确一致；H原生摘要桥接精确 |
| 保护 | 43907原始输出/原Supervisor文件SHA/inode/size/mtime全部exact；首版恢复7个控制文件额外保护 |
| 审计执行 | CPU-only，模型前向/反向/Optimizer更新/新JVP/CUDA均0 |

## 已退休数据的证据边界

第1–59轮原始完整状态、Bank Tensor和G-rollout已由原训练流水线按预定滚动策略退休；保留全部ledger及退休Manifest。逐轮验证其状态/Bank来源、ledgerSHA、后继父态声明、预算记录和冻结指纹，未重新生成资产。这些轮次的原始图像Tensor与G-rollout不能重新实测；不能把账本证据称作读取了已退休Tensor。全部历史H sampled/greedy输出仍在账本，已逐条严格回放。第60轮条件/rollout/Bank均读取保留原资产。

## 两种H摘要与首版恢复失败

原H逐步日志的冻结feedback_backend.state_sha使用(shape,dtype)及原始字节；轮边界tree_sha使用(dtype,shape)及原始字节，因此同一模型会有两个不同摘要。首版恢复工具新增检查把二者混比，13秒停止；没有产生PASS或Test。其源码Commit e2abc63及FAIL/started/exit/log/status保持。第二版176571d新增独立工具和控制身份，使用原生摘要检查全H逐步/跨轮链，使用tree_sha检查边界，并用初始、六对现有Checkpoint和最终完整状态重算两种摘要桥接，未跳过连续性检查。

同一初始H的实际摘要：tree_sha `02dfb3a9a45c6da4b7e739cc4f80bee31230926498938a15337448510ea47d6a`；原生state_sha `9d607ba8b6b348a8783b3dad9b79d0206dd23147edec5fa794b7e932c5a2853f`。真实回归验证两者，篡改一个权重值必须改变原生摘要。

10项不同新增CPU测试+27项相关回归PASS；首版9项同名用例不重复计数。原158份生产源码、496份运行来源及配置SHA不改；恢复工具独立版本清单逐阶段检查。

## 六组权重文件SHA256

每组G.pt为5525492字节，H.pt为628592字节；Manifest及完成回执均验证，详细Manifest和来源见DIAGNOSIS.json。

| Round | G.pt SHA256 | H.pt SHA256 |
|---:|---|---|
| 10 | `c602b57b80acdab8fcce848467dbfc0690bedb1d0104662e7f1879cd4a7a5d88` | `bee49f85329425d8b8195379479a529b08f9444db756a68f6feeb2bcd51454c7` |
| 20 | `bc04aa8f01b255bb4b517696aff4367c5f7d1355fa2acfe5d27dbe1099e355ad` | `d851bec355ca77950cb32945234c3347db6542f8bda27e3357516b9b2593959c` |
| 30 | `bff8ded3958339dd6546543bb2a8f5c8c75b6603e33cbaf7d524f2e86cf7f1ea` | `2c6bc21121aea6d92b5c89c343cbdaa49a978e952e291028c9cfda627a596f51` |
| 40 | `470bb38f768ac75b025890c2ff7057248bda9b4fa31c01f41fd4d991a88c921f` | `ada00ed1e73b5030085447ca0f4984ac539df9cf2d28168afc2c0ebc16c566e7` |
| 50 | `2eef36a18ea946537e9edbd1f703b5f5715c3ac205192b73508dc4ce2e790e59` | `1766e9a0d6a3bf9ab2f871770aa550c0d0388d72b7788efe2eddffbcf2d5c272` |
| 60 | `f9bfbb1912505b1b6b9a477a1eabc596e7ae02ba1f8eb087119f0fb5265cb8c3` | `621fab9ee4f9a2bd19fbb9028c8ba3fd4ba9b0c190deb261b4ace0ef9fdef629` |

执行控制：`/home/lzf/ldx/tmp/gh_alternating_v2_posttraining_recovery_20261108_v2`。原训练Attempt仍为1；新控制只恢复后处理，不调用原Supervisor，不创建训练优化器。
