# G/H probe_v2：IRRA设备字符串最小CPU修复

修复基准：7ba66d9525be243552b66f34b01289a64502257b。
原数值后端：a46c6980c4abf58df02ca1ee199da799892dc81e。
独立分支：codex/gh-probe-device-fix。
工作树：/home/lzf/ldx/projects/idea-test-tbps-gh-probe-device-fix-20261023。

**本轮只完成CPU修复、回归、来源与保护核验；未初始化CUDA，未运行probe_v2或任何Watcher。**

## 1. 原因与最小Diff

probe_v1于2026-10-23 02:41:55失败。
native_backend.py:347把裸cuda传给IRRAAdapter.from_source；
原irra_source.py:83–84只允许cpu或cuda:<index>，因此在权重恢复前拒绝。

~~~diff
- adapter=IRRAAdapter.from_source(source,device="cuda",allow_model_load=True)
+ adapter=IRRAAdapter.from_source(source,device="cuda:0",allow_model_load=True)
~~~

这是唯一生产Python修改。没有全局替换cuda；新真实后端只有这一个官方IRRA加载调用点。
Tensor/模块.to("cuda")保持原样，不修改原加载器的严格校验。
AST比较确认只改变该关键字常量；其余生产源码与修复基准一致。

## 2. CPU验收

| 范围 | 用例 | 结果 |
| --- | ---: | --- |
| 原交替后端相关CPU回归 | 224 | PASS，原断言保留 |
| 既有轮询器CPU回归 | 39 | PASS，未启动轮询器 |
| 新设备、v2来源及资源契约 | 13 | PASS |
| 合计 | 276 | PASS，0 skip，退出码0 |

新增测试实际进入loaded_backend上下文，模拟已完成门禁标志并禁止真实CUDA/权重/模型前向，
捕获真正传给Adapter的cuda:0，验证退出关闭以及加载失败不进入legacy会话。
另直接调用未修改的官方加载器，验证裸cuda及其他非法类型在source.verify/torch.load前拒绝，
合法cuda:0仅推进到受控来源哨兵，不加载模型。

新来源检查覆盖完整producer索引、缺项拒绝、ConditionIdentity来源篡改拒绝、
v1合成回执即使复制到v2路径仍拒绝，以及10239MiB拒绝/10240MiB允许。
所有模拟篡改均限测试Fixture，不修改真实FAIL、配置或历史资产。

实际回归67.66秒；唯一warning为Kornia custom_fwd弃用提醒，不代表初始化CUDA。
随后真实tbps-image环境执行新配置CPU Preflight及完整G60/H10模型/Adam/CPU RNG恢复，退出码0。
启动事件仅cpu_sources_verified→cpu_isolation_exited，无CUDA初始化事件。
G Adam484080；H逐参数20124–20370，完整游标20480 visits/20370更新。
CUDA RNG仅保留原字节，设备恢复仍PENDING；本轮没有扩大修复范围或补写数值PASS。

## 3. 新协议及来源身份

新配置：
/home/lzf/ldx/projects/idea-test-tbps-gh-probe-device-fix-20261023/configs/framework/gh_bidirectional_backend.probe_v2.server.json。

原gh_bidirectional_backend.server.json保持原字节，是v1历史快照，不能在新树冒用其来源。
新配置显式绑定原配置、probe_v1/FAIL、原FAIL证据索引和Watcher deployment.json。

- 完整当前框架producer：140项，包含基准已存在的operations/gh_probe_watch.py；它仅纳入完整来源索引，本轮未调用或改写。
- 固定legacy运行来源：496项，未改。
- prototype.new_producers同步重建；managed_roots迁移新框架路径，历史prototype/模型路径不动。
- ConditionIdentity源摘要同时绑定新alternating源码、原IRRA空间、reader/basis及字节映射，内外字段一致。
- probe_v2与smoke_v2独立；smoke_v2只允许同一新来源probe_v2回执，不接受v1回执。
- 原6144MiB配置门槛在新协议中设为10240MiB，落实此前外部10GiB要求；利用率仍只记录。
- 固定样本、K、Loss/Reward、数值阈值、前向/JVP/更新预算、模型、数据、原生精度均不变。
- 默认gpu_execution_default及prototype.gpu_execution_authorized仍为false；真实执行需要新的明确用户授权和显式命令参数。

文件SHA、规范化来源摘要、旧新协议差异见SOURCE_EVIDENCE.json。
原配置、Checkpoint及旧FAIL不重写；新身份不声称与旧来源完全相同。

## 4. 历史保护

151378项历史文件大小、mtime_ns、inode均一致；关键G60/H10/IRRA Checkpoint SHA一致。
原probe_v1与原Watcher共28份文件，全部SHA、大小、mtime及inode保持。
原冻结后端139项producer SHA保持，原冻结工作树与旧Watcher工作树干净。
全部旧FAIL、Claim、启动标记、日志及部署记录完整保留。

本轮结束时probe_v2、smoke_v2均不存在；未部署或启动v2 Watcher。
该结果不等于真实GPU数值验证通过；GPU设备迁移、JVP/FD、梯度差、实际显存仍PENDING。

## 5. 文件及下一步

- native_backend.py：仅一行设备参数修复。
- 新probe_v2.server.json：派生协议、完整来源索引和独立输出。
- test_alternating_native_device.py：13项新增CPU控制流/契约测试。
- 本目录：报告、CPU预检、AST/来源及保护证据、后续执行协议。
- PROJECT_STATE、ARCHITECTURE、EXPERIMENT_REGISTRY：更新CPU修复状态，保留历史FAIL。

[后续执行条件与命令](EXECUTION_PROTOCOL.md)仅供用户下一次明确授权后使用。
本轮不执行probe_v2、A1/B、训练、Validation、Test或Full。
