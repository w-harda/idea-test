# Probe专用共享GPU资源派生协议

基准f9701c7ea924f734d83f6cfeeaa38b0b83c30f01；独立codex/gh-probe-shared。
本次获授权：CPU检查后，符合资源即直接执行一次原probe_v2，不部署Watcher。

## 唯一资源变化

新配置gh_bidirectional_backend.probe_v2.shared.server.json增加probe_resource_limits.minimum_free_mib=6144。
resource_protocol(c,action)只对probe构造资源副本；原prototype.proposed_gpu_limits仍为10240，Smoke不变。
RAM32GiB、磁盘25GiB、inode80000及所有原检查保留，利用率和其他进程仅记录。
原v2配置字节不改；完整140 producer和条件来源摘要按新树与资源代码正规重建，496 legacy来源不变。
原模型、训练代码、Tokenizer、JVP/FD、K、Loss/Reward、精度、两样本、计数和数值阈值均不变。

probe_v2此前没有创建或消费，沿用该未消费目录；首次Claim只绑定本新配置/source_key。
输出：/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/probe_v2/。
不复用旧FAIL或Claim，不为资源回落自动重试，不自动创建probe_v3。
Smoke仍未授权；若未来授权其独立资源门槛仍为10240MiB，不继承本Probe覆盖。

固定cuhk:26530:0和cuhk:26747:0；G≤4、IRRA图像≤4、IRRA文本≤6、reader≤1、JVP≤1。
Optimizer.step=0、完整图库刷新=0；不得自动接续A1/B、训练或任何评估。

## CPU证据与执行

289项CPU测试PASS（原276+13项资源作用域），0 skip；67.71秒，1项原Kornia弃用warning。
实际CPU Preflight、父状态、历史保护和GPU结果在执行后独立登记，不预先宣称数值PASS。
控制目录/home/lzf/ldx/tmp/gh_probe_v2_shared_execution_20261023，仅该目录新增日志和进程记录。
执行使用原native_cli --action probe --allow-gpu-probe；后台nohup setsid仅包装日志与GNU time，无攻击算法。
本次资源不足则停止，不轮询；运行失败保留FAIL/Claim/pending，无自动重试。
原probe_v1、旧Watcher、旧v2配置及已完成实验全部保持只读。

CPU通过不等于GPU数值通过；CUDA RNG恢复必须按真实报告判断，不能补写PASS。
