# C-PGC/SEP真实启动与失败记录

## 结论

真实后台执行已发起并正确脱离SSH，但在IRRA Target Smoke遇到真实接口错误后按协议停止；没有正式Full，完成0/12，没有后台任务继续运行。不能把Source Smoke或101项CPU PASS当作完整GPU Smoke通过。

## 共享资源执行版本

执行db7e3e2ba382ac944c0911211f0d5d562c100933，基于c134cb3。独立配置SHA be0054037dbade5e73503dad2e354acd431efd7da67127382ba74a1534d38e5e；显存准入8192MiB、RaSa融合batch16但完整Top128、Query Chunk32，其余算法与精度不变。101项CPU测试及真实Source/Victim CPU预检PASS。Full另要求六组同来源真实Smoke/审计及峰值+2048MiB余量，但本次未达到。

启动即时空闲9067MiB，未干预Ollama。nohup+setsid监督器PID/PGID/SID=4083864，运行时PPID=1/无TTY，当前已退出。

## GPU实际结果

C-PGC CUHK Source Smoke真实PASS：128 Gallery范围、2 Query，Source文本样本44/4 batches，图像Source前向0、Generator前向0、backward0、Optimizer0，峰值reserved2048MiB，执行约20.04秒。作者发布UAP与通用词保持，Source参数/buffers exact不变。此阶段仅证明源攻击文本链路，尚未取得Target检索指标。

IRRA真实权重加载后、图像扰动及特征生成前，在target.py:74调用payload_digest(raw_noise)失败。原始作者UAP是CPU FP32叶Tensor，requires_grad=True；provenance.py:17拒绝带梯度标志的持久化payload，因此报ArtifactError: unsupported native payload or non-CPU/gradient Tensor。这不是OOM，不是进行了UAP优化，也不是论文攻击算法失败。Target没有pixels/feature/score单元，只有started、protocol、matched-projection和原FAIL。

00:41:12 +08:00监督器FAIL exit=1 line=34，停止全部后续链。未执行RaSa Smoke、其余数据集、任何SEP GPU或Full；没有覆盖/重试Attempt。

## 最小后续建议（未执行）

只读UAP加载桥需显式detach冻结Tensor，并验证前后像素与文件SHA不变；不能修改官方UAP文件、放宽payload_digest或改算法。应新增CPU回归覆盖作者真实requires_grad元数据，并建立与原FAIL隔离的新Target Attempt及明确旧Source只读复用来源。不得直接重跑本轮启动器或覆盖旧FAIL。本轮仅保存失败证据，没有自动修复或重试。

## 日志

控制目录：/home/lzf/ldx/tmp/cpgc_sep_full_v1_20261028/
输出根：/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v1/
原始FAIL：smoke/cpgc/cuhk/irra/FAIL.json。详细路径、SHA与原始Traceback见SOURCE_EVIDENCE.json。历史模型、数据、官方源码与旧实验没有改写。
