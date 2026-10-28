# 共享GPU最小资源适配

基于c134cb394dd8a21a39c7bc9aa2642fe33e38a6d0建立独立codex/cpgc-sep-shared-resource工作树；原工作树、原配置及所有历史产物不变。

派生配置cpgc_sep_tbps.shared.server.json：显存准入10240→8192MiB（先仅用于容量Smoke）；Query Chunk128→32；RaSa融合batch128→16。完整Gallery、Query、原始精度、原生模型及Top128候选和全部128对ITM均不变。无Loss/预算/DPP/候选修改，无UAP训练。批量拆分与CUDA计算可能产生浮点末位差异，不声称逐位等价；各组Clean/Attack统一使用新配置。

只有对应方法六组同来源真实Smoke和CPU审计PASS，且Source/Target每项实测peak reserved+2048MiB≤8192MiB，才允许Full；另在启动每阶段时重新核验即时资源。容量验证失败不重试、不降低门槛，不终止外部服务。

正式输出与控制目录仍使用用户指定路径，启动前须为空；原子提交、唯一Claim、flock及失败停止不变。Source模型和公开UAP/词表直接只读复用，不重新下载。

后台串行执行C-PGC六组Smoke/审计→CUHK/RSTP/ICFG六组Full/审计→SEP六组Smoke/审计→六组Full/审计。全部完成后在独立codex/cpgc-sep-full-results文档分支发布真实CSV和报告，不改执行源码或打断GPU。若发布失败保留publish.log，不能因此重跑Full。

本准备文档不代表GPU已经PASS，实际以控制phase、各Attempt completed/FAIL和独立审计为准。
