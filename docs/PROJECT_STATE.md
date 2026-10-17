# 项目当前状态（唯一权威入口）

状态依据Git、实际源码及不可变证据。历史阶段报告是当时快照；架构、实验来源见ARCHITECTURE、EXPERIMENT_REGISTRY及ADR。

## 当前阶段：Stage4D-2R CPU-only修复与验收通过，GPU待授权

基准883c4be59284b79f475412eb40ea6d96e8b1742a，独立codex/tbps-framework-stage4d2r；工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r-20261017。实现Commit由后续文档提交登记。

full_gpu.child改为CPU校验/隔离退出/快照恢复→资源及原线程/seed/精度设置→initialize_cuda→立即安装两项GPU安全包装→G工厂。原CPUStartupGate、算法、权重、投影、预处理、加载/Hook计数、优化禁令、Stage4D1源码不改。

新增stage4d2r_generation.server.json及derivation，仅绑定generation_v2和修复入口来源SHA；原config保留。原Plan保持7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5，旧6156文本不重签。完整方案/未来命令见[STAGE4D2R](framework/STAGE4D2R.md)，[验收报告](framework/STAGE4D2R_REPORT.md)。

## 已完成验证

- 626 CPU tests通过，65.47秒，退出0：原611项+新增15项实际child组合时序/禁止操作/异常恢复/Fixture封存测试。
- 严格CPU来源闭包PASS：75固定来源、33实际导入文件；全3074原图/6156文本及原Plan核验通过。
- 30797原Stage4D1文件、7991历史文件、3472缓存、70旧FAIL及16个Stage4D2 FAIL文件SHA/mtime保持。
- 原Stage4D2工作树HEAD、4份未提交Markdown及未跟踪执行报告SHA/mtime/Git状态原样保留。报告完整字节exact收录新分支，SHA dca03f00ccc8333e1c13eec310f24d5601fa37eb9c00f45a9adc12cff1a9c143。
- 只读审计STAGE4D2R_CPU_REPAIR_SOURCES_AND_PROTECTION_VERIFIED，142.45755676925182秒，CPU上下文/函数/导入/锁恢复。
- 本轮真实G/H/IRRA加载/前向、CUDA初始化、GPU Attempt、新图像生成均0；GPU资源未查询。
- git diff --check及提交范围检查完成；提交状态由最终发布记录登记。

CPU证据/home/lzf/ldx/outputs/tbps-framework-stage4d2r/cpu_audit_v1/result/，
Manifest SHA24a39edf72666ae9279009e4e97751d51c268787a29a0a35f5944d987a690461，
report SHA381f1434f5e0df5a49fc5a4ab58a6cb95726ca4dcf63fe142c620b7fb0246682。

## 原失败与当前边界

generation_v1唯一正式Attempt已消费，永久FAIL，不复用、不覆盖。原报错为安全包装在初始化门禁前改变torch.load/autograd函数快照；G恢复/前向0、新图像0。[原正式FAIL报告](framework/STAGE4D2_EXECUTION_REPORT.md)已保留完整内容及证据。

原H Text 6156/6156保持TEXT_FULL_MATERIALIZED，6116编辑/40 NOOP/13537替换。Image 0/3074；Image/Joint全量、旧两图新输出对照及正式独立Joint审计均PENDING。Fixture封存不提升真实完成状态。

## 下一步与授权

等待源码审查及用户明确授权新的generation_v2唯一GPU Attempt；届时重新完整CPU与资源预检，绑定修复Runner SHA，原Stage4D1库/Plan/文本不变。当前v2不存在，无生成进度需要恢复，无旧文本或Claim需要清理。

本轮完成后停止；不得自动生成3074图、封存正式Joint、执行IRRA Full/RaSa/APTM或训练。CPU修复通过不表示GPU数值、全量或跨模型成功。

## 历史索引

- Stage4D2原CPU实现424b9683162e96da2f9a3ee9151946c02b7b124e，文档基准883c4be；资源阻断快照见[原报告](framework/STAGE4D2_REPORT.md)。
- Stage4D2 generation_v1 FAIL：result Manifest de017a15233cc57184108c5a10aaa6effe34f49e7afb736f7b76de3833ec8174；旧只读保护审计通过，不能替代生成验收。
- Stage4D1：82fe88a5a74955838fc6341c1ed9a46921c94d7c，6156历史文本物化/独立CPU审计通过。
- Stage4C3：a5a6f6ae8c9200482ae0c70d228e1873d3ca4dd2，固定两图两文IRRA端到端等价PASS。
- Stage4C2：fcf097d3ab31785381947289a680e5f5ccbf9d42，原整体FAIL/局部G-H PASS保持。
- Stage4B2：c09754e6afc8cf53a6c0c6e8715b6d4eace80946，历史IRRA特征指标重放通过，不是新RGB生成。
