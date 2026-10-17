# generation_v2失败后的最小修复与首图Claim恢复：CPU验收报告

基准dd0a213f2ac1b787412b755372c378ad49f31f9c；独立codex/tbps-framework-stage4d2r2，工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r2-20261017。代码实现Commit：cd6c222fc6751e4dbefc44bb7869a9c08d93a680；本后续文档提交登记。

**修复、647项CPU回归及真实来源/恢复资格审计通过。未运行GPU，generation_v3尚不存在。** 当前仍是Image0、Claim1、Proof0、pending0；CPU资格通过不等于首图已恢复或Joint已完成。

## 原失败已确认

generation_v2恢复G10一次，实际G started/completed=0/0、新图像0。在原Stage4D1生成事务发布首图Claim后，clean_reader→_LegacyG.load_pixels→原pixels_from_pil→torchvision Resize被gpu_safety的全局非G模块拦截拒绝。

Resize是torch.nn.Module形式的正常PIL预处理；ToTensor是普通可调用对象。本次是监督范围过宽，不是G算法、权重、图像预算或GPU资源错误。原v1/v2 FAIL保留，不改签或提升状态。

| 原v2证据 | SHA-256 |
| --- | --- |
| attempt/manifest.json | 40778bdc08e9729202c83bb442353013d6a7bc854cbca13992b27a806af889d6 |
| gpu-start/manifest.json | 7b18ba2fd819a50529c4c2cf5eb1e105873c34af2b292ca15636e8dc5bb7366f |
| gpu-result/report.json | 176ae52ffc1851a3d710c445268569a987baaed912d95696941ec7e73c052d13 |
| result/manifest.json | 9b2dae20980ce9e1943fcaada80e2d73539fd09fb9ea3d4ada106e077d79efa9 |
| 原首图Claim manifest.json | 21ed14c935fdde887cfc6afcbdb67b6e6c6687085f6b8a46e8c3aed5ad5ea1e0 |

原Claim绑定cuhk:37132:0，unit_id=50b19770bb01c5c7e52237a4af73e4186342e77ccafe06204a8c3c91ecbfe3b7，原图SHA6756e64194c981af646ade08ea47de531e4f95c057393e836cb0b3609fb465de。原Plan保持7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5。

## 最小实现范围与调用链

| 文件 | 变更 |
| --- | --- |
| gpu_safety.py | 作用域限定的原像素读取许可，非G模块保护仍默认拒绝 |
| claim_recovery.py | 独立零前向首图证据校验、一次_prepare恢复、原Runner衔接及恢复Proof验收 |
| full_gpu.py | 固定v3；绑定恢复与预处理来源；CPU预检/fresh_images及GPU调度接线 |
| full_gpu_audit.py | 新旧Claim、恢复Start/Proof与正常生成Proof一致性、精确恢复计数 |
| stage4d2r2_generation.server.json | 原范围/资源/Plan不变，独立4项入口、预处理及恢复来源SHA |
| stage4d2r2_recovery.server.json | 固定原v1/v2文件、首图Claim与原30797文件库存证据 |
| 两个新增test_stage4d2r2_* | 真实预处理、实际child、恢复事务、拒绝场景及3074顺序 |
| test_stage4d2r_startup.py | 历史v2配置仍由其SHA固定原入口验证；原控制断言保留，新v3另测 |
| 统一状态/架构/注册表/ADR与本阶段文档 | 更新真实完成、失败保护、执行命令及边界 |

未修改任何冻结Stage4D1文件或历史prototype、G/H/IRRA、Checkpoint、像素投影、预算、Unicode策略、缓存或旧完成回执。生产仍通过冻结Stage4D1/src使用原Plan/插件/算法。

预处理许可仅在原_LegacyG.load_pixels内、按实际G顺序处理已绑定图像时生效；原backend、pixels_from_pil及torchvision源码单独SHA验证。只允许原Resize类/forward、RGB PIL输入、bilinear384×128、原max_size/antialias，无实例forward覆盖、Hook、参数或buffer。作用域外Resize、其他非G神经模块、读取器/图像/参数不一致均拒绝。原函数输出直接返回，没有替换预处理计算。

恢复必须同时验证原FAIL/来源、G加载1与实际前向0/0、唯一准确首图Claim、Proof/图像/pending均无，以及当前Full恰好等于原30797文件＋这3个Claim文件。只在同一G工厂会话中，对首图一次原generate_image_once._prepare开放单线程局部能力；原_finish继续生成正常Proof，旧Claim不变。其余3073图调用原stream_images，其他未完成Claim仍拒绝。

v3额外发布claim-recovery-start及claim-recovery-proof，绑定原失败/Claim、新Attempt、正常Proof与新像素单位。原Runner内部跳过刚完成的首图1次；本次真实生成总数为3074、对外skipped=0、recovered_first=1，同时明示legacy_stream_generated=3073/legacy_stream_skipped=1。没有重复G、重复加载或伪报旧缓存跳过。

## CPU测试与边界

最终647项通过，84.36秒，退出0：原626项＋新增21项。两图Fixture实际调用child、原StartupGate、安全包装、PIL/Resize/ToTensor、CPU Mock G Hook、原Claim/Proof与保存读回；CUDA reset和权重入口Mock，真实CUDA init/lazy/native入口拒绝。两图仍被原3074固定总数检查拒绝，不放宽正式作用域。

覆盖：

- 真实原预处理、作用域外Resize/其他网络拒绝，非绑定图像、伪读取器/SHA、错误Resize参数/Hook拒绝。
- 原Claim保持，首图一次恢复、正常Proof及独立恢复Proof、模型状态不变、真实Hook顺序与次数。
- 非零/不完整/布尔计数、错Plan/身份、篡改失败报告/Claim、其他Claim、Proof、图像、pending、未知库存文件拒绝。
- 恢复失败保留原Claim和新的消耗证明，不能重复恢复；局部_prepare及Torch函数/导入状态恢复。
- 3074成员调度使用原stream_images完整循环，顺序及每步累计计数exact；仅该压力案例Mock像素事务/封存以避免5GiB测试写入，两图案例独立验证真实事务。
- 实际child中正确启动门禁、两项安全包装先于G、真实预处理与恢复Proof组合通过；原611＋Stage4D2R15项均回归。

日志/home/lzf/ldx/outputs/tbps-framework-stage4d2r2/cpu_tests_v2.log，
SHA efcd7faff737ef9b69d5927c8be3c1c2e8d8ab9f0aca62504692e8bd83abf3c6。

## 真实来源、恢复资格与历史保护

独立CPU审计退出0，STAGE4D2R2_CPU_REPAIR_AND_CLAIM_ELIGIBILITY_VERIFIED / ZERO_FORWARD_FIRST_CLAIM_RECOVERY_VERIFIED。

- 75项固定来源、33份实际导入文件PASS，forbidden_attempts为空。
- 全3074原图身份/SHA/顺序、6156历史H文本/动作核验通过；6116编辑、40 NOOP、13537替换保持。
- 原30797发布文件＋原首图3 Claim文件，共30800个已有Store文件SHA/mtime不变。
- 7991历史文件、3472历史缓存、70个Stage4C2旧FAIL文件不变。
- generation_v1及generation_v2各16文件SHA/mtime不变，原整体FAIL及旧Claim保留。
- 冻结Stage4D1工作树HEAD/clean、Stage4D2R原工作树dd0a213/clean不变；更早原Stage4D2未提交文档SHA/mtime不变。
- CPU隔离退出后函数/导入/锁恢复，CUDA_initialized=false；未查询GPU、未加载真实模型、未创建v3、新图像0。

本次CPU审计160.64618672430515秒，其中全文/历史检查124.95993159338832秒、记录Peak RSS612151296 bytes；不是GPU生成耗时。

证据/home/lzf/ldx/outputs/tbps-framework-stage4d2r2/cpu_audit_v2/result/：

| 文件 | SHA-256 |
| --- | --- |
| manifest.json | 73d895b065cac4d3abe9316478811fa5a6cc1fa6eb9414b2f8532668d54e61fc |
| report.json | 030bbdedf5f51becdbaec613219a556a923f3442ee061643da03deca41e4ca7c |
| eligibility.json | 811c5fb0cf2ecf48ab5d4091dbd956d6612a9493e7c8076fd7c0fcefcf3933e6 |

首次新增CPU来源核验因新清单缺少原snapshot的path字段而严格拒绝；逐项确认旧SHA/大小/mtime没有变化后，仅补齐新清单的完整字段，并新增缺字段拒绝测试。失败日志cpu_audit_v1.log、原脚本及第一轮测试日志保留；未改旧证据或放宽字段比较。修正后完成上述第二次CPU核验。没有因此执行GPU或正式Attempt。

## 新来源与后续执行

| 来源 | SHA-256 |
| --- | --- |
| v3执行配置 | f971511be22addb02aa160cb279b0d1c3e7c88341cb4d7ee705c39e55f1ad5ff |
| 首图恢复配置 | 429209225c9a947b1dd254b51addbf41c9ef5a5e8dc0532ea7dfbc3396604bd6 |
| claim_recovery.py | 2c97f50c93abcc8152d10eb06fbc05f6d230687edfc6dfe3b6a1a5533639f334 |
| gpu_safety.py | dbb1e9f0dcff994aebe096e9a04b4f3cea1cd8dde9aa21a48877705016d2b409 |
| full_gpu.py | d0a4040021128066eace97448cd7d61e33d65a797aabd0ec896d2a413b17bc8c |
| full_gpu_audit.py | 9006c6eec3e10ad8aa380e5e6fa9bc3581384e625e5ada2615b9d590052cc628 |

未来准确预检、显式GPU及独立CPU审计命令见[执行说明](STAGE4D2R2.md)。原7200秒、GPU/RAM/磁盘/inode门限不变。只在后续审查和明确授权后执行一次generation_v3；失败保留现场，不自动重试。

当前首图仍未恢复生成，Proof0；Image/Joint全量PENDING。无需人工删除或移动旧Claim、旧文本、缓存或失败目录。本轮停止，不运行IRRA Full、RaSa/APTM或训练。
