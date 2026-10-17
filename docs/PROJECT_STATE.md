# 项目当前状态（唯一权威入口）

状态来自Git、实际代码及不可变证据。历史报告为当时快照；架构见ARCHITECTURE，实验见EXPERIMENT_REGISTRY，设计边界见ADR。

## 当前阶段

**Stage4C-3唯一IRRA-only GPU Attempt PASS：IRRA_MATERIALIZED_SMOKE_VERIFIED。** 仅固定两张CUHK官方Test Gallery及两条Query，复用旧真实RGB/Unicode，不重新生成G/H。真实物化输入→IRRA特征/四组2×2 Score等价通过；不代表Full或跨模型攻击成功。

- 基准9f1f943ed8486a8c2889ab272764e5707a431552。
- 分支codex/tbps-framework-stage4c3。
- 工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4c3-20261017。
- 实现Commit由后续文档提交登记。
- [完整特征/Score原始数值、资源、证据和边界](framework/STAGE4C3_REPORT.md)。

## 固定输入与来源

Gallery cuhk:37132:0/index0、cuhk:37147:0/index15；Query cuhk:37132:0/index0历史2编辑、cuhk:37147:1/index31固定NOOP/K0。

RetainedInputs只读原/home/lzf/ldx/outputs/tbps-framework-stage4c2/materialized_smoke_v1/四单位；原Plan ID 7c0c00daf022bfcba94d6e3583c6de26c89b58c0e9c6696449cdbadb1ab7072d、Claim/生成证明、完整动作和原整体FAIL保持，不复制或重签旧产物。两图预算0.0313725471496582≤8/255。IRRA实际消费的四份原/Adv FP32 RGB与保存Tensor逐值exact；四条文本Tokenizer ID与旧审计exact。

G10/H7本次恢复/前向均0，仅SHA核验原来源；IRRA官方Checkpoint 9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009恢复1次，eval/inference、参数冻结，未修改官方加载器。

## 实测数值与执行

- clean_gallery/image_gallery两组[2,512] FP32特征误差0、bitwise exact。
- clean_text/adv_text两组最大误差均1.9371509552001953e-07；非bitwise exact，满足固定1e-6。
- Score最大误差：Clean7.450580596923828e-08、Image-only1.0151416063308716e-07、Text-only8.940696716308594e-08、Joint1.0151416063308716e-07；均≤1e-6。
- IRRA图像/文本样本前向4/4，8次batch1，started=completed；G/H0，Backward/Optimizer/JVP0。
- IRRA 367参数Tensor及0 buffers前后摘要exact，参数grad=None。
- 没有Ranking、mAP、Full计算、训练、OOM、中断或重试；GPU进程已退出。

## 资源授权与实测

本轮用户明确取消旧2GiB进程RSS硬限制，旧配置/证据不改；新配置rss_hard_limit_bytes=null。系统可用RAM预检≥8GiB、运行时整机可用RAM低于2GiB才安全停止，300秒时限保留。此处2GiB是整机安全余量，不是进程上限。

初始可用RAM123902926848 bytes、GPU空闲24051MiB/无进程；峰值进程RSS2524065792 bytes=2.350719451904297GiB，GPU allocated922.380859375MiB/reserved1002MiB；资源监视26.17609791457653秒，无异常。未执行Stage4C-2M、未优化或替换IRRA加载器。

Stage4C-2R的资源阻断是旧授权/旧上限下的历史结论，已由本轮明确新授权取代；其原记录不重写。

## CPU与保护

532框架+36历史基础=568 CPU tests通过，14新增，38.11秒。严格CPU闭包75项G/H固定来源、33个旧实际执行文件、5个官方IRRA文件通过，CPUStartupGate隔离退出后才初始化CUDA。原启动流程Mock回归通过。

后置CPU审计PASS：7991历史文件、3472Joint Full缓存、原FAIL Attempt70文件SHA/mtime exact；7个受保护旧工作树HEAD/clean保持；新回执及四个feature Artifact完整，无pending。未改变原FAIL为PASS。

## 不可变证据

/home/lzf/ldx/outputs/tbps-framework-stage4c3/irra_only_v1/：

- attempt/manifest.json SHA0787cce4deb3038cedc1cc55dbc448dfcb321d529222d1520714dfcea6c27035。
- result/manifest.json SHAe6aa5f89e9fce861bfd900c5160629b6f54bc196a858b43ee341116539062f70。
- result/report.json SHAc85ffc5a315dcfe233a2c44459400f2489059bedcc2ead6ca6aaf728436de98f。
- 四family features-*各保存实际[2,512]新特征和独立来源Manifest，详细SHA见报告。
- CPU后置审计/home/lzf/ldx/outputs/tbps-framework-stage4c3/cpu_verified_v1/result/manifest.json SHA40e4308c251dfcb42ad9f29b4d631c5e93dd598f4c7e4521ae601859f56963c4。
- stage4c3_gpu_attempt_v1.log及CPU测试日志在/home/lzf/ldx/tmp/，仅资产，不入Git。

## 下一步与权限

固定环境/两图两文的IRRA物化端到端等价由PENDING更新为VERIFIED；全量3074 Gallery物化、6156 Query端到端、Full排名、RaSa/APTM及新训练仍未验证、未授权。唯一irra_only_v1已消费，入口拒绝重试，没有resume或扩规模参数。

完成提交推送后停止，等待审查；不自动开始下一阶段。

## 既往阶段索引

- Stage4C-2R：9f1f943ed8486a8c2889ab272764e5707a431552，[CPU顺序修复/旧产物审计报告](framework/STAGE4C2R_REPORT.md)。
- Stage4C-2：fcf097d3ab31785381947289a680e5f5ccbf9d42，[原唯一Attempt FAIL报告](framework/STAGE4C2_REPORT.md)，整体状态永久FAIL、G/H物化局部PASS。
- Stage4C-1：4d3bc025a7da593368ebd22cc05db1ebe787b446，[来源与存储验收](framework/STAGE4C1_REPORT.md)。
- Stage4B-2：c09754e6afc8cf53a6c0c6e8715b6d4eace80946，历史Full特征METRICS_REPLAY_VERIFIED，区别于新物化端到端小样本证明。
