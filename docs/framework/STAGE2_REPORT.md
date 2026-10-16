# Stage 2交付报告：真实Dataset/Model Adapter基础

## 分支与范围

基准61471db976508411aafe58e97113322c14d6bbc7；独立分支codex/tbps-framework-stage2；独立树/home/lzf/ldx/projects/idea-test-tbps-framework-stage2-20261016。

实现Commit：55b83bb48307bebba3e8a3bf8881570091ef1d15。根AGENTS、PROJECT_STATE、ARCHITECTURE、EXPERIMENT_REGISTRY及ADR-0001/0002已先读取；新增ADR-0003记录成员权限与Model View决定。未移动或修改任何历史攻击程序/资产。

## 文件与接口

详细清单见[Stage 2说明](STAGE2.md)。新增datasets/{__init__,cuhk_pedes,historical_membership}.py；models/{__init__,base,irra_source,irra_io,irra,irra_cpu_checks}.py；preflight.py；固定来源配置；4个CPU fixture/tests文件；中文说明/报告及ADR。

src/tbps/config.py仅增强Joint组件权限及可用Adapter说明。Stage 1新框架文档修正“8个资产索引”；根旧.gitignore不改，通过src/tbps/.gitignore局部开放Dataset Adapter目录，防止源码被datasets/旧规则遗漏。

接口要点：

- CuhkPedes.manifest/public_image/public_text：保留原文、顺序、多正例及准确Span，公开投影不带标签。
- load_historical_partitions/require_member/reward_binding：验证实际历史成员；拒绝仅声明role的直接构造，奖励只接受实际Train。
- IRRASource.verify：只读SHA/YAML，模型权重训练来源与攻击训练来源分离；本地CLIP bootstrap也锁定。
- IRRAAdapter.from_source：真实加载默认拒绝，显式授权才可用；本轮未运行。inference/surrogate两个View参数均冻结，后者保留图像输入梯度。
- ModelAdapter.score不要求global embedding；Fusion Mock和37维表示通过接口测试。
- ModelSpace/InputBinding/RepresentationRecord/ScoreRecord绑定空间、Dataset Manifest、有序ID/输入SHA；不把同维度或同Query ID当作缓存等价证据。

## 真实数据只读核验

| 范围 | Gallery/图数 | Query/固定Caption数 | PID |
| --- | ---: | ---: | ---: |
| official train | 34054 | 68126 | 11003 |
| official val | 3078 | 6158 | 1000 |
| official test | 3074 | 6156 | 1000 |
| attack Train（official train内部） | 2048 | 2048 | 1861 |
| attack Validation（official train内部） | 128 | 128 | 121 |

三份历史Full协议完整身份与顺序均一致，ordered_identity_digest：
8773285858aec7b51e078339a4f37a1ea009cfd0de2cceb0cc2708ba868dadcd。

Train/Validation PID交集0、图像交集0；2176份G metadata SHA及实际身份核验通过。原annotation、internal split、G计划/完成回执、H资格、训练/Validation Gallery及训练访问顺序共同绑定。全部Caption保留，包括空格和多Caption，不因不可编辑而筛选。

成员摘要：f1a54c3caa57f2b0c9e1abf5f2a40b6282a1e1b002af383d12f94b370bfbcc66。
此摘要来自原来源及有序成员，不是攻击效果指标。

## CPU验证

| 验证 | 结果 |
| --- | --- |
| 框架CPU tests | 183通过：114项Stage 1回归+69项Stage 2新增 |
| 历史基础CPU回归 | 36通过 |
| 合计 | 219通过 |
| 官方Tokenizer对照 | 10种合成文本，77-token ID逐值exact；包括Unicode、长文本截断 |
| 官方PIL预处理对照 | 3组RGB/L/RGBA合成图，转换RGB后最大误差0 |
| 官方rank定义 | 4×12合成分数、多正例、37维输入，AP/CMC/mINP公式对照通过 |
| Mock编码器 | 冻结推理无梯度；Surrogate输入梯度非零；参数/buffers不变 |
| 嵌套模式 | 外部inference_mode下Surrogate仍可反传；推理结果可作为正常常量复用 |
| 缓存隔离 | 模型/预处理/Tokenizer/Dataset/输入编辑/顺序改变均拒绝误复用 |
| 无副作用 | 模块导入及未授权加载路径禁止Torch/官方model导入的子进程测试通过，CUDA未初始化 |
| 真实CPU preflight | STAGE2_CPU_PREFLIGHT_PASS；无模型、无排名 |

上述Tokenization是Byte-BPE字符处理，不是IRRA.encode_text。真实图片仅读取字节SHA，没有真实图像预处理或IRRA编码。CPU排名仅合成4×12，没有读取或重算正式Full。

## 官方IRRA契约及验证边界

权重SHA：
9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。
官方源码30份、原配置、BPE、CLIP bootstrap均固定SHA；IRRA fine-tune来源声明CUHK-PEDES official train，CLIP上游预训练另以bootstrap记录，不等同攻击2048 Train。

PIL RGB→Resize bilinear(384,128)→ToTensor [0,1]。mean/std Normalize在可微forward内执行一次；不在Adapter中改动扰动或投影。图像CLS、文本EOT由官方encode_image/encode_text提供，外层float32 L2 Normalize，评分Query×Gallery点积。官方rank降序默认非stable、不新增tie规则，同PID全部正例，无camera过滤，指标百分数。

已验证：源码/权重字节来源、数据身份、CPU Tokenizer/预处理逐值一致、合成排名定义、Mock梯度与冻结View。尚未验证：新加载器实际构造IRRA、GPU原生特征/分数数值、真实输入梯度、GPU显存/耗时及运行中参数/buffers SHA。因此不能宣称真实模型端到端等价已验收。

新空间只锁4个相关Adapter producer，不锁整个框架目录。历史IRRA特征未改写、未复制、未标记为新空间；实际复用仍需明确的只读兼容验证。

## 历史保护与证据

7991个已记录文件SHA及mtime_ns保持不变，涵盖历史tracked源码、固定结果/协议/Checkpoint、官方来源、2176份G metadata及选中Train/Validation/Test原图。不是重新审计全部历史Full特征或condition Tensor。

原实验树HEAD仍40c7476，Stage 1树仍61471db，均干净。所有新增写入仅独立开发树和/home/lzf/ldx/tmp/tbps-framework-stage2。没有模型前向、真实模型加载、GPU任务、训练、后台进程或Full指标重算。

CPU证据（不提交Git）：
/home/lzf/ldx/tmp/tbps-framework-stage2/{cpu_tests_final.log,legacy_cpu.log,cpu_io_final.log,preflight_final.log,protected_before.json,protection_result.json}。

## 下一步仅建议、未执行

申请独立进程冻结IRRA一次加载，2张已核验Train原图+2条Caption，对照Adapter与官方Backend原生接口；另1张原图做输入梯度检查。最多5个图像样本前向、4个文本样本前向，无G/H、无Optimizer、无检索指标。

预先固定fp32/数值max_abs_error<=1e-6，检查参数/buffers SHA、有限性、范数及GPU资源；异常停止，不临时放宽。此建议不授权Stage 3 Attack Plugin/统一Engine。

完成源码与CPU交付后提交推送并停止等待审查。
