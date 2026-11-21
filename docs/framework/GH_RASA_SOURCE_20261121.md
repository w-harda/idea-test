# RaSa 源 Push-pull + 图像增强 Joint

用冻结的官方 RaSa 作为训练代理，在 CUHK-PEDES 上从零训练 G/H 60 轮。沿用原 Push-pull、三视图增强、FP32 图像预算 8/255 和 Natural-K 文本预算（每个原词最多替换一个 Unicode 字符）。每十轮保存配对权重和完整训练状态。

本次复用上一轮已完成的 RaSa Reader、语义基、属性选择和干净训练图库，核对完整 Manifest、SHA、34054 个图库来源及 2048 个训练样本。旧作业在 Smoke 阶段因两个项目作业同时占用显存而失败，因此没有正式 G/H 权重可续训。RaSa 官方权重在 CPU 严格恢复，当前计算所用模块迁至 GPU，迁移前后完整模型 SHA 一致。先做短 Smoke 和独立 CPU 审计，再进入正式训练。

训练结束后先评测第60轮，随后补齐 10、20、30、40、50 轮。每轮生成并审计一份共享图文输入，供 RaSa 白盒、迁移到 IRRA 和迁移到 APTM 使用。每个目标和权重节点有独立评测进程和日志。按顺序执行 GPU 评测，避免本项目自己的多个模型重叠装载，启动时没有固定空闲显存门槛。每组走原生全图库评分和独立 CPU 审计。评测范围为 3074 张图库图像和 6156 条查询。全部通过后汇总指标到输出目录下的 rasa/summary/completed。

服务器查看训练、评测进度和已审计的指标：

    /home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-rasa-source-20261121/operations/gh_native_aug.py status

运行控制目录为 /home/lzf/ldx/tmp/gh_rasa_source_20261121，输出目录为 /home/lzf/ldx/outputs/tbps-framework-gh-rasa-source-20261121。正式结果以独立审计回执为准。
