# ADR-0019：将局部/全局三视图输入加入push-pull交替训练

用户2026-11-11授权参考AoA、SGA、GLEAM增强图像输入并训练G/H。采用原始/局部/全局三个视图，G各视图Global Loss均值、H当前G三配对视图奖励均值，保持原优化器步数。属性Local/Retain只在原始视图计算；每视图扰动相对自身clean输入约束8/255，正式Test始终原图。AoA原文按第5页实现局部/全局结构，官方仓库暂未公开MSA代码；TBPS保留身体水平分区、去掉垂直翻转并保持3:1比例。只借鉴增强，不引入AoA DRAL、MLM或GLEAM Wasserstein目标。

该决定使G学习同一样本多种输入、H适配当前G的多种视觉状态，同时保持原Dynamic-K Unicode推理接口。增强及当前视图库均有recipe/SHA和独立数值/像素审计。预设60轮六IRRA Full不用于Test调参；跨模型泛化收益尚待独立迁移验证。旧结果与失败现场保持。
