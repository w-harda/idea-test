# Push-pull＋图像增强：RaSa/APTM原生代理训练

用户授权CUHK-PEDES，两份独立从零G/H 60轮，代理分别RaSa和APTM。保留seed42、G Adam2e-4/(0.5,0.999)、H Adam1e-4/(0.9,0.999)、三头Byte策略/SCST、Natural-K gamma3/beta1、每个原词最多一个Unicode字符、8/255 FP32累计投影。H原BPE合法性保留，另加原生分词不得新增截断的文本规则。

## 模型适配

此前RaSa/APTM只有评测接口，本轮新增其训练代理。两者全局特征256维、中间层768维；不复用IRRA的512维特征/reader权重/basis/JVP。采用原Canonical标签和冻结PID划分，分别拟合原SlotReader(dim768)，Adam1e-3、固定20轮；text-only六层取第4层，保持2/3分割比例。在reader Train上估计语义轴centroid/正交基，原区域三slot平均、full-body span池化保持。

重新编码34054张官方Train图库和2048固定攻击训练成员；每条原文一次原生配对检索交叉熵梯度（温度0.07），投影S_raw、gamma3/beta1选择无cap4的K。G更新预算由代理特定K决定，不能称作与IRRA等更新次数。当前H文本重新计算原生JVP，零干预exact和中心有限差分2%门槛保持。

原始/局部/全局三视图、8块身体条带及五类变换完全复用增强版。G每属性一次更新，最后原图联合更新；Local/紧邻上一步Retain仍绑定原始视图。Push-pull KL温度0.07、pull/push各1。H使用当前G三视图库、三视图平均pull_gain+push、sampled−greedy SCST，不增加排名目标。

官方Tokenizer单句只加CLS、不加SEP。Fast仅提供字符偏移，ID逐值核对官方Tokenizer。官方utils父模块与冻结H的utils.simple_tokenizer子模块显式共存；冲突模块仍核验精确官方文件SHA。旧生产代码、原模型权重和历史实验不改。

## 执行链

两个训练作业train_rasa/train_aptm独立PID、日志、状态、输出。先等显存、准备原生reader/basis/gallery/selection，完成真正GPU短Smoke与独立CPU审计，才开始Round1。正式训练从空Adam/新G/H开始，保存初态及连续状态链。

每轮JSONL账本和2048×3视图库保留并独立CPU审计；严格逐行解析覆盖重复键、非有限数值、尾随，单行最多4MiB，不修改全局read_json限制。每10轮保存配对G/H推理权重与完整双模型/双Adam/双RNG状态。

每来源第10轮权重及训练CPU审计通过后，自动启动三个独立常驻目标评测进程，合计六个。每进程消费10/20/30/40/50/60权重，共36组正式Joint Full。每来源每节点由同模型进程生成一次共享攻击输入，两个迁移进程等待输入CPU审计后使用相同输入。

每组3074 Gallery/6156 Query，IRRA原生全Gallery余弦；RaSa原生Top128融合；APTM原生Top128及完整矩阵归一化。query_chunk128、fusion microbatch1。每组独立CPU核对来源、预算、全部评分块与完整排名。Test反馈不进入训练。真实失败保留停止，不自动重试；共享输入生成失败则迁移进程停止。

进程独立不等于无条件同时分配显存。训练准入空闲6500MiB、评分5000MiB、生成1800MiB；30秒构造竞争保护，不结束外部任务。WAITING_FOR_GPU不能报作正式训练中。

## 验证与边界

12项合成CPU回归通过：维度/梯度/参考detach、K0/聚合、无cap4、累计预算、多视图K+1真实Adam、严格JSON、原生截断mask、三个独立评测进程和失败停止。两源真实官方权重CPU探针验证256维图像梯度、768维前缀、JVP零干预exact及FD<2%、模型冻结。2048×2原生Tokenizer逐值预检通过。以上不是正式攻击效果。

GPU Smoke、60轮及36组Full尚未完成，只认后台不可变回执。开发期CPU预检错误/修复日志保留development目录，没有删除旧FAIL或重启历史实验。

## 路径与检查

工作树：/home/lzf/ldx/projects/idea-test-tbps-gh-pushpull-aug-native-20261114
输出：/home/lzf/ldx/outputs/tbps-framework-gh-native-aug-20261114
控制：/home/lzf/ldx/tmp/gh_native_aug_20261114

服务器查看全部进程、阶段和已审计结果：

    /home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-pushpull-aug-native-20261114/operations/gh_native_aug.py status

训练日志：

    tail -f /home/lzf/ldx/tmp/gh_native_aug_20261114/train_rasa/launcher.log
    tail -f /home/lzf/ldx/tmp/gh_native_aug_20261114/train_aptm/launcher.log

六个后续日志位于eval_<source>_to_<target>/launcher.log，状态为同目录status.json。助手仅核验部署后退出，由用户检查后续。只发布代码/配置/小型文档，不上传模型、原始账本、图库或运行日志。

## 部署核验快照（2026-11-14 14:39，Asia/Shanghai）

执行源码Commit：3c6800a3f5982148146bca7ee7cbd2e3ef541937。
冻结配置SHA256：fa2b2153505174c7b979df31d1ef0b618504552df8d34e2abc97dd7ccac25c13。

两个独立作业真实启动：RaSa PID3297036、APTM PID3297037。二者PPID1、PGID/SID各自等于PID、无TTY，关闭SSH后继续。部署时两者均WAITING_FOR_GPU，实测空闲3865MiB低于新作业6500MiB准入，因此正式训练Round1、GPU Smoke和后续评测都尚未开始，不能报作已开始60轮计算。

RaSa真实CPU JVP FD相对误差约0.0038925，APTM约0.0025661；零干预exact，图像梯度范数分别约0.05472/0.02920，均无Victim更新和CUDA。CPU探针不替代GPU Smoke或正式Full。12项合成CPU回归通过。开发期预检失败回执仍在控制目录preflight_，详细日志在相邻gh_native_aug_20261114_development；未删除或改签旧失败。

仅做上述启动核验，之后助手不持续检查。两个来源首组权重及审计通过后由已冻结代码自动创建六条独立目标评测进程。
