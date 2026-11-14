# RaSa/APTM增强Push-pull：发布回执与显存修复

## 原因与修复

共享显存Attempt保留在gh_native_aug_shared_20261114。RaSa在Reader首批CUDA计算中OOM，APTM已完成全部准备和短Smoke更新，却在发布完成回执时漏传ArtifactStore.publish的provenance。独立检查还发现Bank完成与评测claim两处同类错误，本次三处一起修复，未修改ArtifactStore契约或全局JSON上限。

新分支codex/gh-native-aug-repair，新输出tbps-framework-gh-native-aug-repair-20261114、新控制gh_native_aug_repair_20261114，旧失败、started、缓存和权重不覆盖。原Attempt正式G/H Round1未开始；本次正式G/H仍从新初始化与空Adam开始，旧诊断G/H不复用。

## 现有准备产物复用

- APTM：原preparation/completed的Manifest/SHA、34054张官方Train原图来源、256维图库/2048配对锚点、语义基正交性、gamma3无cap4选择和更新预算重新核对。原Reader/basis/selection/gallery等payload逐字节复用，新来源回执显式绑定原完成Manifest和新配置，不重训已完成Reader。
- RaSa：原prefix缓存没有完成Manifest，不能当作完整准备已通过。本次结合原被冻结producer与Reader阶段OOM证据，独立CPU完整检查67946条浮点特征及官方Tokenizer内容Mask，记录10.4GB文件SHA与新恢复回执；所有旧数组只读。核验耗时约32秒。新作业复用该前缀，不重新编码全部文本，仍需完成Reader/语义基/图库/选择。
- 原始数据、旧失败及缓存的SHA/stat保护清单位于新控制目录，发布前后比对。

## 显存修复与数值

RaSa Reader改在CPU拟合，仍batch512、Adam1e-3和固定20轮；其CPU数值轨迹单独记录，不伪称与未完成的旧CUDA训练bitwise一致。完成后Reader回到GPU供冻结定位使用。APTM直接复用已完成的CUDA Reader。

RaSa未使用的EMA编码器、投影和训练队列转存CPU，逐值SHA核对完整官方参数/buffers保持；实际使用的视觉编码器、BERT和投影仍在GPU。Source只注册使用中的组件供G梯度检查，完整模型仍单独核验冻结状态。

视觉Block/SwinTransformerBlock采用非reentrant activation checkpoint，降低保存中间激活的峰值；eval的编码特征和梯度公式不变，重计算计数记录到账本。真实CPU官方权重探针对两源验证：特征逐值exact、输入梯度最大差0、完整Victim SHA保持；RaSa12个、APTM24个视觉块真实执行了反向重计算。此探针不替代正式GPU Smoke或Full。

保留原G/H架构、Push-pull KL、Local/Retain、三视图增强、8/255、原Natural-K和每词最多一字符。不设置最小空闲显存，不干预外部任务。13项既有CPU与3项新增回归共16项PASS，新增覆盖三处真实ArtifactStore发布和契约、CPU Reader反向更新。

## 后台链与检查

两个独立训练进程；各自GPU Smoke和独立CPU审计通过后进入60轮。每10轮保存G/H、双Adam、RNG，首个节点通过后各自创建三个独立评测进程，共六个、36组完整Joint Test。真实错误保留停止，不自动重试。

服务器检查：

    /home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-native-aug-repair-20261114/operations/gh_native_aug.py status

本轮只修复、完成必要验证及重新启动；启动核验后退出，不持续监控60轮。

## 修复后真实启动核验

执行源码Commit：3c4246eb43f0472627c12a9968b9e73cd4d477ef。

RaSa PID99632、APTM PID101317已独立后台运行。启动核验快照：RaSa已复用完整前缀、CPU Reader达到8/20；APTM已复用已验证准备，通过新真实GPU Smoke及独立CPU审计（9次G、2次H诊断更新），进入正式Round1的G训练，154/2048条、641次正式G更新。Smoke图像最大L∞0.03137043118476868；独立审计3.60秒、无CUDA/神经前向/JVP重入。

这证实此前Smoke发布漏参的真实失败点已越过；Bank完成和评测claim由静态契约与真实ArtifactStore回归覆盖，尚不把未来节点宣称已经完成。两条60轮训练/六节点Full仍由后台推进。本轮结束不持续监控。

原41份失败、started、完成准备等保护文件SHA/size/mtime核对保持；大前缀只读引用其新验证SHA，没有覆写。原源码和原FAIL不改。
