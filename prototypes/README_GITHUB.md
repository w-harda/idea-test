# 图像条件生成器及冻结依赖源码

本次同步包含 attribute_image_generator_20261014 的训练、事务 checkpoint/恢复、独立生成、官方 Test 共享图库评测和 CPU tests，以及其引用的属性接口、S_raw、Dynamic Top-K、JVP、TTA、PID 指标 prototype 源码。历史 prototype 保持原实现，不代表全部是当前方法。

## 验证与边界

同步前在原服务器运行 test_generator.py、test_checkpoint.py、g_test_tests.py：94 passed。没有重新训练、运行攻击或完整评测。各源码文件按字节复制，SHA 见 SOURCE_MANIFEST.json。

GitHub 只归档源码、测试、配置和使用说明，不包含数据、模型权重、条件/特征缓存、原始输出或日志。AGENTS.md 和 Windows 项目管理文档不在本次提交中。

## 入口

- attribute_image_generator_20261014/README.md：离线条件准备、训练、恢复、一次前向生成。
- attribute_image_generator_20261014/README_TEST.md：官方 Test 条件准备及共享图库评测。

CPU tests（进入 G 目录后）：

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -B -m pytest -q -p no:cacheprovider test_generator.py test_checkpoint.py g_test_tests.py

## 外部资产与冻结快照

代码保留实际实验的服务器路径及 SHA 审计门禁，不能将仅下载源码视为可独立复现实验。运行需要原 IRRA 源码/权重、reader 权重、semantic bases、Stage01 标签、score/condition cache、来源审计文件及原数据；按各 README 的实际服务器环境准备。部分使用说明保留编写当时的 smoke 状态，不作为当前全量结果证明。

_frozen_sources/stage03-attribute-unit-v2-20261002/code 保存此前位于服务器外部 outputs 下的源码快照，未包含实验产物。现有实现仍读取原外部快照路径；此归档没有修改运行时路径、冻结实现或来源 SHA。需要恢复环境时，该快照对应 /home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/code。
