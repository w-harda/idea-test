# 六组跨模型迁移v2：修复冷启动并加入图像增强版

## 范围

三版均固定各自IRRA训练的Round60配对Checkpoint：Cooperative、原Push-pull、Push-pull＋图像增强。每版迁移至RaSa和APTM，共六组；完整CUHK-PEDES 3074 Gallery／6156原Query，保存Clean与Joint及R@1/R@5/R@10/mAP，另保留mINP。Round60在运行前固定。

只读复用三版各自原IRRA Full封存的FP32攻击RGB与Unicode文本，不重新训练、不重新运行G/H生成、不加载IRRA，不计算JVP或目标梯度。两版v1完整输入CPU审计继续有效，沿原生产者身份和SHA只读复用；增强版新增完整独立CPU输入核验。三份G/H SHA均不同；每版自己的图文共享给两目标，禁止跨方法混用。

## 旧失败原因与修复

v1在首次导入torchvision前建立CPUStartupGate快照；torchvision间接导入setuptools与PyTorch分布式辅助模块，会将setuptools/_vendor和自动生成的临时Python目录加入sys.path。两个全新CPU进程均复现：仅sys.path多两项，函数、meta_path、bytecode开关、导入锁和CUDA均未泄漏，原快照因此拒绝。失败发生在目标模型加载和CUDA之前，六组训练产物未损坏。

v2在快照前完成受控依赖bootstrap。bootstrap禁止torch.load/jit.load、神经前向、CUDA；核对函数/meta_path/bytecode不变、原搜索路径顺序不变，新增路径最多两项且仅允许对应setuptools目录和本任务TMPDIR内的临时目录。然后仍使用原封不动的CPUStartupGate，在其严格快照中核验目标来源、Tokenizer、输入索引与完整数据身份，退出护栏后才允许CUDA。没有删除或降低隔离断言，没有关闭SHA检查。

`startup.prepare()`由正式GPU入口和`--action startup-check`共用。新增回归同时覆盖三版×两目标接线、隔离未恢复/未知路径/NN/权重/CUDA拒绝、完整缓存与评分的独立CPU审计、缺失或Smoke回执拒绝正式报告。另在全新解释器中执行真实来源/Tokenizer/完整索引的冷启动核验，不依赖pytest提前导入的状态。

## 预算与评分

图像在封存的384×128 FP32 RGB空间相对原图严格L∞≤8/255，容差0；逐图SHA及原始PIL Resize/ToTensor像素exact核验。目标随后执行自己的原生Tensor预处理：RaSa 384×384，APTM 384×128，不进行JPEG/PIL量化或目标自适应扰动。

文本为原Natural-K，可编辑多个原词，每个原词最多替换1个Unicode字符；原contracts/constraints逐条重放。目标Tokenizer仅编码固定文本并统计Token实际变化，不用于修改或筛选编辑。

RaSa保留原Top-128与ITM正类logit；APTM保留Top-128、原填充规则、完整6156×3074矩阵归一化及0.002全局项。各组独立ModelSpace/Tokenizer/特征/原生Score缓存，精度FP32/math/no_tf32；图像、文本和融合microbatch1，缓存16，Query chunk128。正式每组image6148/text12312/fusion1575936，与真实神经Hook计数核对。Smoke仅10图10Query，不作为正式效果。

## 执行与审计

独立分支`codex/gh-cross-model-v2`，工作树`/home/lzf/ldx/projects/idea-test-tbps-gh-cross-model-v2-20261113`。旧v1工作树、输出、started.json、日志、FAIL不修改；新配置固定13份旧控制文件SHA，每次预检复核。增强版原训练、六节点权重、IRRA Full与原审计不改。

新的CPU准备完成后，后台顺序：

1. 逐方法核对三份完整输入审计与原文件SHA/stat。
2. 三版×两目标共六组全新进程CPU冷启动完整输入核验。
3. 六组真实Smoke及各自独立CPU审计，全部通过才执行Full。
4. 六组正式Full及各自独立CPU审计。
5. 全部PASS后输出12行Clean/Joint指标、来源和中文报告，更新PROJECT_STATE，仅提交报告到本分支，push后核对远端SHA。

独立CPU审计禁止NN前向、CUDA、导数、Optimizer；核对全部特征和Score块、各版Checkpoint/输入绑定，重建RaSa全部Score块和APTM完整矩阵归一化，重算完整排名并要求五指标exact。最终重新核对历史文件完整SHA/stat。任一真实异常立即FAIL，保留原Attempt，无自动重试。

输出`/home/lzf/ldx/outputs/tbps-framework-gh-cross-model-v2-20261113`；控制`/home/lzf/ldx/tmp/gh_cross_model_v2_20261113`。单模型顺序使用共享GPU，不设固定VRAM floor，不干预外部进程。TMPDIR、HF/Torch/CUDA缓存均在/home/lzf/ldx。只确认脱离终端的启动身份，之后由用户检查进度。

```powershell
ssh tbps-server "/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-cross-model-v2-20261113/operations/gh_cross_model_v2_status.py"
```

最终成功`SUCCESS_SIX_FULL_TRANSFER_TESTS_PUBLISHED`；失败`FAIL_STOPPED_NO_RETRY`。修复前后证据见GH_CROSS_MODEL_V2_CPU_EVIDENCE.json；运行和迁移结果以新后台实际回执为准。

## 固定Checkpoint身份

| 方法 | G SHA256 | H SHA256 |
|---|---|---|
| cooperative | d0920c614b7f2ad01e564ac803d933b5ba97d91e512324792f43091e086e41c7 | 4186211b933f8a18033828506c678d885b0594b28c8828a41b114f050cc5c6d3 |
| attraction | 8066e9017a7151688a8aaad02de4f8d2c29fbf027d230c6f00914795d6e3d738 | 9f70acd11da097c42253f316bce5ffe0feb610b36c2d7241684067ff3b7698b1 |
| augmentation | 5e81cd10d58fcc3e2eebad6a20ea893912986ba0cb34f67709e4f1dfdbd5070d | 536efd88c7eebe9888084355b69b786cd3c8ae81cf3c084d0badf398356f7183 |
