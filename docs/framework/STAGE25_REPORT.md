# Stage 2.5审计报告：IRRA真实GPU最小等价性

## 结论与Git范围

**PASS：唯一一次真实RTX 4090 Smoke完成，所有官方/Adapter数值对照逐值一致。** 输入梯度有限且非零，模型冻结状态及参数SHA不变。没有GPU重试、G/H、Optimizer、攻击Loss、训练或Full排名。

基准：`4e2a5838693c5bc04a22b31ab058daa69a987d50`。分支：`codex/tbps-framework-stage25`，复用Stage 2独立开发树`/home/lzf/ldx/projects/idea-test-tbps-framework-stage2-20261016`。实现Commit：907fdcb82de1ee8abd6df101f8e8b253c129fb88；由后续文档提交登记，避免提交自指。

新增3个实现文件：smoke_irra.py、固定server JSON、test_stage25_smoke.py；新增本报告和[运行说明](STAGE25.md)。更新PROJECT_STATE、ARCHITECTURE、EXPERIMENT_REGISTRY与ADR-0003验收状态。Stage 2 Dataset/Model Adapter及全部历史源码、协议、模型和结果均未修改。

## 来源与实际样本

先读取AGENTS、PROJECT_STATE、Stage 2报告、ARCHITECTURE、EXPERIMENT_REGISTRY及ADR-0003；工作树clean/HEAD匹配后创建新分支。执行前GPU0为RTX 4090，利用率0%、已有compute进程0，可用显存24051MiB；加载前再次通过同一资源检查。

来源验证复用Stage 2实际成员工厂，校验annotation/internal split/G metadata/H资格及Gallery/训练协议。2048 Train、128内部Validation的成员摘要：
`f1a54c3caa57f2b0c9e1abf5f2a40b6282a1e1b002af383d12f94b370bfbcc66`。

仅以下2张不同Train原图和对应caption_index=0参与解码/模型前向；均official split=train、role=attack_train，未使用official Test或内部Validation做前向。

| row_id | image_rel_path | native PID |
| --- | --- | ---: |
| cuhk:9955:0 | train_query/p6841_s9083.jpg | 3193 |
| cuhk:29253:0 | CUHK03/467_4.jpg | 9460 |

图像SHA依次：
- `622ce009e5b1927f4a65171f9cdbb5a5e36e1070c84a0d66b3f09feb4f7e4fa9`
- `e854851a865d25e486d57d504abe98bb36d3a82d5c81017c07bb3a2be261028b`

原始Caption（空格不改）：

```text
He has  short  black hair  and  is  wearing  a  black shirt . He is  also  wearing white  pants  .
The man is wearing black tennis shoes with grey socks and a pair of long black shorts that go past his knees. He is wearing a white tee-shirt with black graphics on the front and a plaid grey long-sleeved shirt. He has a backpack on his back.
```

Caption UTF-8 SHA依次：
- `0c85e03c4508cdb222a9bc0480402c835cdcb2b93aada3bd09cb7acd3c73ae29`
- `7bcba2f663b6d124ac0941ff0bfa525455d34da815ad6f84cd423e168a318b5d`

官方权重：`/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth`；
SHA `9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009`。

官方Tokenizer源码SHA：
`19b06ade7532bcff961c4b90590197fcac0d466f08d0b6b801d3d19a36568d67`。
BPE文件SHA：
`924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`。
官方来源整体摘要：
`7d9e69be5f618a13b802d74bc03d73698ea3cfaac5a19b819c31ade628ab55b7`。

原始源码30份、YAML配置、CLIP bootstrap、BPE、权重及输入均来自已锁定FileRef；本轮2236个相关来源SHA/mtime_ns前后相同。preflight及attempt Manifest保存完整路径/SHA，不锁整个框架目录。

## 独立对照与数值结果

A使用SHA核验的官方build_transforms、tokenize函数及独立SimpleTokenizer实例，直接调用同一个Backend的原生encode_image/encode_text；B使用Adapter各接口。A未借用Adapter的Tokenizer/预处理/编码辅助函数。只共用冻结Backend和FP32/math/no_tf32执行设置。

预先固定max_abs_error阈值1e-6，无临时放宽；官方与Adapter均PIL RGB→384×128 Resize→ToTensor→官方mean/std；原生特征外层FP32 L2 Normalize，Query×Gallery点积。

| 检查 | 实测 |
| --- | --- |
| Token ID | [2,77] torch.int64，逐值exact，max_abs_error=0 |
| 原始RGB像素 | [2,3,384,128] torch.float32，逐值exact，max_abs_error=0 |
| Normalize后像素 | 相同shape/dtype，逐值exact，max_abs_error=0 |
| 图像检索特征 | [2,512] torch.float32，有限，逐值exact，max_abs_error=0 |
| 文本检索特征 | [2,512] torch.float32，有限，逐值exact，max_abs_error=0 |
| 图像/文本L2范数 | 两路径所有4个特征的范数均1.0 |
| 2×2 Score | torch.float32，有限，逐值exact，max_abs_error=0 |
| Token未截断长度 | 21、57，均未截断 |

两路径共同的Score矩阵（仅数值对照，不计算检索指标）：

```text
[[ 0.49033620953559875, -0.026163756847381592],
 [-0.04855334758758545,  0.5023472309112549  ]]
```

## 输入梯度与实际调用

对第一张原始Train图通过Surrogate View，计算归一化图像特征[0,0]的输入导数。输入shape=[1,3,384,128]；梯度存在、全部有限且非零：
- L2 norm：`0.3470962643623352`
- max_abs：`0.0168803408741951`

此探针只验证导数链路，不使用攻击Loss/配对奖励。IRRA始终eval、requires_grad=False，所有参数grad=None。Inference路径均inference_mode=True、grad_enabled=False；仅Surrogate输入导数路径grad_enabled=True、inference_mode=False。

| 路径 | 图像样本前向 | 文本样本前向 | 输入backward | optimizer.step |
| --- | ---: | ---: | ---: | ---: |
| 官方原生 | 2 | 2 | 0 | 0 |
| Adapter Inference | 2 | 2 | 0 | 0 |
| Adapter Surrogate | 1 | 0 | 1 | 0 |
| 合计 | **5** | **4** | **1** | **0** |

实际Backend加载尝试1次、成功1次。计数在Backend原生方法入口执行，不只是从配置推导；所有9个样本调用均正常返回，共3个图像方法调用、2个文本方法调用。没有重复加载、自动恢复或GPU重试。

367个参数Tensor前后SHA：
`b0f0c4e01e6601eb280698f8ee5a3304d9ed117e22968db9f7be17f453330d0f`。
该Backend无registered buffers（0个）；空集合摘要前后相同：
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

## 资源、异常及CPU验收

- GPU peak allocated：`930.5205078125 MiB`。
- GPU peak reserved：`1060.0 MiB`。
- 受控run_once段：`12.00397365167737 s`，包含真实构造、状态hash和结束来源核验，不包含CLI前置CPU成员预检/attempt发布。
- 前向及输入导数段：`0.9041682779788971 s`。
- 无OOM/NaN/数值超限/身份或SHA冲突；唯一GPU进程正常退出，结束后nvidia-smi compute进程为空。
- 官方构造函数输出Training Model with tasks和位置编码resize信息；它们是构造日志，不是训练；加载后显式eval/frozen，optimizer调用0。

CPU结果：
- GPU前213项框架tests通过。
- GPU后213项框架tests通过（原183+新增30）。
- 36项历史数据/Top-K/生成器/Joint基础tests通过。
- **合计249项不同CPU tests通过**，未冒称重跑全部历史prototype测试。
- 新attempt/result Manifest及完成回执只读SHA核验通过：`STAGE25_RECEIPTS_SHA_PASS`。

历史保护：复用Stage 2保护快照，对7991份历史文件独立只读复核，SHA与mtime_ns全部不变。范围含历史tracked源码、固定Checkpoint/协议/结果、官方来源、2176份G metadata及选定数据文件；没有重算Full，也没有审计全部Full Tensor缓存。
原实验树HEAD仍40c7476、Stage 1树仍61471db，均clean。Stage 2既有源码/config/tests未改。

## 不可变证据与命令

完整命令、停止语义及边界见[STAGE25.md](STAGE25.md)。输出仅在：
`/home/lzf/ldx/outputs/tbps-framework-stage25/`。

主要证据：
- `irra_equivalence_v1/attempt/{preflight,resources_before,manifest,completed}.json`
- `irra_equivalence_v1/result/{report,manifest,completed}.json`
- `gpu_smoke.log`、`preflight.log`
- `cpu_before.log`、`cpu_after.log`、`legacy_cpu.log`
- `protection_result.json`、`receipt_audit.log`

SHA-256：
- attempt Manifest：`b66bccdf4be30bae0b90cde4b6f5e67518e60eaaba2fad14964554ee6e5f740e`
- result report：`b27212f576573f13d5129fcb644effb0dcb535bf13bea259ed2e55a0d09897d9`
- result Manifest：`afd95bd2dac35902b2d32eb94705dde4e65b1d9a6d2c2f42de830a1535f62e03`
- GPU日志：`1de5585a3092e89b64af03ce5d041576f49fb9c94325c5e6c888998f775c72ac`
- 历史保护报告：`615e349e1aa13f24b79212a726c01e712168a994e4f43f9737391409edc83c15`

输出、模型、日志和缓存不提交Git。attempt一旦发布，不再自动重复GPU，失败/pending均保留；本轮完成后未再次执行GPU。

## 验收边界与停止

已验证的是本硬件、固定FP32设置、2个原始Train样本及1次输入导数的等价性和冻结状态。不能推广为所有输入/截断文本/batch/精度/硬件逐值相同；未验证Full排名、G/H Attack Plugin、统一Evaluation Engine、RaSa/APTM或其他Dataset。

没有重写历史特征或给其赋予新ModelSpace；未来复用仍需明确桥接核验。本阶段没有改变算法/预算/训练权限，不进入Stage 3或Full评测，停止等待审查。
