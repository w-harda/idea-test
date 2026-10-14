# Step7.2：Top-K Attribute-Guided IRRA-TTA image prototype

冻结 AttributeInstance / S_raw / gamma3 clean plan，不训练 generator，不修改正式 Stage03/Top-K 或官方 TTA。

## Guidance 定义

原 IRRA block8 取 H；e 使用保存实例的 projected_context，Q 使用原 v5.5 semantic basis，w 使用 Step5.1 保存的 frozen clean localization。先检查 e 属于 Q 的 span，设 B=w·normalize(e)^T。

通过原 caption 的 frozen IRRA forward 在 block8 注入标量 eta·B，保留 blocks9–12、ln_final、text_projection、EOT 与 L2 normalization，求 v=JVP(f_T,H,B)。v 是最终 retrieval text feature 的局部变化，位于最终检索空间，可与最终 image feature 比较；没有直接用 block8 e 点积图像特征，没有训练跨模态 projector。

属性攻击最大化 L=-image_feature·v。它等于原负 paired cosine 对选中 contextual 属性方向的一阶方向导数。不是 literal attribute removal，也不声称其语义已完全解耦。caption 只定义原 forward 的上下文/线性化点；TTA 的属性模式只收到 attribute_unit:instance_id 标签及预计算的 v，不再编码 canonical phrase/raw word、不加入完整 caption embedding loss。去掉 guidance 的诊断明确使用单 full-caption 原 baseline。

原 S_raw 的 paired soft rank/tau=.02 保持原样；此攻击 loss 复用 TTA 的负配对相似度结构，不把它称为 paired soft-rank attack。

## 复用与限制

attribute_tta.py 只适配存储计划、原图 JVP 和官方 text supervision bridge；官方 Attack.img_attack 原样加载、SHA 校验。随机初始化、IG/transformed IG、adaptive transformations、0.9 momentum、2/255 sign step、8/255 原图累计投影、clamp 均复用。每轮noise initialization仅先clamp像素，首个内部梯度点可能超出累计epsilon；step/round完成时再投影，不能声称所有内部evaluation point均已投影。

新 payload 与旧 slot/int scheduler 不直接兼容，使用独立 image-only frozen-plan 适配循环，沿用旧存储顺序和 current-image 累计语义；没有替换旧 Stage05/06 调度器。没有 text attack 或 generator。重复实例仍保留固定 image rounds，不重排/重选。

CPU tests：30 passed。GPU smoke 固定 cuhk:34:0 / 35:1 / 0:1，自然 k1/2/3；每轮1步、每尺度1变换、scales=None。正式默认仍是10步、每尺度6变换和(.5,.75,1.25,1.5)。另在同一34:0上用完整默认scale/transform配置执行1步，不增加caption。

最终 smoke：outputs/smoke_final/；对应 log：outputs/smoke_final.log。前两个 smoke / smoke_debug 记录保留；此前停止原因为 momentum 逐值一致断言过严，最终特征和图像原已逐值相同，改为记录 fp32 momentum 数值容差，没有改公式、样本或机制。

原 project_image 的 fp32 实测 L∞=.03137257695，比数学8/255多2.79e-8；保持原代码的1e-6容差，不能声称实数意义上零误差。JVP 使用 math SDPA、关闭MHA fastpath以支持double backward/zero exact；全部权重与reader/basis不变。

## 测试（服务器执行）

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python -m pytest -q -p no:cacheprovider prototypes/attribute_tta_step72_20261003/test_attribute_tta.py tests/test_attack_scheduler.py tests/test_joint_baseline.py
```

## 用户手动小规模诊断

固定原25句，不是full validation、共享全图库攻击或正式TBPS评测。只输出image机制/梯度/投影/metadata诊断，不保存adversarial dataset、不给attack success结论。没有full模式、训练或resume。

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python prototypes/attribute_tta_step72_20261003/run.py --mode diagnostic --allow-diagnostic > prototypes/attribute_tta_step72_20261003/outputs/diagnostic.log 2>&1
```

后台等价命令：

```bash
tmux new-session -d -s tbps-step72 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python prototypes/attribute_tta_step72_20261003/run.py --mode diagnostic --allow-diagnostic > prototypes/attribute_tta_step72_20261003/outputs/diagnostic.log 2>&1'
```

查看：`tmux attach -t tbps-step72` 或 `tail -f /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step72_20261003/outputs/diagnostic.log`。

输出目录：/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step72_20261003/outputs/diagnostic/。protocol.json锁定配置/source/checkpoint/SHA，results.json保存每句selected metadata、JVP核验、clean-image梯度cosine与逐round累计预算，audit.json和completed.json确认完成和冻结。目录存在时拒绝覆盖；无resume，如需重跑应显式传入另一个ldx下--output新目录。

同一4090默认1step/6trans/5scales约2.30秒；25句共75selected rounds ×10steps，估计20–60分钟，实际以首句CASE_DONE外推。首先看CASE_DONE的k和linf、mapping.central_fd_relative_error、rounds.caption_loss_terms=0 / text_encoder_calls_during_attack=0，以及末尾COMPLETE的source_SHA_unchanged / frozen状态。出现Traceback或completed.json缺失不能视为完成。
