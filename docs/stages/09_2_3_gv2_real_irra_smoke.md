# G-v2 Phase 2.3：真实IRRA接入与5条GPU smoke

2026-10-15：独立适配器、真实缓存接线与GPU验证完成。177项CPU tests通过，最终5条真实smoke合计20次独立更新通过；按用户授权发布后停止等待独立审查，不进入Phase 2.4。

## 位置与新增文件

工作树：`/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/`，分支`codex/gv2-per-attribute-cache`，本阶段基准`afae393237cec06f438d3caf25717f1e4a1cf16f`。

所有新实现位于`prototypes/attribute_image_generator_v2_20261014/`：

| 文件 | 职责 |
|---|---|
| irra_image_adapter.py | 官方IRRA只读加载、像素读取、可微冻结图像前向 |
| phase23_inputs.py | 固定5条缓存、完整身份与图像/协议/来源审计 |
| smoke_phase23.py | CPU preflight、显式GPU诊断、调用/参数/梯度/预算核验 |
| test_phase23.py | 30项新增CPU tests |
| README_PHASE23.md | 接口、结果和手动命令 |

本中文报告从Windows同步纳入同一提交。G-v1、IRRA原始项目、Phase 1/2.1/2.2源码和原实验资产均未修改。新代码不import G-v1或integration，不重算文本监督、S_raw、Top-K或JVP。

## 官方流程与独立适配器

只读核对官方datasets/build.py、model/build.py、model/clip_model.py和utils/metrics.py。测试变换是PIL Resize(384×128)→ToTensor→CLIP Normalize；encode_image返回CLS [B,512]，正式检索前L2归一化。

G-v2将Resize+ToTensor后的RGB [0,1]作为G/投影空间，mean/std归一化放在可微encoder.forward内，再执行官方encode_image和L2归一化。mean=(0.48145466,0.4578275,0.40821073)，std=(0.26862954,0.26130258,0.27577711)。扰动不会加在标准化特征或标准化像素上。

官方checkpoint SHA：`9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009`。严格加载官方完整state_dict后转fp32，与Phase 1文本/JVP精度保持；不启用AMP/TF32。复用已存在且核SHA的CLIP cache，不下载。只在当前进程替换下载器为只读cache读取，不修改任何官方文件。

IRRA始终eval/参数冻结，图像梯度保留；适配器不提供文本推理接口。GPU诊断期间官方model/base_model的encode_text被失败断言保护。5条PIL像素与官方定义的显式变换一致；适配器与同一官方权重fp32直接图像前向的特征最大差均为0。

旧34054图库使用较早官方默认混合精度、batch64生成，输出容器为fp32，不能据此要求与当前fp32单图bitwise一致。五条paired clean feature对照最大绝对差0.0005377531051635742，最小cosine0.9999992847442627；其精度/批大小边界已在GPU前从冻结源码核对并报告。原图库、S/选择/JVP监督保持，不据此修改方法或冒充Gallery评测。

## 样本与状态隔离

读取原Phase 1完成的5条缓存，核对row_id、image_rel_path、caption_index、PID、caption、split、role、raw annotation、内部v4 split成员资格、源score/entry、图像SHA与paired index。缓存完成标记、schema、协议与来源SHA均严格。5条均属官方train划分；路径含test_query不代表官方Test。

每条重新创建默认width32/4-block ImageOnlyGenerator及Adam(lr=2e-4, betas=(0.5,0.999))，种子42；5条G初始SHA相同，optimizer初始state为空。validation只作独立诊断；每例结束丢弃G/optimizer，不合并为训练状态，不保存权重/checkpoint。

调用未修改的train_single_sample：默认三个lambda均1、原方案A、当前Local与上一轮Retain、相对原图8/255投影不变。逐例核验G输入detach、上一step后的参数版本、Retain前一JVP/实际图像分数，以及最后联合步精确重新输入原图。

## 最终GPU结果

| row_id | role | K | G/IRRA/backward/step各自次数 | 诊断秒数 | peak allocated MiB |
|---|---|---:|---:|---:|---:|
| cuhk:26530:0 | train | 0 | 1 | 0.213 | 1023.334 |
| cuhk:26747:0 | train | 1 | 2 | 0.424 | 1031.950 |
| cuhk:9955:0 | train | 3 | 4 | 0.641 | 1031.954 |
| cuhk:26196:0 | train | 7 | 8 | 1.137 | 1031.961 |
| cuhk:17663:0 | validation | 4 | 5 | 0.809 | 1031.956 |

最终运行合计20次G forward、20次IRRA forward、20次backward、20次optimizer.step。另有5次clean适配器forward和5次官方直接图像forward单独记录，均无反传。没有文本forward、新score backward、Top-K或JVP。

- 20轮G参数均有变化，loss/梯度/图像有限；零G梯度轮次为空，最终GPU异常为空。
- G梯度L2范围0.4595405–4.5520652，真实IRRA输入梯度L2范围0.2607865–0.7888917。
- 最大累计L∞=0.03137254901960784≤8/255，无额外预算容差。
- 最大预算端点比例58.0295%，严格裁剪比例58.0865%；精确端点与严格越界的导数约定不同，不能把所有端点比例称为零梯度比例。
- Retain正分支4次：K7的round2/3/7与validation K4的round2；没有替换Retain定义。
- IRRA参数/buffers逐例完整state_dict SHA完全不变，参数梯度None，保持eval。
- 峰值allocated1031.961 MiB、reserved1078 MiB，为本进程PyTorch统计；不包含服务器其他进程。其他进程未停止或修改。
- 计时包含单样本更新与G状态核验，不含来源审计、官方加载、clean对照与IRRA完整状态hash，不作为正式训练耗时承诺。

## CPU回归、协议修复与证据

最终177项CPU tests通过（新增30+原147），覆盖归一化/梯度/冻结、错配拒绝、固定5条门禁和JSON协议比较。完成态在禁CUDA初始化、禁模型加载、禁backward条件下CPU重入通过。

首轮5条GPU已全部通过，随后完成态复核发现JSON将tuple转list导致协议相等性误判。仅修复新runner的规范化比较并补2项CPU测试，来源SHA仍严格，没有改研究方法或旧阶段代码。为锁定最终源码，重新运行同5条；两轮各20次更新，本阶段共40次诊断更新，未增加样本。首轮与修复后证据均保留，两轮不构成bitwise CUDA重复性保证。

50份基线保护文件和最终4527项来源SHA保持，包括G-v1与原Phase 1缓存、已发布Phase 1/2.1/2.2文件及官方源码/权重。GPU进程已退出。

原始证据只在服务器G-v2目录`outputs/phase23/`：

- `smoke_verified/`：最终protocol、5条逐例JSON、summary与completed；method_audit_passed=true。
- `smoke/`、gpu_smoke_initial.log、cpu_tests_initial175.log：首轮与修复前记录。
- gpu_smoke.log、cpu_tests.log、preflight.log、verification.json、protected_sources.json：最终日志与审计。

最终summary SHA：`6267a1e69066644918c937bebcdacf469b476523d55ce44502fba1eb1062ea53`。

最终CPU日志SHA：`def967a9f8f375f7560e4f69ace8762b30d12dfe3da1f0c9afcee0a54ca5f22d`。

## 手动复核与交付边界

在服务器G-v2目录：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_phase23 test_phase22 test_phase21 test_conditions -v
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python smoke_phase23.py
PYTHONDONTWRITEBYTECODE=1 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python smoke_phase23.py --allow-smoke --output outputs/phase23/smoke_verified
```

最后一条对当前完成输出仅做CPU核验；源码/来源变化仍拒绝。旧首轮输出不覆盖。Git仅纳入5个新增文件与本报告；数据、权重、条件缓存、outputs和日志不提交。只克隆GitHub不含真实运行资产。

本结果只证明真实IRRA梯度与单样本机制接线，不证明攻击性能、属性监督增益或完整训练稳定性。未准备2176条缓存、未实现epoch/batch调度、未运行完整训练或Gallery检索；发布后停止，等待独立审查，不进入Phase 2.4。
