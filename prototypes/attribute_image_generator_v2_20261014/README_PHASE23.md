# G-v2 Phase 2.3：真实IRRA接线与固定5条GPU smoke

2026-10-15；基准afae393237cec06f438d3caf25717f1e4a1cf16f。
仅真实接线与5条独立诊断，不运行正式训练、全量缓存、Gallery检索或Phase 2.4。

## 新增文件

| 文件 | 职责 |
|---|---|
| irra_image_adapter.py | 独立官方IRRA只读加载、RGB像素与可微图像前向 |
| phase23_inputs.py | 5条原缓存及图像/caption/PID/role/协议/来源SHA核验 |
| smoke_phase23.py | 默认CPU preflight、显式GPU smoke与计数/状态审计 |
| test_phase23.py | 30项新增CPU tests |
| README_PHASE23.md | 本说明 |

原G-v1、IRRA源码、Phase 1/2.1/2.2代码和资产均保持。
不import G-v1或integration；只用G-v2缓存IO/schema、模型、loss和train_single_sample。

## 编码与监督空间

官方datasets/build.py测试变换为PIL Resize(384x128)、ToTensor、CLIP Normalize。
官方model/build.py的encode_image取CLS [B,512]；utils/metrics.py检索前L2归一化。
G-v2先执行Resize+ToTensor，扰动/投影始终在[0,1]空间；
可微encoder前向执行(pixels-mean)/std、官方encode_image和L2归一化。

mean=(0.48145466,0.4578275,0.40821073)，std=(0.26862954,0.26130258,0.27577711)。
官方checkpoint SHA为9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。
严格加载官方模型/权重后转fp32，与Phase 1文本/JVP精度保持；无AMP/TF32。
只读使用已有CLIP cache，下载器在当前进程中替换为SHA核验后的只读读取，
不修改官方文件，不允许下载。缓存缺失则失败。
IRRA始终eval/参数冻结，像素输入梯度保留；新适配器没有文本推理入口。
运行时官方encode_text和base_model.encode_text被失败断言保护。

5条PIL像素及独立官方fp32直接前向均exact，特征误差0。
历史34054图库来自较早官方混合精度/batch64，输出虽为fp32容器，不能要求与当前fp32单图bitwise一致。
仅对五个paired feature报告差异：最大5.3775311e-4，最小cosine .9999992847；
这是既有精度/批大小边界，不是新的Gallery评测或监督变更，原缓存保持。

## 固定样本、独立状态与核验

读取原Phase 1 outputs/smoke，原协议锁及完成标记核验后只取以下5条：

| row_id | role | K | 更新数 |
|---|---|---:|---:|
| cuhk:26530:0 | train | 0 | 1 |
| cuhk:26747:0 | train | 1 | 2 |
| cuhk:9955:0 | train | 3 | 4 |
| cuhk:26196:0 | train | 7 | 8 |
| cuhk:17663:0 | validation | 4 | 5 |

同时核对完整identity、原annotation首caption/PID/split、内部v4 PID split成员资格、
源score/entry、图像SHA、paired index及原图库路径。错误配对拒绝，无评分/重选/JVP。
目录名含test_query不能代替官方split判断；5条均属于官方train划分。

每例重新初始化width32/4-block ImageOnlyGenerator和Adam(lr2e-4, betas .5/.999)，种子42。
5例G初始SHA完全相同，optimizer初始state为空；validation仅独立诊断，全部状态结束后丢弃。
不保存G、optimizer、训练权重或checkpoint。

复用原方案A和三个默认系数1：
属性轮G输入为上轮detach图像，下一轮用上一step后的参数；
Retain参考逐值核验；联合步明确从原图重新开始。
每例G/IRRA forward、backward和optimizer.step严格K+1。
最终一轮5条合计20次；额外5次clean适配器与5次官方直接前向单独计数。

## 最终验证结果

177项CPU tests通过（新增30+原147），覆盖像素/归一化/输入梯度/冻结/配对/默认CPU门禁。
真实20轮均更新G，所有loss/梯度/图像有限，零G梯度轮次为空，无最终GPU异常。
G梯度L2范围.4595405–4.5520652；IRRA输入梯度L2 .2607865–.7888917。
最大L∞=0.03137254901960784，不超过8/255。
预算端点最高58.0295%，严格裁剪最高58.0865%；不可把端点比例都等同零梯度比例。
Retain正分支触发4次，原图联合重启、参数顺序、stop-gradient引用全部通过。

| K/role | 单例训练诊断秒数 | peak allocated MiB |
|---|---:|---:|
| 0/train | .213 | 1023.334 |
| 1/train | .424 | 1031.950 |
| 3/train | .641 | 1031.954 |
| 7/train | 1.137 | 1031.961 |
| 4/validation | .809 | 1031.956 |

峰值reserved1078 MiB。计时包含更新与G状态核验，不含加载/来源审计/clean对照；
不据此推算正式训练速度。显存是本进程PyTorch统计，不含服务器其他进程。
IRRA state_dict参数/buffers逐例SHA完全不变，参数梯度None；50份基线文件和4527项来源SHA保持。

首轮5条GPU已通过，随后CPU完成态复核暴露JSON tuple/list比较问题。
仅修复协议规范化比较并新增2个CPU测试，来源SHA仍严格；
为锁定最终源码重新运行同5条，保留首轮结果。两轮各20次，共40次诊断更新，未扩大样本。
两轮不作为bitwise CUDA重复性保证。最终完成态禁CUDA/模型加载/backward复核通过。

## 输出与手动命令

所有原始记录仅在本目录outputs/phase23/，不进入Git：
smoke_verified/{protocol,summary,completed}.json及5条逐例JSON为最终证据；
smoke/保留首轮，gpu_smoke_initial.log/cpu_tests_initial175.log保留早期证据；
gpu_smoke.log/cpu_tests.log/preflight.log/verification.json/protected_sources.json分别记录最终验证。

~~~bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_phase23 test_phase22 test_phase21 test_conditions -v
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python smoke_phase23.py
PYTHONDONTWRITEBYTECODE=1 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python smoke_phase23.py --allow-smoke --output outputs/phase23/smoke_verified
~~~

最后一条在当前完成输出上只CPU核验；源码/来源变化仍拒绝，首轮旧输出不会被覆盖。
仅克隆GitHub不包含官方权重、dataset、Phase 1缓存与成功smoke资产。
当前结果只证明真实梯度接线，不证明攻击效果或属性模块增益。发布后停止等待独立审查。
