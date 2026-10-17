# Stage4C-3：IRRA-only真实物化输入等价Smoke PASS

## 结论与范围

唯一正式Attempt退出码0，状态 **IRRA_MATERIALIZED_SMOKE_VERIFIED**。仅覆盖固定CUHK-PEDES官方Test两张Gallery、两条Query，在本机RTX4090及FP32/math/no_tf32环境中的物化RGB/Unicode→IRRA特征及四组2×2 Score等价。

**不是Full评测或跨模型攻击成功。** 未计算mAP/Ranking，没有全量物化、训练、G/H加载或前向。原Stage4C-2整体FAIL及原Plan/生成证明保持不变。

基准9f1f943ed8486a8c2889ab272764e5707a431552；分支codex/tbps-framework-stage4c3；工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4c3-20261017。

## 1. 复用输入与来源

只读来源：/home/lzf/ldx/outputs/tbps-framework-stage4c2/materialized_smoke_v1/。

Gallery按原index0/15：cuhk:37132:0、cuhk:37147:0。Query按原index0/31：cuhk:37132:0（2次历史Unicode编辑）、cuhk:37147:1（固定NOOP/K0）。未按效果替换样本。

沿用Stage4C-2R RetainedInputs/FrozenPlanReference、原MaterializedStore.load和原动作校验；全部原/Adv RGB、effective_delta、UTF-8、编辑动作/Tokenizer审计、生成Claim/证明、权重/训练协议通过。

原Plan ID **7c0c00daf022bfcba94d6e3583c6de26c89b58c0e9c6696449cdbadb1ab7072d**，新Attempt只发布只读引用与IRRA表示，不重签、复制或重新发布旧攻击单元。原整体FAIL Manifest c1922bdf32935623dc9e85b1aeffe8a817cef2ab42b3ba767f62eda90e45155b保持。

| 输入 | 原始像素内容SHA | 对抗像素内容SHA |
| --- | --- | --- |
| cuhk:37132:0 | 90218e2189d60abd931eaaf0e5b8a49c24777424e37ee7d3901b74af63129045 | 54c594950652698fb62925eee8e50bf87c3505f55b66850bdc332b2572cf4be6 |
| cuhk:37147:0 | 64e176c41cf3c61ed672b7a488ec7390183900cae3238e6b7b113a029382d713 | bad51abb2f0e46affbb7ec745f9f436b1f63a9fcb61a976c9c439a6caffcaabe |

两图均FP32 [1,3,384,128]、像素[0,1]，L∞0.0313725471496582≤8/255。编码观察器实际逐值核对IRRA收到的Tensor与保存输入exact，分别记录原始及标准化后Tensor SHA；没有重新PIL转换或Resize、没有调用G。

H编辑保留原位置3的a U+0061→а U+0430及位置44的o U+006F→о U+043E；NOOP的UTF-8原/Adv SHA同为ae57e322322decc31f9491e17fa36f27837fb5b91b6156d1fdb9bcda9747f400。完整action仍在旧单位及新只读引用中，没有新H决策。

冻结来源：

- G10：f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e，仅SHA核验。
- H7：487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df，仅SHA核验。
- 官方IRRA：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009，真实恢复一次。
- 原官方源码、CLIP bootstrap、配置、BPE、Tokenizer及输入文件均SHA绑定；未优化或替换IRRA加载器。

## 2. CPU门禁、预处理与Tokenizer

严格CPU门禁先通过75项G/H来源、33个旧实际执行文件及5个官方IRRA执行文件；旧stage-g/text PASS和整体FAIL、完整四单位以及历史Full固定对应行验证通过。

复用CPUStartupGate真实顺序：

`cpu_read_started → cpu_sources_verified → cpu_isolation_exited → cuda_initialization_call → irra_model_load_entry`

所有CPU隔离上下文退出，Torch函数、导入环境及legacy锁恢复，CUDA未初始化，才允许峰值统计与恢复模型。IRRA原生Adapter使用已保存FP32 RGB，之后只执行规定mean/std与归一化；原始/攻击文本经同一官方77-token Byte-BPE。

四条原/Adv文本Tokenizer IDs与历史before/after记录exact。初版CPU预检将历史字段误读为original/adversarial而拒绝，已按真实before/after schema修正并补两个测试；发生于GPU Attempt前，无模型加载，旧证据未修改。

推理使用fp32_runtime/math/no_tf32，autocast禁用，eval/inference_mode、requires_grad=false；不以测试标签或目标评分改变攻击产物。

## 3. 四组512维特征对照

每组shape=[2,512]、torch.float32、有限、L2归一化合法。参考独立加载Stage4B-2核验过的旧Joint Full对应行，仅读取涉及的小批次，不聚合Full特征。

| Family | 最大绝对误差 | bitwise exact | 两行L2范数 |
| --- | ---: | --- | --- |
| clean_gallery | 0.0 | true | 1.0 / 0.9999999403953552 |
| image_gallery | 0.0 | true | 1.0 / 1.0000001192092896 |
| clean_text | 1.9371509552001953e-07 | false | 0.9999999403953552 / 0.9999999403953552 |
| adv_text | 1.9371509552001953e-07 | false | 1.0 / 0.9999999403953552 |

全部≤固定1e-6。文本不是bitwise exact，未扩大结论或改变精度/Tokenizer/阈值来匹配。

## 4. 四组2×2原生Score

分别以新物化输入真实编码结果与独立历史对应特征计算归一化点积。行序固定Query 37132:0/37147:1，列序固定Gallery 37132:0/37147:0。

| 模式 | 最大绝对误差 |
| --- | ---: |
| Clean | 7.450580596923828e-08 |
| Image-only | 1.0151416063308716e-07 |
| Text-only | 8.940696716308594e-08 |
| Joint | 1.0151416063308716e-07 |

全部≤1e-6；四组均非bitwise exact，不能写成Score误差0。完整精度：

```text
Clean 新：
[[0.5500701665878296, -0.11703988909721375],
 [-0.16345402598381042, 0.3023773729801178]]
Clean 历史：
[[0.5500701665878296, -0.11703986674547195],
 [-0.1634541004896164, 0.3023773729801178]]

Image-only 新：
[[0.46966469287872314, 0.28123706579208374],
 [-0.23555117845535278, -0.00021468382328748703]]
Image-only 历史：
[[0.46966463327407837, 0.2812371253967285],
 [-0.23555125296115875, -0.00021478533744812012]]

Text-only 新：
[[0.5017630457878113, -0.14967656135559082],
 [-0.16345402598381042, 0.3023773729801178]]
Text-only 历史：
[[0.501763105392456, -0.14967647194862366],
 [-0.1634541004896164, 0.3023773729801178]]

Joint 新：
[[0.42762959003448486, 0.2707691192626953],
 [-0.23555117845535278, -0.00021468382328748703]]
Joint 历史：
[[0.42762964963912964, 0.2707691788673401],
 [-0.23555125296115875, -0.00021478533744812012]]
```

未进行2样本mAP或正式检索评测，不以小矩阵指标解释攻击有效性。

## 5. 实际调用、冻结与资源

- G/H恢复0/0、图像/文本策略前向0/0；原历史生成次数单独保留，不算本次新调用。
- IRRA加载尝试1、恢复1；图像4/文本4，8次batch1编码，started=completed。
- Backward/Optimizer/JVP=0；禁止操作监视记录forbidden_attempts={}。
- IRRA 367参数Tensor前/后SHA均b0f0c4e01e6601eb280698f8ee5a3304d9ed117e22968db9f7be17f453330d0f；buffer数0、空摘要一致；eval/参数冻结/grad=None保持。
- 单次Attempt，没有OOM、来源/预算/数值/冻结异常、中断或重试。GPU进程已退出。

本轮用户明确取消旧2GiB进程RSS限制；旧配置不改，新配置rss_hard_limit_bytes=null。使用实际系统RAM预检≥8GiB，并以整机MemAvailable低于2GiB作为安全停止条件，**后者不是进程RSS上限**；300秒时限保留。

| 资源 | 实测 |
| --- | ---: |
| 系统RAM总量bytes | 134714720256 |
| 运行前MemAvailable bytes | 123902926848 |
| 监视期间最小MemAvailable bytes | 122121281536 |
| 进程峰值RSS bytes | 2524065792 |
| 进程峰值RSS MiB / GiB | 2407.13671875 / 2.350719451904297 |
| GPU初始free/used MiB | 24051 / 40 |
| GPU peak allocated/reserved MiB | 922.380859375 / 1002.0 |
| 执行段elapsed_seconds | 24.73483756557107 |
| 资源监视段elapsed_seconds | 26.17609791457653 |

执行/监视时间均包含CPU准备、来源验证与IRRA工作，非纯神经前向时长；不含此前严格CPU预检和最终回执发布，整条命令墙钟未单独记录。实测RSS超过旧2GiB，按本轮新授权不作为失败；未开展额外Stage4C-2M诊断或更换加载器。

## 6. 文件、CPU测试与保护

新增irra_only.py、stage4c3_irra_only.server.json、test_stage4c3_irra_only.py及本报告。新树smoke.py::stage_irra仅增加显式store_factory/result_store/on_cpu_ready注入，复用原CPU准备、IRRA编码、归一化、特征/Score比较循环；默认调用保持兼容，不复制编码算法。旧工作树所有文件不变。

**532框架+36历史基础=568 CPU tests通过，14新增，38.11秒。** 覆盖默认CPU无副作用、唯一Attempt、来源拒绝、G/H调用禁止、真实可用RAM监视、超过2GiB不失败、固定计数与阈值、Tokenizer旧schema/ID差异等；原完整启动流程Mock回归继续通过。

后置CPU审计不初始化CUDA、不加载模型、不反序列化特征或重排名，只验证新完成回执、特征artifact SHA、计数/冻结/误差字段及来源保护：

- 7991项历史保护文件SHA/mtime exact。
- 3472个Joint Full缓存文件SHA/mtime exact。
- 原Stage4C-2 Attempt 70文件SHA/mtime exact，整体仍FAIL。
- Stage4C-2R/C2/C1/B2/Stage3B2 PASS/Stage3B FAIL/原发布树HEAD及clean保持。
- 新输出回执齐全，无pending；旧Plan、Claim、生成证明及Checkpoint不覆盖。

## 7. 证据路径与SHA

根：/home/lzf/ldx/outputs/tbps-framework-stage4c3/irra_only_v1/。

- attempt/manifest.json：0787cce4deb3038cedc1cc55dbc448dfcb321d529222d1520714dfcea6c27035。
- result/manifest.json：e6aa5f89e9fce861bfd900c5160629b6f54bc196a858b43ee341116539062f70。
- result/report.json：c85ffc5a315dcfe233a2c44459400f2489059bedcc2ead6ca6aaf728436de98f。
- features-clean_gallery/manifest.json：fd668e91a7252bcb6539bd10ebe7ca226a9cff23510d757e33d375ecae03f76d。
- features-image_gallery/manifest.json：2434942192c8a9c3ce41380e850a28e1a606871a957ee80bdf147c8dad84705a。
- features-clean_text/manifest.json：13f1b79a766d43d056f3dcdbb5cd02327d9f30bfd2030950101125fb11d089ce。
- features-adv_text/manifest.json：f6d7f09cf8fe7f36dd9c91be8eb9959d733ac5924fbad073b0d69c0a54b6916c。
- CPU完成审计：/home/lzf/ldx/outputs/tbps-framework-stage4c3/cpu_verified_v1/result/manifest.json，SHA40e4308c251dfcb42ad9f29b4d631c5e93dd598f4c7e4521ae601859f56963c4。
- 日志/home/lzf/ldx/tmp/stage4c3_gpu_attempt_v1.log、stage4c3_final_cpu_tests.log、stage4c3_cpu_preflight_pass.log。

结果Manifest绑定本次Attempt及四个新feature Manifest；每份特征绑定原历史批次SHA和明确的IRRA ModelSpace，不能用于RaSa/APTM或冒充新G/H攻击产物。

## 8. 命令与停止边界

CPU预检可复核：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4c3-20261017
CUDA_VISIBLE_DEVICES="" TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.materialization.irra_only --config configs/framework/stage4c3_irra_only.server.json
```

本轮唯一GPU命令使用同一入口，CUDA_VISIBLE_DEVICES=0并增加--allow-gpu-smoke；已执行成功。固定irra_only_v1已用，入口拒绝第二次运行，没有resume/换输出/扩规模参数；不应再次运行。

本固定两图两文的物化IRRA端到端状态从PENDING更新为VERIFIED；不推广到Full、其他样本、硬件、模型或Tokenizer。没有验证全量3074张物化、6156条全量新编码、Full排名或跨模型；没有新的G/H训练或Checkpoint选优。

提交推送后停止等待审查，不自动进入全量物化、Stage5、RaSa/APTM或新训练。

实现/CPU测试/唯一GPU PASS Commit：bef5f407be75e7aa0dc6550e74185012deb3b06a。本条由后续文档提交登记，最终远端HEAD以Git核验。
