# Stage 4B-2验收报告：历史IRRA Full特征CPU指标重放

## 结论

**PASS / METRICS_REPLAY_VERIFIED**。唯一一次正式Attempt退出码0，日志出现METRICS_REPLAY_VERIFIED，attempt/evaluation/result三套不可变Manifest及完成回执只读复核有效。没有failure或pending，没有自动重试/恢复。

基准a4c938743aae246b541bbb303c2a2d8053566c28；独立分支codex/tbps-framework-stage4b2；独立树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4b2-20261017。实现Commit在PROJECT_STATE由后续文档提交登记。

状态仅表示**原IRRA历史特征的指标重放成功**，不是重新生成对抗RGB、重新运行端到端攻击或跨模型迁移成功。原adversarial_pixel_sha256保持UNKNOWN。

## 实现与固定来源

新增replay_cpu受控入口、replay_control资源/历史保护、replay_oracle独立数值仪器及23项CPU测试；evaluation仅补齐历史Score读回Tensor类型、Manifest/报告shape、CPU FP32、有限性检查。原Ranking/mAP/R@K/mINP/DR实现没有修改或复制。

默认CLI只CPU预检。stage4b2_acceptance.json字节未修改，execution_authorized仍false；本次授权来自用户指令及--authorize-cpu-full-replay。唯一目录replay_v1一旦被Claim就拒绝第二次执行，不提供改目录重试或恢复参数。

固定原IRRA/CUHK-PEDES官方Test3074 Gallery/6156 Query、G-v2 E10/H-v1 E7，原四family及全部成员参与，没有zip配对、正确匹配筛选或camera过滤。PID只进入最终Ranking身份层。Frozen Checkpoint：

- G：f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e
- H：487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df
- IRRA：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009

Stage4B1 FEATURES_VERIFIED原Manifest SHA5cd8dcf0e8ccbc61b58adc7dae37e26e51f361369e15f6fefa77c9465bfc1370、report SHA7d8dd8364df3537471658a0d3ce3c30f5e4609f700bb75a68e2f68e0fdc979bf再次核验。全套6740项FileRef绑定原来源、Selection、Checkpoint、原完成回执、3472特征文件、原验收producer、新代码及冻结配置。

## 实际计算结果（完整精度，单位%）

下表来自本次统一Engine实际计算并由publish_result/load_result发布/读回的结果，非直接填入历史指标。

| Group | mAP | R@1 | R@5 | R@10 | mINP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 66.09744262695312 | 73.14814758300781 | 89.87979125976562 | 93.71344757080078 | 50.27766799926758 |
| Image-only | 32.28228759765625 | 37.443145751953125 | 60.769981384277344 | 70.07797241210938 | 17.216564178466797 |
| Text-only | 51.471561431884766 | 56.57894515991211 | 75.97466278076172 | 82.5211181640625 | 36.36254119873047 |
| Joint | 23.553220748901367 | 26.526966094970703 | 46.832359313964844 | 57.19622802734375 | 11.816774368286133 |

相对Clean的DR，单位%；沿用原Stage4A drops，定义(Clean-Adv)/Clean×100，Clean=0时null：

| Group | mAP | R@1 | R@5 | R@10 | mINP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Image-only | 51.15955124034977 | 48.81190161451046 | 32.38749163463977 | 25.221006986040862 | 65.75703515382297 |
| Text-only | 22.12775655726238 | 22.651568044553873 | 15.470806378283712 | 11.943141242650842 | 27.67655572398429 |
| Joint | 64.36591218538784 | 63.735286577328345 | 47.894450290151944 | 38.9668937490195 | 76.49697203844404 |

Joint_minus_ImageOnly_mAP_pp=-8.729066848754883。这次对已有结果做可追溯重放，不新增攻击效果实验。

## 独立数值对照

- 868个批次先由JointFullLoader校验/clone，再走第二条原文件torch.load/clone路径逐批torch.equal；四family内容exact，3472文件SHA/回执/身份通过。
- 新IRRACachedScorer经EvaluationEngine.run(query_chunk=128)计算；对照路径独立执行原eval_metrics.metrics的128分块点积表达式，未调用Scorer辅助函数。
- 原评分表达式文件SHA5ee30d981edf62a8b8abf57972c9c91a9545b56ef59881d37be709f4a090f637；官方utils/metrics.py SHA94910252a9e42f0820d49c11203b31d3b67951f58624f66abff4bf662500d5fb。官方rank从原文件独立加载，不使用新排名函数冒充官方。
- 仪器包裹Engine原ranking调用，仅收集对照；实际新Ranking仍为未修改的Stage4A实现。
- 四组score max_abs_error均0.0；四组新/官方Ranking索引torch.equal、different_indices=0，双方索引摘要相同。
- 四组五指标对独立官方结果及原Joint Full对应summary的绝对误差全部0.0。
- DR对原summary最大绝对误差1.4210854715202004e-14个百分点，低于冻结1e-5。微差来自原历史drops与Stage4A drops在Python浮点乘除规约顺序上的差别；未修改公式、实现或阈值。
- 新评分196次（4组×49 Query块），独立历史点积196次；新Ranking4次，官方rank4次。每块面对完整Gallery。
- 四组Score经过统一publish_result/load_result，shape6156×3074、CPU FP32、有限性/SHA及逐值读回exact。

## 资源、安全及CPU回归

CPU FP32、Torch2.5.1+cu124、2线程；无GPU初始化、IRRA/G/H模型恢复或前向、Optimizer更新、Backward、JVP、新条件准备或攻击生成，forbidden_attempts={}。

峰值RSS1952915456 bytes（1862.4453125 MiB / 1.8187942504882812 GiB），低于2GiB。验收发布前94.109630048275秒；日志记录发布完成总用时98.56915779039264秒，低于1800秒。result/report记录前一时间并注明范围；总时间来自同一日志最终resource_after_result_publication。未发生资源中断或异常。逐模式释放官方参考Score/Ranking，旧临时索引不跨模式保留；最终新四组Score用于统一发布/读回。

**435框架tests（412原有+23新增）+36历史基础=471项通过。** 覆盖独立官方小型数值对照、错误Score/索引诊断、读回shape/dtype/NaN/type、紧凑Storage、默认预检/显式授权/单次目录、资源边界和历史保护。

开发CPU预检曾因dataclass元组与JSON列表直接比较误拒绝；保留preflight.log，修成规范化JSON字节exact比较并增加回归测试后preflight_final.log通过。该开发失败未创建正式Attempt、未执行真实评分，不涉及正式运行重试。

## 保护与不可变证据

7991个旧保护文件及3472个Joint特征文件SHA/mtime前后不变；原Stage4B1 FEATURES_VERIFIED证据及其全部来源继续有效。原源码/Checkpoint/Selection/缓存/正式指标/完成回执未修改。

证据基目录：/home/lzf/ldx/outputs/tbps-framework-stage4b2/。

| 证据 | SHA-256 |
| --- | --- |
| replay_v1/attempt/manifest.json | 44202a5d829b51045bbbd0ef499154c047a04b79d0d17383341799b8033fe01e |
| replay_v1/evaluation/manifest.json | 0d0f69173d313e697b62a7b9a3531cbf20366789e3bca210082f8c53dfb108b5 |
| replay_v1/result/manifest.json | f11b7af27e4c87d4c4f9be4e30f908d8e0afbd68f3db547759ce191f6cb0a39a |
| replay_v1/result/report.json | 8c13994ab049ad7f76dbe380b6647887387b9be9e8be78f16382bfacf8a96f89 |

日志replay_v1.log；CPU预检preflight_final.log；CPU回归cpu_framework_final.log/cpu_legacy.log。完整逐组Score、Ranking摘要、五指标/DR误差、调用次数和来源SHA见result/report.json及各Manifest。源码与文档提交，任何输出/Score/缓存/权重/日志不入Git。

## 未验证与停止边界

没有重新验证G/H全量生成过程或对抗RGB像素身份，也未验证不同Torch、硬件、原生Tokenizer、Dataset、Victim或RaSa/APTM。历史模型权重只SHA，不反序列化。模型冻结在本轮体现为无模型实例/前向/参数更新，不是重新加载模型做state对照。

本阶段完成后停止等待审查，不自动Stage4C、GPU Full、RaSa/APTM或训练。唯一正式Attempt许可已使用；不能重跑命令或换目录复用许可。

本轮CPU验收回执：/home/lzf/ldx/outputs/tbps-framework-stage4b2/cpu_verified_v1/result/；
Manifest SHAd5a59929f48226dd12bf28b7382f619b6b6d19306b1c1cb7b0496e464ebca810；
report SHA907077ea4bba8af9f88b4b541f2e799628d2af5cc7a0ff5de95baa2aaea3ac78。
绑定本轮代码/测试/冻结配置、CPU日志和唯一正式重放回执。原实验发布树40c7476、Stage4B1树a4c9387及Stage4A树32f5582仍保持原HEAD且工作树干净。

实现Commit：26f70a480a0eef0c943fb9204afc7cfaeb221986。本条由后续文档提交登记；最终发布HEAD见Git。
