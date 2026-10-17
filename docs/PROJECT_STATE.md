# 项目当前状态（唯一权威入口）

状态来自Git、实际代码和不可变证据；历史文档仅为阶段快照。架构见[ARCHITECTURE](ARCHITECTURE.md)，实验来源见[EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)。

## 当前阶段

**Stage4B-2唯一一次历史IRRA Full特征CPU重放PASS，METRICS_REPLAY_VERIFIED。** 全部原3074 Gallery/6156 Query及四模式参与；这仅证明原IRRA历史特征的指标重放，不表示重新生成对抗RGB或完成新的端到端攻击。

- 基准：a4c938743aae246b541bbb303c2a2d8053566c28。
- 独立分支：codex/tbps-framework-stage4b2。
- 独立树：/home/lzf/ldx/projects/idea-test-tbps-framework-stage4b2-20261017。
- 实现Commit：待本轮提交后由后续文档提交登记。
- [接口与已执行命令](framework/STAGE4B2.md)、[完整精度结果/验收报告](framework/STAGE4B2_REPORT.md)。

## 本轮实现与实际验收

新增CPU受控入口、独立历史点积/官方rank数值仪器、RSS/耗时保护及23项合成测试；evaluation仅补强历史Score读回类型/shape/FP32/有限性，Ranking/指标/DR代码及冻结配置未改。默认只预检，独立显式参数授权真实计算，固定replay_v1目录拒绝重复执行。stage4b2_acceptance.json execution_authorized仍false；本次一次性授权已使用。

Stage4B1不可变FEATURES_VERIFIED回执及6740项完整来源通过；四family868批/3472文件内容二次独立读取exact。新评分196次、独立历史点积196次，新Ranking/官方rank各4次；四组score误差0、完整索引exact、五指标对原summary误差0，DR最大误差1.4210854715202004e-14个百分点，全部固定阈值通过。

| 组 | mAP（%） | DR_mAP（%） |
| --- | ---: | ---: |
| Clean | 66.09744262695312 | 0.0 |
| Image-only | 32.28228759765625 | 51.15955124034977 |
| Text-only | 51.471561431884766 | 22.12775655726238 |
| Joint | 23.553220748901367 | 64.36591218538784 |

统一publish_result/load_result的四组6156×3074 CPU FP32 Score、指标/来源回执读回exact。435框架+36历史基础=471 CPU tests通过。唯一正式Attempt退出码0、无failure/pending/中断；开发预检元组/JSON列表误拒绝在正式运行前修复并保留旧日志，不是正式重试。

CPU FP32/Torch2.5.1+cu124/2线程；峰值RSS1952915456 bytes（1862.4453125 MiB），发布完成98.56915779039264秒，低于2GiB/1800秒。真实IRRA/G/H模型加载/前向、GPU初始化、Optimizer更新、Backward、JVP、攻击生成均0。

## 不可变证据与保护

目录：/home/lzf/ldx/outputs/tbps-framework-stage4b2/replay_v1/。

- attempt Manifest：44202a5d829b51045bbbd0ef499154c047a04b79d0d17383341799b8033fe01e。
- evaluation Manifest：0d0f69173d313e697b62a7b9a3531cbf20366789e3bca210082f8c53dfb108b5。
- result Manifest：f11b7af27e4c87d4c4f9be4e30f908d8e0afbd68f3db547759ce191f6cb0a39a。
- report：8c13994ab049ad7f76dbe380b6647887387b9be9e8be78f16382bfacf8a96f89。
- 总时长在同一replay_v1.log最终resource_after_result_publication；result/report资源时间94.109630048275秒明确不含最终回执发布。

7991旧保护文件、3472Joint缓存文件SHA/mtime不变。Stage4B1回执及绑定来源继续有效；Stage3B原FAIL、其他历史验证/实验结果保留。原Checkpoint/Selection/算法/Tokenizer/数据及正式结果未改。

## 历史与权限边界

Stage1 61471db；Stage2 4e2a583；Stage2.5 b7aac69；Stage3A ef835a6；Stage3B 0fb90cd保持FAIL；Stage3B-R a2988e0；Stage3B-2 0d90e79固定两Train样本真实G/H输出等价通过。Stage4A 32f5582统一CPU引擎；Stage4B1 a4c9387（实现2357985）建立独立历史分型并完成FEATURES_VERIFIED，本轮在其基础上完成指标验收。

历史adversarial_pixel_sha256仍UNKNOWN；严格materialized契约未放宽，IRRA特征不能重标RaSa/APTM。没有验证新的像素生成、其他环境/模型/数据集或跨模型迁移。正式G/H/Joint/消融/AP及8个历史资产索引继续冻结。

完成提交推送后停止等待审查；不自动Stage4C、GPU Full、RaSa/APTM或新训练。唯一正式CPU Attempt许可已经使用，不得重跑、自动恢复或换目录绕过单次门禁。

本轮CPU验收回执：/home/lzf/ldx/outputs/tbps-framework-stage4b2/cpu_verified_v1/result/；
Manifest SHAd5a59929f48226dd12bf28b7382f619b6b6d19306b1c1cb7b0496e464ebca810；
report SHA907077ea4bba8af9f88b4b541f2e799628d2af5cc7a0ff5de95baa2aaea3ac78。
绑定本轮代码/测试/冻结配置、CPU日志和唯一正式重放回执。原实验发布树40c7476、Stage4B1树a4c9387及Stage4A树32f5582仍保持原HEAD且工作树干净。
