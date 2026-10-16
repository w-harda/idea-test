# 项目当前状态（唯一权威入口）

状态来自Git、代码和不可变证据；历史文档是阶段快照。架构见[ARCHITECTURE](ARCHITECTURE.md)，历史实验见[EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)。

## 当前阶段

Stage 3A已审查；**Stage 3B唯一一次真实尝试FAIL：旧运行来源闭包缺少condition_schema.py，尚未执行G/H输出对照。** G/H等价保持PENDING/legacy_unverified，未进入Stage 4。

- 基准：ef835a60a85c30910ef4a3442230882f72a05c44。
- 独立分支：codex/tbps-framework-stage3b。
- 独立树：/home/lzf/ldx/projects/idea-test-tbps-framework-stage3b-20261016。
- 实现Commit：待提交后由后续文档提交登记。
- [入口/CPU命令/禁止重试](framework/STAGE3B.md)；[FAIL证据与最小下一步](framework/STAGE3B_REPORT.md)。

## 本轮实际执行

- GPU前按原2048 Train资格顺序锁定首条eligible cuhk:9955:0和首条固定NOOP cuhk:26530:0，两者实际official train/attack_train。
- 原CPU预检通过其既有范围，但漏查G像素读取器传递导入。不能把该PASS当作完整加载闭包许可。
- 唯一真实尝试恢复G10一次并转GPU；随后irra_image_adapter→cache_io→condition_schema在严格来源门禁失败，H尚未恢复。
- 实际权重恢复G=1/H=0；实际G/H/IRRA前向全部0，backward/Optimizer/JVP/新准备0。没有输出误差、H动作、Joint真实组合或内存状态前后摘要，不得填0或宣称通过。
- 失败段1.0158903375267982s，GPU peak allocated5.2587890625MiB/reserved6MiB；进程已退出，没有重试/恢复/换权重/放宽阈值或kill其他进程。
- 执行脚本及其绑定Plugin源码保持原SHA；没有补改Stage 3A来源清单。失败后仅新增严格CPU闭包检查及回归，当前明确拒绝缺项，无GPU选项。

## 验证与保护

- **298框架CPU tests通过**：原285+新增13；另36历史基础通过，合计**334项不同CPU测试**，不是全部prototype tests重跑。
- 新测试包含当前未锁定condition_schema必须拒绝的回归；通过这些测试不代表真实等价通过。
- 7991份历史文件SHA/mtime_ns不变；本次attempt/result回执及2358项绑定来源SHA通过，pending为空。
- 原实验树40c7476、Stage 2.5树b7aac69、Stage 3A树ef835a6均clean；没有历史源码/模型/缓存/协议/结果修改。
- 输出仅/home/lzf/ldx/outputs/tbps-framework-stage3b/，不提交Git。FAIL report SHA e72a65118cb0c216e2fbb0877152d11524b0254f73cc9f2c8547e91900c7a79c。

## 已完成与冻结历史

Stage 1发布61471db；Stage 2发布4e2a583；Stage 2.5发布b7aac69，真实IRRA最小等价Smoke通过。Stage 3A实现487584a/发布ef835a6，G/H/Joint薄插件及321项CPU tests通过；不等于真实G/H输出已验收。

Stage 2官方Test3074/6156身份及2048 Train/128内部Validation成员已核验。G-v2完整训练/Full、H正式SCST及Epoch7 Selection、Joint Full、消融/AP实验与8个历史资产索引全部冻结。历史IRRA Embedding不能自动重标或跨模型使用。

## 阻断与下一步

缺项原件SHA：f13932c2688cab4e135d063eed6f066016f6ba26e2316cbc201f19b246b298b4。

建议在用户批准后新建派生来源配置显式补齐该FileRef，并让严格CPU预检覆盖实际加载助手/像素读取/状态检查链。当前新check_attack_runtime入口已能在CPU明确发现该阻断；后续H加载路径是否还有问题尚未确认。

本轮单次GPU许可已用完。任何新尝试需要重新明确授权及新不可变attempt，不能恢复/覆盖失败证据或通过换输出目录绕过。不得自动修清单、重试GPU、Full、训练或Stage 4。提交推送后停止等待审查。
