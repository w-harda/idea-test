# 项目当前状态（唯一权威入口）

本文件记录Git、实际代码及不可变证据支持的当前状态。历史入口见[EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)；架构见[ARCHITECTURE](ARCHITECTURE.md)。旧README的CURRENT及聊天记忆不是状态依据。

## 当前阶段

Stage 1基础契约、配置dry-run、历史资产只读桥及独立资产管理已实现并完成CPU验证，等待用户源码审查。未授权Stage 2。

- 审计基准：40c7476f92d6bc2458d86029cf7bf8317da0a086。
- 独立分支：codex/tbps-framework-stage1。
- 开发树：/home/lzf/ldx/projects/idea-test-tbps-framework-stage1-20261016。
- 管理规范提交：16a5e3f3086b633b413cfe149e10bf9fc0b589dd。
- 实现提交：IMPLEMENTATION_COMMIT_PENDING（由后续文档提交登记，避免自指SHA）。
- 交付及使用：[Stage 1报告](framework/STAGE1_REPORT.md)、[README](framework/README.md)。

## 本轮完成与验证

- src/tbps：不可变身份/输入/预算/能力/产物契约；配置/权限校验；新输出原子存储；旧资产只读桥。
- configs/framework：真实服务器路径配置示例与Git固定历史来源目录。已知组件仅有描述符，无可执行插件。
- 114项框架CPU tests + 36项历史基础CPU回归通过，共150项；未声称重跑全部历史prototype tests。
- 8类资产、192个唯一来源文件完成元数据/直接证据只读描述；不加载权重或特征，不重算排名。
- 消融日志双标志已从实际文件确认；11个trial子协议绑定父协议。AP B1/B2/B3保持metadata级索引。
- 555个保护文件SHA及mtime_ns不变，含412个旧tracked文件和核验范围内的历史证据；原实验树HEAD仍40c7476且干净。
- 未运行GPU、模型forward、旧恢复/prepare、训练或Full指标重算；没有复制、转换或重写旧资产。

## 历史研究完成事实

G-v2 10 Epoch及IRRA/CUHK Full；H-v1正式SCST和独立Validation Epoch7 Selection；Joint Full及Joint消融Full均已有完成证据。索引及SHA集中维护在EXPERIMENT_REGISTRY，不在此复制全部指标。

## 下一阶段与限制

- Stage 2候选：CUHK官方Dataset Adapter、IRRA Model Adapter和原生Score接口；当前未实现、未授权。
- RaSa/APTM、ICFG/RSTP及未来Baseline不能被标记为可执行；所有真实执行请求在Stage 1拒绝。
- 旧资产路径在其他主机缺失时返回NOT_READY；METADATA_VERIFIED不等于全特征/内部Adam/RNG/训练轨迹新审计。
- 不自动处理pending发布、不承诺Step恢复、不提供GPU/训练/评测入口。
- 新框架只读桥可索引旧结果，实际跨框架特征复用仍需后续Adapter及独立等价验证。
