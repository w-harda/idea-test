# G-v2 Epoch Sweep共享GPU资源策略及唯一Smoke

基准0dba135a3bd0a59fdf3ab220acfd6066a6102c5a。本轮只调整资源准入；
旧GPU利用率≤10%/空闲≥8192MiB改为空闲≥6144MiB，利用率及其他进程仅监控。
共享由用户明确授权，不终止/暂停/抢占其他任务；快照不保证后续不会OOM，失败停止且不重试。

resources之外运行函数AST exact；配置仅资源字段和对应单项producer SHA改变。
Target Runner、模型、权重、输入/样本、FP32、batch1、query_chunk128、原生Top-K、
Tokenizer、图像预处理、8/255投影及生成函数不变。来源SHA门禁保留。
[策略差异证据](results/gv2_epoch_sweep_shared_smoke/resource_policy.json)。

资源及公共CPU测试142项PASS，原生适用92项PASS；旧APTM绝对路径用例沿用原有单项排除。
本修复提交时GPU Smoke尚未开始；后续结果须由不可变Attempt证明，不预先声明PASS。
仍不授权正式Full或训练。旧准备报告的RESOURCE_BUSY是历史快照，不改写旧证据。
