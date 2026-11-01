# 共享GPU执行授权与隔离切换

2026-11-01用户明确允许与SEP共享GPU、保留SEP运行。本变更仅新增独立operations包装器。

- 数值执行Commit仍为7973128ec1fc73ab8e2428dfbdea600a565421a9，原工作树与配置完全不变。
- 原配置SHA仍为8d6d63e90fa79828de8e5fd45f6bcf76872bc4ae32f48b386aec0973f31fe8e7，原43项CPU测试和真实来源Preflight继续有效；启动时重新验证全部producer与90869份保护记录。
- 新包装器7项CPU测试PASS；仅取消本消融对SEP全局调度锁及其他GPU计算进程退出的等待。
- 保留同一消融专用互斥锁；旧等待器1063550已核验无子进程、状态WAITING_SEP、原生输出未创建后定向SIGTERM。旧控制文件SHA逐项未变。
- SEP进程2960820/3041291未收到信号，工作树、配置、日志和输出不修改。
- 评分仍是IRRA后CPU审计，再RaSa后CPU审计；任何真实OOM、数值不一致、SHA或审计失败首错停止，不重试。
- 原生消融输出身份未消费，合法沿用原v1输出空间；新控制目录独立，保留旧等待事件而不改写。

控制目录：/home/lzf/ldx/tmp/cpgc_cuhk_modal_ablation_shared_v1_20261101/。
执行入口：operations/run_cpgc_modal_shared.py --allow-shared-gpu-ablation。
底层源码/配置继续从原冻结执行树读取，新包装器不能冒充原producer SHA。

最终结果仍由两Victim评分及独立审计的真实回执生成，并自动发布codex/cpgc-cuhk-modal-results。未取得回执前不宣称评分或审计PASS。
