# C-PGC / SEP 取消固定显存门槛与隔离恢复

本轮仅改变资源准入和显式旧成功产物复用，不改变模型/攻击/数值定义。
实时确认旧监督器909831停在SEP ICFG IRRA Target之前，没有GPU子进程、没有该Target Attempt。
Ollama PID2749989占用19700MiB，旧诊断空闲4343MiB，其他主机资源达标；仅停止旧等待监督器，不干预Ollama。

runtime、活动调度器、CLI所用资源函数和Smoke峰值审计已删除固定显存否决；历史冻结版本保持不变。
独立profiling入口的固定20GiB要求也移除，但本轮不运行新的profiling实验。

历史6组CPGC Full+CPU审计、27个v3已完成Source/Target/Audit绑定通过只读Manifest/SHA/CSV-summary核验。
Smoke各单元额外校验payload SHA；历史Full复核已有完成/审计证据，不重新计算排名、不重新运行原审计。
旧成功产物在真实父配置身份下引用，不复制或重写Manifest。

新GPU验证将直接使用首个未完成的SEP ICFG IRRA Target Smoke及其CPU审计，不重复已有Source或其他完成Smoke。
正式Full没有因CPU PASS而被提前视为GPU通过。最终GPU结果与后台PID由后续独立启动证据登记。

## CPU验证完成

资源/算法/只读桥CPU回归56项PASS，缓存与调度回归19项PASS，共75个不同用例。
调整调度用例预期时出现一次括号语法错误，仅测试收集失败，原日志保留；修正后19项完整通过，不影响或重跑GPU。
8个数值/攻击模块target、victims、cache、sep_source、sep_bridge、official、dataset、verified_store与v3逐字节相同。
真实即时资源检查在空闲显存低于旧门槛时返回READY；所有主机门禁仍有效。
