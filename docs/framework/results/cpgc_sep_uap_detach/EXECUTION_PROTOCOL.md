# v2执行与状态

仅使用operations/run_cpgc_sep_full.sh的新工作树版本，不重跑v1控制目录。新配置configs/framework/cpgc_sep_tbps.uap_detach.server.json。

```bash
cat /home/lzf/ldx/tmp/cpgc_sep_full_v2_20261029/phase
tail -n 50 /home/lzf/ldx/tmp/cpgc_sep_full_v2_20261029/status.log
tail -n 50 /home/lzf/ldx/tmp/cpgc_sep_full_v2_20261029/launcher.log
nvidia-smi
```

首次启动前检查新控制/输出不存在、无相同进程，完整CPU预检及即时资源通过后nohup setsid后台启动；started、FAIL与Full六Smoke门禁完全保留。失败不得直接重跑，不覆盖旧或新失败产物。
