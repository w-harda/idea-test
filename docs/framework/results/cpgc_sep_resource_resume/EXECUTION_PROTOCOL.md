# 恢复调度器运行说明

冻结原生执行树：/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-uap-detach-20261029，Commit36cf91f。恢复工具来自独立codex/cpgc-sep-resource-resume分支，不改变原生producer或来源身份。

```bash
cat /home/lzf/ldx/tmp/cpgc_sep_resume_v1_20261029/status.json
tail -n 40 /home/lzf/ldx/tmp/cpgc_sep_resume_v1_20261029/status.log
cat /home/lzf/ldx/tmp/cpgc_sep_resume_v1_20261029/latest_resources.json
tail -n 40 /home/lzf/ldx/tmp/cpgc_sep_resume_v1_20261029/0001_cpgc_full_cuhk_rasa_target.log
nvidia-smi
```

等待状态为WAITING_RESOURCE；运行状态为RUNNING_STEP，实际组合在status.json的step，日志在log字段。资源完整快照追加resources.jsonl，GPU阶段不依赖调度器外部Agent。flock、独立started及每个原生Attempt锁防重复；不能直接重跑旧启动器或本调度身份。
