# SEP WordBudget1：逐组独立报告与即时发布

## 当前授权与边界

六组Full连续执行，不暂停、不等待人工审核。原Supervisor682941及当前科学子进程保持运行；独立发布器不发送信号，不写原工作树、控制目录、配置、Manifest或产物，不重新执行生成、评分、Clean或CPU数值审计。

截至接线预检，CUHK Source已完成6156条，CUHK × IRRA正式Full及独立CPU审计PASS，原链已自动进入CUHK × RaSa。进度以原控制目录实时回执为准，未完成组不发布正式指标。

## 独立发布机制

- 发布工作树：/home/lzf/ldx/projects/idea-test-tbps-sep-word-budget1-reports-20261105。
- 分支：codex/sep-word-budget1-reports，独立于科学执行分支codex/sep-word-budget1。
- 入口：operations/sep_word_budget1_publish.py。
- 配置：configs/framework/sep_word_budget1_reports.server.json，锁定原实验配置SHA256。
- 发布控制目录：/home/lzf/ldx/tmp/sep_word_budget1_reports_20261105。
- 原Full控制目录：/home/lzf/ldx/tmp/sep_word_budget1_20261105。
- 每30秒检查已原子提交的Full CPU审计回执。只有正式Full、完整Query/Gallery、WordBudget1零违规、Checkpoint及指标来源一致时才发布。
- 校验审计Manifest、Target汇总、CSV、输入身份、冻结Source完整索引和来源SHA。沿用已完成的独立CPU数值审计，不重复读取大型评分Tensor，不加载模型或CUDA。
- 每组独立Git提交并立即push，核对远端SHA后记录固定Commit报告链接。发布失败保留本地报告、提交和错误日志，每300秒仅重试Git发布，不干扰科学执行。
- 全六组均通过审计后才生成SUMMARY.md。尚未完成的组没有占位正式指标。
- flock及独占started.json阻止重复发布器启动。完成六组发布后自动退出。

## 报告路径

docs/framework/results/sep_word_budget1下分别生成cuhk_irra、cuhk_rasa、rstp_irra、rstp_rasa、icfg_irra、icfg_rasa/REPORT.md。报告包括完整覆盖、四项指标、NOOP及原因、替换1词数、零违规统计、Checkpoint和配置、耗时、产物路径及独立审计SHA。每组GitHub链接记录在status.json的groups字段及events.jsonl的REPORT_PUBLISHED事件。

## 针对性验证

新增发布器CPU测试12/12 PASS，耗时3.49秒。覆盖未审计拒绝发布、真实产物结构、只读且不打开评分Tensor、Smoke/不完整覆盖/预算违规/Checkpoint及指标异常拒绝、六组汇总门禁、GitHub失败保留本地报告、重复轮询幂等。未重复既有176项或攻击预算测试，未运行任何新攻击、评分或GPU实验。

CPU证据：/home/lzf/ldx/tmp/sep_word_budget1_reports_20261105/tests.log和tests.xml。真实只读接线已识别CUHK × IRRA Full及CPU审计，其余组等待正式完成。

## 运行查看

发布进程、启动Commit及隔离信息见控制目录started.json，实时发布状态见status.json。查看已发布链接与运行日志：

```bash
cat /home/lzf/ldx/tmp/sep_word_budget1_reports_20261105/status.json; tail -n 20 /home/lzf/ldx/tmp/sep_word_budget1_reports_20261105/publisher.log
```

GitHub发布与原Supervisor最终汇总使用不同工作树和分支，避免Git索引或报告提交冲突。本功能不改变六组执行顺序、WordBudget1、Source-only协议或Victim原生评分。
