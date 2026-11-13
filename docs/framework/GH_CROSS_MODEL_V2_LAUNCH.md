# 三版六组迁移v2启动记录

2026年11月13日13:01（北京时间）已启动独立v2流水线。执行Commit `f0d1afb7848a5f412f163b539a51f3d2f838d53c`，配置SHA `9a875cf8a28c2d3d1a3b3598fa87a81466575c02e5037765cd72465e1294b203`。

13:02一次启动核验：Supervisor PID/PGID/SID均3753008、PPID1、无TTY；子进程3753035存活，阶段verify_inputs_cooperative，执行原文件完整SHA复核。启动时无FAIL。这里只确认启动，不代表真实GPU Smoke或六组Full已通过。

三版均固定各自Round60：Cooperative、原Push-pull、图像增强Push-pull，分别迁移RaSa/APTM，共六组。90项所选CPU测试通过，1项固定旧APTM工作树配置的用例不适用而排除；三版完整输入审计有效，增强版新增审计87.10秒；12条Smoke/Full真实全新进程冷启动路径通过、CUDA和真实目标模型加载均0。旧v1失败现场和13份控制文件SHA保持。

后台顺序：三份原输入SHA复核→六组完整输入CPU冷启动核验→六组真实Smoke及独立CPU审计→六组3074/6156正式Clean/Joint Full及独立CPU审计→中文报告/原精度表格/来源发布。任一真实失败停止保留，无自动重试。没有重新训练或生成G/H攻击。

```powershell
ssh tbps-server "/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-cross-model-v2-20261113/operations/gh_cross_model_v2_status.py"
```

完整范围、预算、原生评分和修复证据见[协议](GH_CROSS_MODEL_V2.md)。启动后由用户执行检查命令，助手不持续监控。正式结果以新独立CPU审计回执为准。
