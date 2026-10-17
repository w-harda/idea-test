# Stage4C-2受控物化Smoke接口与执行边界

本轮最终状态为 **FAIL**：G像素物化、历史H文本物化通过；IRRA尚未恢复或前向，不能声明端到端等价。具体结果见[阶段报告](STAGE4C2_REPORT.md)。

## 文件与职责

- `src/tbps/materialization/smoke_support.py`：固定样本/预算、C1计划派生、75项G/H导入闭包、官方IRRA实际导入闭包、历史完成来源、限定特征批次只读载入、计数与数值比较。
- `src/tbps/materialization/smoke.py`：单次Attempt、带令牌的私有子进程入口、G→历史文本→IRRA三个独立会话、资源监视、不可变Claim/阶段报告/最终回执。
- `configs/framework/stage4c2_smoke.server.json`：冻结样本、来源SHA、FP32、seed42、1e-6、每进程2GiB与Attempt 300秒。
- `tests/framework/test_stage4c2_smoke.py`：CPU门禁、计数、数值/Storage、来源配置、无默认GPU副作用、失败关闭和此次启动顺序问题复现。

公共契约、旧Plugin、C1 MaterializedStore、历史Loader与IRRAAdapter均未修改。新的计划只派生本工作树的公共来源身份，保留C1父计划/回执和所有旧来源，未覆盖旧协议。

## CPU预检命令

以下命令只校验来源和规则，不加载权重/神经网络，不初始化CUDA，不计算排名：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4c2-20261017
CUDA_VISIBLE_DEVICES="" TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.materialization.smoke --config configs/framework/stage4c2_smoke.server.json
```

成功标志：`GH_RUNTIME_PREFLIGHT_PASS`及`MATERIALIZED_SMOKE_CPU_PREFLIGHT_PASS`。官方模型build和Tokenizer导入也处于禁止torch.load、神经构造/前向、CUDA与导数的CPU闸门下，执行文件逐项SHA核验、命名空间恢复。

## 本轮唯一GPU执行记录（不要再次运行）

本轮使用与CPU命令相同入口，显式设置`CUDA_VISIBLE_DEVICES=0`并增加`--allow-gpu-smoke`。固定输出：

`/home/lzf/ldx/outputs/tbps-framework-stage4c2/materialized_smoke_v1/`

该Attempt已经消费。入口拒绝现存输出，不提供resume、替换输出、增加样本或重试选项。私有阶段入口必须有父Attempt回执、CPU通过证据、令牌及前一阶段PASS，不是可绕过预检的独立GPU命令。本轮之后不得再次运行GPU命令。

失败原因：child在历史特征/文本只读CPU校验之前调用CUDA峰值统计初始化，原CPU门禁拒绝；需要后续独立CPU-only修复并另行申请新的GPU Attempt。**此处未修复或重试。**

## 产物与隔离

G阶段只加载G10一次，各图一次前向；原RGB、Adv RGB、effective_delta保存FP32 Tensor，raw delta单独存储，全部读回bitwise exact。文字阶段不恢复H，复用原完整action与UTF-8内容。IRRA阶段设计为只读已保存的像素/文本，并独立对照历史对应行；实际在读取阶段被门禁阻断。

没有调用完整3074/6156集合的seal_collection。四个小样本单元是独立Smoke证明，未伪装为全量物化。攻击产物没有目标模型身份或反馈入口；目标表示另存，不回传给生成器。

本轮没有Full排名、mAP评测、G/H训练、JVP、RaSa/APTM或后续阶段授权。
