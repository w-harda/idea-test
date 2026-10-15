# G-v2 Phase 2.9A：官方IRRA/CUHK-PEDES Test评测

仅开发程序并完成一次固定10 Gallery+10 Query GPU Smoke。100+100、Full3074+6156仅由用户亲自在服务器终端执行；本README中的命令不是自动脚本，不会由tests/安装/后台任务执行。Smoke后已停止GPU，没有进入Phase2.9B。

## 文件与协议

eval_common.py/eval_protocol.py：独立路径、官方数据布局和来源SHA；eval_metrics.py：直接调用官方utils/metrics.py::rank；eval_storage.py：不可变原子特征批次；eval_backend.py：官方IRRA与image-only G；evaluate.py：权限/前向预算、Clean gate、恢复和CLI；test_evaluation.py：40项CPU测试。均为新增独立文件，原训练和IRRA源码不改。

官方CUHKPEDES reader核对3074图/6156原始caption；全部同PID正例、无camera过滤。Resize384×128+RGB/ToTensor后在[0,1]生成扰动，再执行官方mean/std和L2归一化。TextDataset/tokenize原样复用，77 token、官方截断规则。每组同一官方IRRA/fp32权重；G只接收图像，每图/Checkpoint一次G，无梯度/optimizer/属性/JVP。

小规模子集固定seed42：SHA排序选择5/50个具有至少2图的PID，每PID2图，选这2图caption_index=0作为2条原始Query；最终保留官方index顺序。不根据属性、K或攻击结果挑样本。Full包括全部Test图和全部caption。

40项新CPU+289项历史回归共329项通过。原环境缺少官方reader/rank的prettytable/easydict，本阶段仅补装prettytable==3.16.0、easydict==1.13，未升级Torch/NumPy。协议记录依赖版本和BPE/官方源码SHA，环境现已具备。

## 四类可直接复制命令

每条命令显式包含真实目录、Python、CUDA/cache环境、三个Checkpoint、规模和隔离输出。全部以前台执行并将日志写文件；另一个终端查看进度。默认CLI只preflight。Smoke当前完成态重复同命令只做CPU/SHA重入，不重新调用模型。不要改变已完成输出的源码、规模、batch或Checkpoint再强行复用。

### CPU Preflight：Codex可运行

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/evaluation_phase29a
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python -u evaluate.py \
  --action preflight --mode smoke --gallery-count 10 --query-count 10 --image-batch 1 --text-batch 1 --checkpoint1 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0001/checkpoint.pt --checkpoint3 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0003/checkpoint.pt --checkpoint10 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/evaluation_phase29a/smoke10 > /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/phase29a_smoke10_preflight.log 2>&1
```

### 10+10 Smoke：Codex可运行；本阶段已完成，不要另建目录重复GPU测试

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/evaluation_phase29a
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python -u evaluate.py \
  --action run --mode smoke --gallery-count 10 --query-count 10 --image-batch 1 --text-batch 1 --checkpoint1 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0001/checkpoint.pt --checkpoint3 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0003/checkpoint.pt --checkpoint10 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/evaluation_phase29a/smoke10 > /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/phase29a_smoke10_run.log 2>&1
```

### 100+100：仅用户亲自在服务器终端执行，Codex未运行

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/evaluation_phase29a
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python -u evaluate.py \
  --action run --mode diagnostic100 --gallery-count 100 --query-count 100 --image-batch 1 --text-batch 1 --checkpoint1 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0001/checkpoint.pt --checkpoint3 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0003/checkpoint.pt --checkpoint10 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/evaluation_phase29a/diagnostic100 --user-run-100 > /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/phase29a_diagnostic100_run.log 2>&1
```

### Full官方Test：仅用户亲自在服务器终端执行，Codex未运行

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/evaluation_phase29a
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python -u evaluate.py \
  --action run --mode full --gallery-count 3074 --query-count 6156 --image-batch 1 --text-batch 1 --checkpoint1 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0001/checkpoint.pt --checkpoint3 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0003/checkpoint.pt --checkpoint10 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/evaluation_phase29a/full --user-run-full --reuse-clean-reference > /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/phase29a_full_run.log 2>&1
```

Full命令显式复用SHA锁定的历史官方Clean图像/文本特征，避免再次编码原始3074/6156；这不是本阶段先编码全量再截取。Smoke和100+100命令则真实编码所选Clean图像与原始Text各一次。Full中所有Checkpoint共用同一完整Text/对抗Gallery检索协议。若用户希望Full重新编码Clean，可在首次运行前移除--reuse-clean-reference，使用新的空输出；当前Codex不得执行。

## 进度、停止和恢复

```bash
tail -f /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/phase29a_diagnostic100_run.log
# Full对应日志：
tail -f /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/phase29a_full_run.log
```

日志先打印CPU_PREFLIGHT，再逐批FEATURE_COMPLETE text/clean/epoch1/epoch3/epoch10，最后EVALUATION_COMPLETE。SIGINT(Ctrl-C)/SIGTERM只设置停止请求，当前原子批次保存完成后停止；看到SAFE_STOP后可用完全相同命令/同目录恢复，不改batch/协议。完整有效批次SHA验证后跳过，已完成G不重复前向。

另一终端可先查询evaluate.py PID，再对该PID发送SIGTERM；不要停止其他任务。SIGKILL、断电或OOM若发生在计算后而原子发布前，会留下未提交receipt；程序拒绝自动重算，保留pending和failure证据，需人工审查，不能删除receipt绕过一次G约束。最高权限不允许Codex通过新目录或拆批重复大规模实验。

```bash
ps -eo pid,args | grep '[p]ython.*evaluate.py'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

GPU退出以evaluate.py PID从进程列表和nvidia-smi compute-apps均消失为准；不要用无关GPU进程是否存在判断本任务退出。

## 结果、SHA及CPU只读验证

每个输出目录含protocol.json（身份、Checkpoint/源码/数据SHA）、features/<family>/<batch>/features.pt和metadata.json、receipts、attempts/<UUID>/execution.json或interruption/failure.json、clean_verified.json、summary.json、metrics.csv、completed.json。特征批次原子发布、不可变、SHA绑定；不存在损坏缓存自动删除/重算回退。只缓存float32特征与预算诊断，不保存PNG或有损对抗图，图像真实生成后立即编码。

summary含五项指标及相对Clean下降率：(Clean−Adv)/Clean×100，负数保留，Clean=0时为null；单victim IRRA，不称跨模型迁移。Smoke/100子集指标不能当作Full结果。

Clean在任何G生成之前验证：新编码Clean/Text与所选固定参考最大绝对差≤2e-5；五项指标与对应参考的差≤0.001个百分点。Full固定参考Clean mAP66.0974695/R1 73.1481481/R5 89.8797921/R10 93.7134503/mINP50.2776649。容差在Smoke前固定，不会自动放宽；失败写clean_mismatch.json并停止。本阶段未重算Full Clean指标或验证Full运行性能。

10+10结果CPU只读审计（不编码、不修改已完成特征/结果）：

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/evaluation_phase29a
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python -u evaluate.py \
  --action audit --mode smoke --gallery-count 10 --query-count 10 --image-batch 1 --text-batch 1 --checkpoint1 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0001/checkpoint.pt --checkpoint3 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0003/checkpoint.pt --checkpoint10 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/evaluation_phase29a/smoke10 > /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/phase29a_smoke10_audit.log 2>&1
```

100/Full审计仅由用户执行，将相应运行命令的--action run改为--action audit、CUDA_VISIBLE_DEVICES改为空字符串，仍保留对应--user-run-100/--user-run-full、规模、batch、reuse和输出参数。审计验证结果摘要SHA、每个缓存SHA/身份/预算/指标及当前来源；不构造模型。

```bash
sha256sum /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0001/checkpoint.pt /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0003/checkpoint.pt /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt
```

预期Checkpoint SHA：

- Epoch1：876547ffeb827cbb4691bbf5b5d473da495590d15df2adba12c9e1415e52a27a
- Epoch3：a42a6dc0fcae269ee2e8efd379b4a3f3c947f22c50181a56d15acf82a8450d02
- Epoch10：f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e

恢复必须保留完整输出、关联训练Checkpoint日志、官方资产和固定参考缓存；移动/修改来源会被拒绝。CPU preflight会核验全部10 Epoch资产及历史Clean来源，不进行图像/文本编码。

## 已完成Smoke与阶段边界

仅一次10+10真实GPU链路：5 PID、每PID2个正例。Text10，Clean/epoch1/epoch3/epoch10各10张IRRA图像编码；三个G各10次，合计G30、IRRA图像40、文本10。11.6636秒（不含外层preflight），peak allocated795.328MiB/reserved874MiB；最大L∞0.0313725471496582≤8/255，所有特征有限/归一化，IRRA/G完整状态SHA保持。

fresh Clean图像与历史参考差0，Text最大差1.75088644e-7；小样本四组五项指标均100%、下降率0，不能据此判断攻击有效或无效。479项来源SHA核验，完成态CPU禁模型/CUDA重入以及只读审计通过，100个缓存文件SHA/mtime保持。GPU已退出，无100+100/Full执行、无重训练/评分/JVP，等待独立审查。详细报告见仓库docs/stages/09_2_9A_gv2_official_test_smoke.md。
