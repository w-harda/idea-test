# G-v2 Phase 2.9A：官方Test评测开发与10+10 Smoke报告

基准Commit：2a3f166208c577b5cabb68e54424c936af07b105；仓库w-harda/idea-test，分支codex/gv2-per-attribute-cache。新增独立evaluation_phase29a目录，原训练及全部资产不改。

## 实施、验证与边界

本阶段完成7个Python源码/tests文件、README和本中文报告。329项CPU tests通过（40新+289原）；真实CPU preflight禁止CUDA/模型/backward/JVP通过。直接复用官方CUHKPEDES/TextDataset/build_transforms语义和rank()，解析多正例AP/INP/CMC例、相同分数/tie、PID缺失及下降率边界通过。原官方图像reader无限IO重试替换为等价PIL RGB单次读取，失败立即停止；像素预处理与官方Normalize逐值相等CPU验证通过。

安全与恢复覆盖：默认CPU；Smoke真实模型硬限10+10，100/Full独立手动确认；不改caption；所有G只图像/一次前向；干净文本一次编码后共享；Clean gate先于G；单元SHA/身份/Checkpoint隔离；原子保存、并发锁、有效缓存不覆盖、未提交回执拒绝自动重复G；Ctrl-C/SIGTERM批次边界恢复合成CPU验证；完成态无模型重入、只读结果重算通过。

未执行100+100或Full，未运行全量Clean指标重算、训练、属性评分/Top-K/JVP、任何新方法。Full和100模式的真实GPU规模性能、显存和正式攻击结果尚待用户手动运行。

## 真实GPU Smoke

固定官方index和seed42，5 PID各2 Gallery/2 Query，严格10+10，查询均存在两张同PID正例。原始caption和PID保持，无camera过滤。一次GPU进程完成Clean、Epoch1、Epoch3、Epoch10，随后停止。

| 项目 | 实测 |
| --- | --- |
| Gallery/Query/PID | 10/10/5 |
| Text编码 | 10条/10次调用；三Checkpoint复用 |
| IRRA图像编码 | Clean/三个Checkpoint各10，总40 |
| G前向 | Epoch1/3/10各10，总30 |
| backward/optimizer/JVP | 0/0/0 |
| 受控链路耗时 | 11.663571秒，不含外层CPU preflight |
| peak allocated/reserved | 795.328125/874 MiB |
| 最大L∞ | 0.0313725471496582≤8/255 |
| fresh Clean图像参考误差 | 0 |
| fresh Text参考最大误差 | 1.7508864402770996e-7 |
| IRRA/G状态 | 参数及buffers SHA exact |
| 479项来源SHA/100缓存文件SHA和mtime | 核验通过 |

四组五项小样本指标均100%、相对下降率均0。这是小规模链路结果，不作为正式攻击成功或失败结论，不挑选新子集重测。

实际Gallery row_id：37501:0、37503:0、38488:0、38489:0、39172:0、39173:0、39176:0、39177:0、39490:0、39491:0（均cuhk前缀）。对应PID12123/12444/12672/12674/12767各两张；Query为各图官方caption_index=0。部分路径含train_query，这是原始目录名，划分严格按annotation.split=test及官方reader核验，不按目录名推断。

Smoke前nvidia-smi已不列出此前其他GPU进程；本任务未发出停止或修改其他进程的命令。Smoke后nvidia-smi已无本任务GPU进程。只补装prettytable3.16.0/easydict1.13以原样导入官方模块，未升级原Torch/NumPy。

## 协议与SHA

官方Test3074/6156，原顺序/原始文本、同PID全部正例；384×128 RGB像素空间投影，官方mean/std、fp32、L2 feature、dot similarity。mAP、R@1/5/10、mINP直接官方rank；下降率(Clean−Adv)/Clean×100，负数允许，零Clean为null。同模型IRRA，不宣称迁移。

Checkpoint固定SHA及完整训练关联日志核验，原Epoch1–10、2176属性缓存、训练/Validation日志不写入。新输出仅ROOT/outputs/evaluation_phase29a。CPU预检核验训练源/官方/BPE/数据/参考/评测源码和配置；缓存协议含scope、layout、Checkpoint、batch和runtime，任何变化拒绝混用。

历史Clean参考SHA：3ef23d81e097f88975d9f9151cba2a58ca50b2a20547fba9c908027674bf1800。只读取旧官方fp32特征和layout，不导入G-v1或integration、不使用属性/JVP。每张原图SHA与历史来源逐一匹配。CPU读取全量既有特征后切片不是新编码；GPU Smoke只前向所选10+10。

Full Clean参考mAP66.0974695/R1 73.1481481/R5 89.8797921/R10 93.7134503/mINP50.2776649，仅引用旧SHA锁定报告，本阶段未进行全量评测。Clean feature阈值2e-5、指标阈值0.001个百分点在GPU前固定；异常写clean_mismatch.json并停止，不自动放宽。

## 四类实际命令（100/Full仅用户）

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

## 操作与恢复说明

命令以前台运行、日志重定向。用tail -f查看对应outputs/phase29a_<scope>_run.log，FEATURE_COMPLETE显示进度；Ctrl-C或向evaluate.py PID发送SIGTERM，当前原子批次提交后SAFE_STOP。恢复原样执行同一命令/同输出，不改batch和协议；完成有效批次不重复G/编码。SIGKILL/OOM后未发布receipt拒绝自动重算，保留证据人工审查，不删除回执绕过限制。

用ps -eo pid,args匹配evaluate.py和nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader确认本任务PID退出。结果在各scope输出的summary.json、metrics.csv；protocol.json含Checkpoint/来源SHA，completed.json绑定摘要；features目录各单元metadata绑定tensor SHA；attempts含资源、计数、停止或失败记录。

结果CPU审计：将同scope命令--action run改为--action audit、CUDA_VISIBLE_DEVICES设为空，保持其余规模/Checkpoint/batch/reuse/手动确认参数。Smoke CPU审计已实际通过；100/Full审计只在用户完成相应模式后由用户执行。sha256sum对三个Checkpoint核对SHA。完整可复制审计和进度命令见README。

最终G-v2三Checkpoint SHA：Epoch1 876547ffeb827cbb4691bbf5b5d473da495590d15df2adba12c9e1415e52a27a；Epoch3 a42a6dc0fcae269ee2e8efd379b4a3f3c947f22c50181a56d15acf82a8450d02；Epoch10 f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。

## 已知限制与交付

Full/100实际规模尚未执行，未来可能暴露批量数值或运行问题；Full Clean检查不通过必须停止报告，不能套用Smoke通过结论。仅支持原子批次边界安全停止与有效缓存恢复，不承诺SIGKILL中途一次性精确恢复；未提交回执故意阻止重复G。只保存float32 feature和预算诊断，不保存对抗PNG；所有feature来自真实生成图的即时编码。

只提交新增评测源码/tests/README及本中文报告；数据、Checkpoint、缓存、outputs、日志、训练资产和Windows状态文档不提交。329CPU tests及10+10 Smoke验证通过后发布，停止等待独立审查；没有自动运行、后台调度或拆批执行100/Full，不进入Phase2.9B。
