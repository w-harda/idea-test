# G-v2 Phase 2.9C：Full特征缓存Storage最小修复

基准Commit：ab3de99a7209923cba23a5ed9f5fc017f70f3d2f；分支codex/gv2-per-attribute-cache。

## 修复结论

原Full输出可原地恢复，不需要删除、转换、重写任何缓存或计算回执。只新增源码、明确限定的来源兼容校验和中文文档；正式恢复及CPU Full审计由用户亲自执行。

旧PID3354580在实施前已经退出，本任务未发送kill信号；系统可用内存约117GiB。原Full有6156 Text、3074 Clean批次和相同数量回执，无pending批次，无Epoch1/3/10缓存/回执、无clean_verified/completed标记。原失败attempt目录保留。

## 代码变更

- eval_storage.py：publish保存前detach().cpu().clone()；load_unit在读取旧文件后立即clone逻辑Tensor并释放原Storage。兼容旧metadata和新增生产者标记；SHA、身份、统计与预算检查不弱化。load/gather增加逐批安全停止检查。
- evaluate.py：将停止回调贯穿Clean gather、结果gather及指标阶段边界；仅新增显式--allow-storage-fix-compat和兼容记录接线。
- eval_protocol.py：普通SHA门禁保持；仅显式修复参数进入限定兼容分支，返回原protocol，任何其他训练/评测参数差异仍拒绝。
- 新增eval_compat.py及storage_fix_compatibility.json：精确旧/新源码SHA清单，基准Git对象证明旧SHA，当前Git HEAD证明已提交补丁；仅允许原Full路径与固定协议。其余全部旧来源继续校验。
- 新增test_storage_fix.py；更新README及本报告。eval_metrics.py、官方IRRA rank、G、训练代码、Loss、Checkpoint与数据协议定义未改。

## CPU验证

最终357项通过：28新增Storage/兼容/停止测试，原40评测tests，原289历史CPU tests。首轮1项测试夹具以不存在路径模拟SHA错误而得到FileNotFoundError；修正夹具为真实文件错误SHA后全通过，证据保留outputs/phase29c_storage_fix/。未据此放宽生产校验。

覆盖旧Storage重现、clone紧凑保存/读取、加载原Tensor释放、新旧缓存混用、逐值一致、官方指标小规模精确一致、SHA/身份失败拒绝、未提交兼容清单拒绝、无隐式来源绕过、其他参数或未列出来源变更拒绝、新metadata生产者记录完整性、SIGTERM真实信号及逐批停止/恢复。

另对真实旧Text/Clean缓存各只读1条：底层Storage原12607488/6295552字节，读取返回均2048字节，512维特征torch.equal exact，原文件SHA及mtime exact。该CPU进程峰值RSS403.64MiB包含Torch导入和原始对照Tensor；不是Full规模内存实测。没有运行Full指标/100+100/GPU Smoke、G或IRRA前向、训练、编码、JVP。

## 证据与兼容边界

原协议规范化SHA：1aceae9a0452faea888a2656d318d9084ae0048251fa1cdb084fb47cc8ae721d。
原protocol.json文件SHA：13f8287da3324c578198437a9e8c7d4501f6c2619b741db5f9636804c2aa55dc。

3543项原来源中仅3个评测文件替换为清单中的精确补丁SHA；其余3540项保持原校验，新增2个Python和1个清单来源。基准旧代码从Git ab3de99对象核验，不修改原protocol及其旧source_sha256。补丁必须已提交且工作区字节与Git HEAD/清单一致；不提供任意SHA忽略或通用fallback。

用户恢复时才在原Full目录新增storage_fix_compatibility_record.json，记录原协议、发布Commit、清单和补丁SHA；新生成的紧凑批次metadata绑定其SHA。已有metadata和回执原样保留。新日志使用phase29c_full_*，不覆盖旧运行日志。原历史大缓存约90.3GiB不缩盘，恢复仍需读取和SHA核验，其磁盘I/O可能较慢，但不会在gather列表中累计保存整块Storage。

## 用户实际恢复命令

以下命令仅交付，Codex没有执行。

### 用户手动CPU Preflight

只核对来源与兼容性，不编码、不计算Full检索指标。新发布Commit须已在当前服务器工作树，不能先在未提交补丁上运行。

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/evaluation_phase29a
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python -u evaluate.py \
  --action preflight --mode full --gallery-count 3074 --query-count 6156 --image-batch 1 --text-batch 1 \
  --checkpoint1 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0001/checkpoint.pt \
  --checkpoint3 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0003/checkpoint.pt \
  --checkpoint10 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt \
  --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/evaluation_phase29a/full \
  --user-run-full --reuse-clean-reference --allow-storage-fix-compat \
  > /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/phase29c_full_preflight.log 2>&1
```

### 用户手动原地恢复Full

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/evaluation_phase29a
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python -u evaluate.py \
  --action run --mode full --gallery-count 3074 --query-count 6156 --image-batch 1 --text-batch 1 \
  --checkpoint1 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0001/checkpoint.pt \
  --checkpoint3 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0003/checkpoint.pt \
  --checkpoint10 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt \
  --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/evaluation_phase29a/full \
  --user-run-full --reuse-clean-reference --allow-storage-fix-compat \
  > /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/phase29c_full_resume.log 2>&1
```

### 进度与停止

```bash
tail -f /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/phase29c_full_resume.log
```

旧Text/Clean加载与SHA检查仍有较多磁盘I/O，期间未必有FEATURE_COMPLETE输出；不能仅凭暂时无日志判断内存问题复发。新停止检查覆盖逐条load/gather、拼接和指标阶段边界；Ctrl-C/SIGTERM会在检查点停止，仍不承诺中断正在执行的单个torch.load、hash或官方rank操作。保持输出、batch、参数原样，再执行同一命令恢复。新日志不会覆盖旧phase29a_full_run.log。

### 用户完成后CPU审计

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/evaluation_phase29a
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python -u evaluate.py \
  --action audit --mode full --gallery-count 3074 --query-count 6156 --image-batch 1 --text-batch 1 \
  --checkpoint1 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0001/checkpoint.pt \
  --checkpoint3 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0003/checkpoint.pt \
  --checkpoint10 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt \
  --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/evaluation_phase29a/full \
  --user-run-full --reuse-clean-reference --allow-storage-fix-compat \
  > /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/phase29c_full_audit.log 2>&1
```


## 发布与停止

仅提交评测修复、测试、清单、README及本中文报告；不提交缓存、outputs、Checkpoint、数据或日志。修复提交后停止等待独立审查，不自动恢复Full。
