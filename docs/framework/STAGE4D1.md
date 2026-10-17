# Stage4D-1：CUHK Test全量物化接口与执行说明

本阶段只执行历史H7文本CPU物化。真实G/H/IRRA模型加载、神经前向及GPU均为0；3074图真实G生成须另行授权Stage4D-2。实际结果见[阶段报告](STAGE4D1_REPORT.md)。

## 接口与来源

- `full_prepare.prepare(config)`核验Stage4C1来源审计、Stage4C3真实小样本PASS、官方Test顺序、冻结G10/H7/Selection及相关SHA，返回同一完整MaterializationPlan。
- `HistoricalTextSource`只读取旧Joint Full的Unicode字符串与完整动作，不读取Embedding、不调用H、不根据目标模型调整动作。
- `publish_texts`独立逐Query原子发布、立即读回；全6156完成及首尾SHA通过后，才发布`text-full`语义回执。保留6116编辑、40 NOOP及13537字符替换；不是Joint全量完成。
- `stream_images`先核验pending/Claim/已完成证明，再进入一次显式G工厂，以batch1逐Gallery生成。复用原`generate_image_once`、G插件、PIL RGB/bilinear Resize384×128/ToTensor及原投影；当前仅Fixture验证。
- `full_audit`只读逐条对照旧源、完整动作/trace、UTF-8 SHA、官方顺序及固定动作摘要；不计算检索指标。

图文产物共享Plan，Gallery/Query独立，未zip配对。产物无victim ModelSpace，不能反馈目标Score/Gradient/Tokenizer。后续RaSa/APTM读取同一FP32 RGB/Unicode，并分别产生自身Representation/Score缓存。

## 大规模存储局部优化

新增`StreamingMaterializedStore`保持原Plan/单元/集合格式，不修改原ArtifactStore、MaterializedStore或历史Store。成员索引避免反复遍历完整Manifest；有限作用域`BulkSession`仅缓存明确固定来源的已验证SHA，开始、强制集合边界、结束均完整SHA；直接来源每次检查inode/size/mtime/ctime，全部来源stat扫描间隔最多1秒。未知文件、输出内容SHA仍每次完整核验。

来源变化即拒绝，异常退出恢复模块和锁；完整语义回执在会话结束及强制SHA通过后发布。`ValidatedActions`最多缓存64条完全相同Caption+完整action验证结果，不改变属性或Unicode动作。原子fsync、Claim、pending、不可变发布均保留。

## 服务器命令

工作树固定：`/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017`。环境固定：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017
export TMPDIR=/home/lzf/ldx/tmp
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
```

默认CPU来源/资源预检，无发布或模型加载：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.materialization.full_prepare \
  --config configs/framework/stage4d1_materialization.server.json
```

成功标志：`FULL_MATERIALIZATION_CPU_PREFLIGHT_PASS`。

本轮已执行且已消费的Pilot/全量命令，仅用于复现说明；原目录存在时入口拒绝重试，不要再次执行：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.materialization.full_prepare \
  --config configs/framework/stage4d1_materialization.server.json --run-pilot

/home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.materialization.full_prepare \
  --config configs/framework/stage4d1_materialization.server.json \
  --allow-full-h-text-materialization
```

分别写入新命名空间`/home/lzf/ldx/outputs/tbps-framework-stage4d1/text_pilot_v1/`与`materialized_full_v1/`。Pilot不能seal全量集合；Full只有全部6156及来源通过后输出`TEXT_FULL_MATERIALIZED`。进度为`TEXT_MATERIALIZATION_PROGRESS`。

独立只读审计（输出目录须全新；不会修改原Full产物）：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.materialization.full_audit \
  --config configs/framework/stage4d1_materialization.server.json \
  --output /home/lzf/ldx/outputs/tbps-framework-stage4d1/user_readonly_audit_v1
```

成功标志：`TEXT_FULL_CPU_AUDIT_PASS`。本轮Codex审计使用另一个`full_audit_v1`目录。CPU测试：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/framework tests/test_dataset_adapter.py tests/test_dynamic_topk.py \
  tests/test_generator_framework.py tests/test_joint_baseline.py
```

进度只读查看：

```bash
tail -n 20 /home/lzf/ldx/tmp/stage4d1_text_full_v1.log
```

运行中Ctrl-C保留已提交单位及Claim/pending，最终FAIL记录若能发布会保留。SIGKILL/断电可能没有最终回执，不能因此重跑或删除Claim。Full入口不提供自动重试/恢复；需先审查现场，再单独授权。原子单元在异常之后仍可只读核验，禁止覆盖。

## Stage4D-2最小受控方案（未授权、未执行）

1. 固定本轮Plan及producer路径，复用已完成H文本集合；不能在新工作树重新计算Plan然后改签旧H单元。
2. 新受控GPU入口必须严格G运行闭包CPU预检、来源/历史保护、磁盘/inode和实际可用RAM检查，并单独冻结GPU空闲显存/时限策略；当前CLI没有GPU生成选项。
3. 同一Plan/Store调用`stream_images`，显式传入`allow_real_generation=True`与资源检查。工厂仅进入一次`load_legacy_plugins(...,components=("gv2",),device="cuda:0")`；不加载H/IRRA/RaSa/APTM。
4. 每图先Claim→一次G→严格相对原图8/255检查→保存原RGB/Adv/effective_delta→独立读回与证明；不缓存全量Tensor。
5. 已完成图严格核验后跳过；未提交前向、pending或损坏在加载前拒绝。禁止自动重复已消费前向。
6. 全3074有序图和生成证明通过后才seal Image Collection；最后Joint仅引用既有图文集合，不重新生成。不得据此宣称跨模型效果。

源模型SHA及预算固定不变；资源阈值留待Stage4D-2授权冻结。本阶段不提供可误触发真实G的执行命令。
