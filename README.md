# TBPS 文本属性提取（第一版）

固定的 13 个槽位及候选值以 [`ontology/upar_attribute_space.yaml`](ontology/upar_attribute_space.yaml) 为准；[`ontology/alias_map.yaml`](ontology/alias_map.yaml) 只维护文本别名。未提及、冲突或无法可靠绑定的值为字符串 `"null"`；明确否定为 `"no"` 或 `"no_glasses"`。

安装后可直接调用：

```powershell
python -m pip install -e ".[dev]"
```

```python
from attributes import extract

attributes = extract("a man wearing a white shirt and black pants")
```

如需记录属性在原始 caption 中的文本与字符位置，可调用 `extract_with_provenance(caption)`。它返回 `attributes` 和包含 13 个槽位的 `provenance`；最终值为 `"null"` 的槽位对应 `null`，其余槽位包含 `canonical` 与 `mentions`。每个 mention 的 `start`、`end` 满足 `caption[start:end] == raw`，衣物颜色和长度还包含 `linked_object`。

```python
from attributes import extract_with_provenance

result = extract_with_provenance("A woman in a navy blue jacket.")
```

通用批处理接受 JSON 数组或 JSONL。每条记录可为 caption 字符串，或 `{ "id": "...", "caption": "..." }`；输出包含 `id`、`caption`、`attributes`。

```powershell
python scripts/batch_extract.py --input captions.jsonl --output attributes.jsonl
python scripts/batch_extract.py --input captions.jsonl --output attributes_with_spans.jsonl --with-provenance
python -m pytest
```

核心提取器只接收 caption 字符串。未来各数据集的 annotation 格式可分别实现 `CaptionAdapter`，转换为 `CaptionRecord` 后复用提取器。

## 在服务器上试跑 CUHK-PEDES

CUHK-PEDES 的原始 `reid_raw.json` 是图像记录数组，每条含 `split`、`captions`、`file_path`、`id`。格式见[原项目说明](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description#data-preparation)。CUHK adapter 将每条 `captions` 展开为独立记录，输出 ID 使用 `file_path#caption_index`。这里只读取标注，不需要图像、GPU 或模型权重。

```bash
git clone https://github.com/w-harda/idea-test.git
cd idea-test
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q

# 改成服务器上 reid_raw.json 的实际绝对路径
CUHK_JSON=/absolute/path/to/reid_raw.json
mkdir -p outputs
python scripts/batch_extract.py --adapter cuhk --split train --limit 20 \
  --input "$CUHK_JSON" --output outputs/cuhk_train_smoke.jsonl
head -n 2 outputs/cuhk_train_smoke.jsonl

python scripts/batch_extract.py --adapter cuhk --split train \
  --input "$CUHK_JSON" --output outputs/cuhk_train.jsonl
wc -l outputs/cuhk_train.jsonl
```

去掉 `--split train` 可处理标注中的全部 split。若使用已经展平的 caption JSON/JSONL，请使用默认的 `--adapter flat`。

## 在服务器上处理 ICFG-PEDES 与 RSTPReid

两种 adapter 都会把一张图像的 `captions` 逐条展开，输出 ID 分别为 `file_path#caption_index` 和 `img_path#caption_index`。`--split` 按标注中的原始名称筛选；省略时处理全部 split。可先加 `--limit 20` 抽查结果。

```bash
mkdir -p outputs
ICFG_JSON=/home/lzf/TBPS/Datasets/ICFG-PEDES/ICFG-PEDES.json
python scripts/batch_extract.py --adapter icfg --split train \
  --input "$ICFG_JSON" --output outputs/icfg_train.jsonl

RSTP_JSON=/home/lzf/TBPS/Datasets/RSTPReid/data_captions.json
python scripts/batch_extract.py --adapter rstp --split train \
  --input "$RSTP_JSON" --output outputs/rstp_train.jsonl
```
# 图像侧属性提取

图像推理入口为 `scripts/batch_extract_image.py`。输入只使用图片与 annotation 中的图像路径、`split`；同一相对路径只推理一次。输出 JSONL 含 13 个 canonical 槽位与按[官方测试阶段推理代码](https://github.com/caodoanh2001/upar_challenge/blob/main/infer_upar_test_phase.py)顺序排列的 UPAR40 概率。`upper_clothing_type` 恒为 `null`；低置信度、候选冲突及 `Other` 颜色会弃判。

模型结构来自仓库外的[官方 C2T-Net 源码](https://github.com/caodoanh2001/upar_challenge)。`--upar-source` 指向其源码目录，`--checkpoint` 指向官方 `best_model.pth`。本仓库不保存模型源码或权重。模型加载时跳过上游的额外 Swin/EVA 预训练下载，并严格加载最终 checkpoint；若权重结构与源码不符会直接报错。图像预处理采用[官方验证阶段的设置](https://github.com/caodoanh2001/upar_challenge/blob/main/dataset/augmentation.py)：缩放到 256×128、转 RGB tensor、ImageNet 均值和标准差归一化。

在已安装 CUDA 版 PyTorch 与 torchvision 的独立环境中安装其余推理依赖：

```bash
python -m pip install -e '.[image]'
```

单张图像：

```bash
python scripts/batch_extract_image.py \
  --image /path/to/person.jpg \
  --upar-source /path/to/upar_challenge \
  --checkpoint /path/to/best_model.pth \
  --output outputs/single.jsonl
```

数据集抽样示例（`--limit` 计数的是不同图片）：

```bash
python scripts/batch_extract_image.py \
  --dataset cuhk --annotation /path/to/reid_raw.json \
  --image-root /path/to/CUHK-PEDES/imgs --split test --limit 20 \
  --upar-source /path/to/upar_challenge \
  --checkpoint /path/to/best_model.pth \
  --output outputs/cuhk-test-20.jsonl
```

ICFG-PEDES 使用 `--dataset icfg` 与其 `ICFG-PEDES.json`；RSTPReid 使用 `--dataset rstp` 与其 `data_captions.json`。`--image-root` 应是 annotation 图像相对路径的起点，实际路径需按服务器数据布局填写。初始 batch 大小为 16，可用 `--batch-size` 调整。

### 服务器全量后台提取

三个数据集各 20 张图像的 GPU smoke test 通过后，可以在服务器运行：

```bash
nohup bash /home/lzf/ldx/projects/idea-TBPS-test1/scripts/run_full_image_extraction.sh \
  > /home/lzf/ldx/outputs/idea-TBPS-test1/upar/full-run.log 2>&1 < /dev/null &
```

关闭 SSH 或本地电脑不会中断 `nohup` 任务。脚本按 CUHK、ICFG、RSTP 顺序处理，每张不同图片只推理一次，默认 batch 大小为 2。结果分别保存在 `/home/lzf/ldx/outputs/idea-TBPS-test1/upar/full/{cuhk,icfg,rstp}.jsonl`；进度与错误写入 `full-run.log`，每 100 张新图像报告一次。可用 `tail -f /home/lzf/ldx/outputs/idea-TBPS-test1/upar/full-run.log` 查看进度。

再次运行同一脚本会从已有 JSONL 续跑，跳过已写入的图像。已有结果若包含损坏或重复的图像记录，脚本会停止并报出行号，以免静默混入错误输出。数据集、checkpoint 和实验输出均在 Git 仓库外。

## 图像侧视觉属性质量验收

运行 `scripts/visual_quality_audit.py` 可从已经生成的 CUHK、ICFG、RSTP 图像结果中各选约 100 张，生成独立人工验收材料。它只读取图像结果及原图，不读取 Caption、文本属性或 provenance，也不调用或修改 UPAR 模型。示例（`audit-v1` 已生成；再次抽样须换一个新目录，避免覆盖人工标注）：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python \
  /home/lzf/ldx/projects/idea-TBPS-test1/scripts/visual_quality_audit.py sample \
  --results-dir /home/lzf/ldx/outputs/idea-TBPS-test1/upar/full \
  --cuhk-image-root /home/lzf/TBPS/Datasets/CUHK-PEDES/imgs \
  --icfg-image-root /home/lzf/TBPS/Datasets/ICFG-PEDES/imgs \
  --rstp-image-root /home/lzf/TBPS/Datasets/RSTPReid/imgs \
  --output-dir /home/lzf/ldx/outputs/idea-TBPS-test1/upar/audit-v2 \
  --per-dataset 100 --seed 20260925
```

抽样按每张图像的 12 个可评估槽位的平均审核分数分成低、中、高三个等人数层；每层再优先选择尚未充分覆盖的「槽位 × 预测值 × 槽位分数段」组合。因此样本覆盖常见和罕见取值、弃判以及分歧边界，不是纯随机样本。槽位审核分数是 UPAR 独立 sigmoid 概率的启发式决策强度：二元槽位取 `max(p, 1-p)`；多候选槽位取 `(最高概率 + 1 - 次高概率) / 2`。它**不是校准后的正确概率**。槽位分段为低于 0.70、0.70 至低于 0.85、至少 0.85；图像级低/中/高是本数据集平均分数的三等分。

输出目录包含：

- `manifest.jsonl`：每张图的 `sample_id`、数据集、相对及绝对图像路径、13 槽位预测、12 槽位审核分数与分段、完整 UPAR40 概率。
- `cuhk.html`、`icfg.html`、`rstp.html`：嵌入缩略图的离线检查页。可用 Xftp 下载 HTML 后直接在浏览器打开；预测与概率默认折叠，以便先独立看图。
- `labels.csv`：人工真值模板，使用 `sample_id` 对齐。只填写 `truth_<slot>` 列，合法值见 YAML 或 HTML 页。空白表示未标注，字符串 `null` 表示仅凭图像无法判断；明确不存在应填 `no` 或 `no_glasses`。可填写 `notes`。保留所有行及 `sample_id`、`dataset`、`image` 列。CSV 使用 UTF-8 BOM，便于表格软件读取。
- `selection_summary.json`：各数据集的抽样数量、分层数量及覆盖的预测值。

请只依据图像标注真值，不借助 Caption。当前 `upper_clothing_type` 预测固定为 `null`；模板允许预留真值，但本轮评估始终跳过此槽位。

填写并保存 `labels.csv` 后运行：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python \
  /home/lzf/ldx/projects/idea-TBPS-test1/scripts/visual_quality_audit.py evaluate \
  --manifest /home/lzf/ldx/outputs/idea-TBPS-test1/upar/audit-v1/manifest.jsonl \
  --labels /home/lzf/ldx/outputs/idea-TBPS-test1/upar/audit-v1/labels.csv \
  --output /home/lzf/ldx/outputs/idea-TBPS-test1/upar/audit-v1/report.json
```

`report.json` 分别报告三个数据集及合计的每槽位指标，并包含按槽位审核分数段统计的准确率。真值为空的单元格不参与统计；人工真值为 `null` 时记入单独计数，不参与准确率和覆盖率。对于有明确真值的单元格，`accuracy` 是非 `null` 预测中的正确比例，`overall_accuracy` 把模型弃判计作未答对，`non_null_coverage` 和 `null_ratio` 分别是非 `null` 与 `null` 预测比例。分段准确率也只对该段内有明确真值且模型非 `null` 的预测计算。由于验收样本刻意分层，指标用于定位问题，不能直接当作全数据集无偏准确率。

## 属性匹配与判别力计算（Stage 03）

对原始标注中的 train、val、test 全部 caption 计算属性判别力。命令必须同时指定数据集、原始标注和该数据集的全量图像属性图库：

```bash
/home/lzf/ldx/envs/tbps-text/bin/python scripts/score_attributes.py \
  --dataset cuhk \
  --annotation /home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json \
  --gallery /home/lzf/ldx/outputs/idea-TBPS-test1/upar/full/cuhk.jsonl \
  --output /home/lzf/ldx/outputs/idea-TBPS-test1/stage03/cuhk_all_scores.jsonl
```

ICFG 使用 --dataset icfg、ICFG-PEDES.json 和 icfg.jsonl；RSTP 使用 --dataset rstp、data_captions.json 和 rstp.jsonl。可用 --limit 20 抽查前 20 条 caption。运行前会比较原始标注和图库的完整图像路径集合；不一致时拒绝输出，防止跨数据集混合。图库始终包含所选数据集的全部图片，不按 train、val、test 分割。

每条 JSONL 记录保留 dataset、split、稳定 row_id、原始 annotation_row_index、caption_index、原 id、image、原始 caption、完整 13 槽 attributes、Stage 01 原样输出的 13 槽 provenance，以及 shared_attributes、gallery_count、valid_gallery_count、excluded_gallery_count、candidate_count 和 scores。ICFG 的原 id 可重复，应以 row_id 唯一定位。scores[slot] = S(a_i)，等于删除该属性后新进入候选集的图片数。upper_clothing_type 当前不参与匹配与评分。输出文件属于实验产物，不提交 Git。

Stage 04 的 `select_dynamic_topk.py` 保留 Stage 03 整条记录，因此 `provenance` 会原样进入 Top-K JSONL。Stage 05 的 `run_attack()` 直接读取该字段定位所选属性，并校验原始文本位置；输入的非零轮记录必须包含有效的 `provenance`。

# Stage 06：生成器训练总框架

训练依赖可在含 CUDA 版 PyTorch 的项目环境中安装：`pip install -e '.[train,dev]'`。`FrozenCLIP` 从调用方指定的本地 OpenAI CLIP 权重加载 source，冻结参数，并对 `[0,1]` RGB Tensor 做可微的 resize、中心裁剪和 CLIP 归一化。权重不随仓库分发，也不会由本模块自动下载。

- `attributes.cuhk_training.load_cuhk_training_index(annotation_path, stage04_path)` 从原始 CUHK-PEDES 标注与 Stage 04 JSONL 关联训练图片、查询和 ID；ID 仅供损失使用。
- `attributes.rank_objective` 提供 `soft_first_hit_rank`、`query_utility`、`rank_loss`、`exact_first_hit_rank` 和 `calibrate_tau`。`tau` 应从训练集干净 CLIP 分数差校准并在训练期间固定。
- `attributes.generator_framework.prepare_query` 通过 Stage 05 校验属性顺序和原文位置映射，交给生成器的是不含 ID、配对图片和 split 的 `QueryInput`。
- `GeneratorTrainer.train_step(TrainingBatch(...))` 先更新图像生成器，再用已扰动图库的候选期望最终排名更新文本生成器。图库子集必须包含查询 ID 的全部图片，并在干净、扰动计算中保持相同成员。
- `infer_gallery` 对各图库图片独立生成并投影到相对原图 `L∞ ≤ 8/255`；`infer_queries` 只调用冻结文本生成器直接选择。`k_star=0` 的文本保持原样。二者不调用 source 或 victim 打分。

具体 `G_I`、`G_T` 与合法字符候选规则须由调用方实现。`G_I` 仅接收单张图像 Tensor；`G_T.distribution(QueryInput)` 返回完整最终文本编辑选项及 logits，`G_T.select(QueryInput)` 返回测试时直接选择的编辑。每个编辑由 `CharacterEdit(slot, offset, replacement)` 表示，框架按 Stage 04 的有序属性与位置映射校验：只在选中属性词内，每个属性最多替换一个字符，允许跳过。

## 第一版属性引导联合攻击 baseline

此 baseline 读取 Stage 04 记录，原样调用 Stage 05 `run_attack`，逐个 `selected_slots` 固定执行 Image → Text。图像回调使用 [TTA 官方仓库](https://github.com/YanGGGL/Transform_to_Transfer_Attack) 的 `attacker_TTA.py:Attack.img_attack`（已核查提交 `fe4f1ece1718d475736d25e14a27b4d430e8f1b8`，文件 SHA-256 固定在适配层），接到现有 Frozen CLIP source。每轮同时用当前完整 caption 和由 M 定位的当前属性词作为官方接口支持的一图多文本监督。官方 `TTAttacker.attack` 还调用其词语替换并执行第二次图像攻击，不符合当前 Stage 05 顺序，故此版只装载原样 `Attack.img_attack`，不复制官方项目。官方模块导入时会读取 GloVe，适配层只编译官方文件中的 `Attack` 和图像路径辅助定义，图像算法仍运行官方实现。需要在仓库外提供该版本的 TTA 源码；图像环境需要 `kornia==0.7.3`。默认图像超参数与官方评测脚本一致：每属性 10 步、每尺度 6 个变换、尺度 `0.5,0.75,1.25,1.5`、步长 `2/255`；每轮均相对同一原图投影到累计 `L∞≤8/255`。

文本回调直接读取该轮 `provenance` 中的属性词 mention，只允许一个字符替换或跳过；候选是 [Unicode UTS #39 confusables](https://www.unicode.org/Public/security/latest/confusables.txt) 中与对应拉丁字母直接映射的单字符形近字。替换保持 caption 长度不变，并更新后续轮次可能重叠的 mention 原文切片。开发阶段的候选选择会使用 Frozen CLIP、当前受攻击图库和已知 ID 计算 soft first-hit rank；这属于开发期候选 oracle，不是冻结模型测试或 victim 评测。

小规模真实样本入口：

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1
git clone https://github.com/YanGGGL/Transform_to_Transfer_Attack.git /home/lzf/ldx/external/Transform_to_Transfer_Attack
git -C /home/lzf/ldx/external/Transform_to_Transfer_Attack checkout fe4f1ece1718d475736d25e14a27b4d430e8f1b8
/home/lzf/ldx/envs/tbps-image/bin/python -m pip install -e '.[attack]'
PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python scripts/run_joint_baseline_poc.py \
  --annotation /home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json \
  --stage04 /home/lzf/ldx/outputs/idea-TBPS-test1/stage04/cuhk_all_topk.jsonl \
  --image-root /home/lzf/TBPS/Datasets/CUHK-PEDES/imgs \
  --checkpoint /home/lzf/ldx/cache/clip/ViT-B-16.pt \
  --tta-root /home/lzf/ldx/external/Transform_to_Transfer_Attack \
  --output /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-official-params.json \
  --queries 3 --negative-images 20
```

PoC 只在 CUHK train 记录上做配对图像诊断：为当前 query 攻击其配对图库图片，其余图库图片保持干净；图库保留这几条 query 的全部同 ID 图片，再采 20 张错误 ID 图片。输入图像先按 CLIP 的 resize/center-crop 进入 224×224 攻击空间；`8/255` 约束相对该空间的干净图像。输出 JSON 逐条保存 clean、TTA-only、Text-only、TTA+Text 的真实首次正确 ID 名次、soft rank、累计扰动、字符编辑和固定轮次。该诊断使用配对关系和 ID，结果不能当作冻结图库一次生成或未知测试 query 的迁移攻击成绩；Stage 06 的生成器框架和其测试信息边界保持独立。

已跑通的 3 条真实 query / 29 张图库小样本，官方 10 步、6 个变换、四个附加尺度参数：平均真实名次增量分别为 TTA-only `+0.67`、Text-only `+1.00`、联合 `+1.33`；单条结果存在联合弱于纯文本的情况。这只验证链路与可比较输出，不据此得出稳健的效果结论。


### 同样三条样本的 Vanilla TTA 图像对照

`scripts/run_vanilla_tta_poc.py` 仅从 CUHK 原始 `reid_raw.json` 重建固定的 `cuhk:5:1`、`cuhk:11:0`、`cuhk:20:0` 与同一 29 张图库；图库路径和 ID 顺序按固定 SHA-256 校验。Vanilla 攻击函数只接受干净图片和完整原始 caption，不读取 Stage 04 的 A*、评分或 provenance，也不调用 Stage 05。它按官方 `TTAttacker.attack` 的两个图像阶段各调用一次原样 `Attack.img_attack`，各 10 步、6 个变换、四个附加尺度，沿用第一次的动量和相对原图 `8/255` 预算。为满足本次纯图像对照，省略官方流程中的文本词替换阶段，两次均使用原始 caption；因此这里的 Vanilla TTA 明确指**不改文本的官方图像攻击两阶段基线**，并非官方完整图文联合攻击。

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1
PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python scripts/run_vanilla_tta_poc.py \
  --annotation /home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json \
  --image-root /home/lzf/TBPS/Datasets/CUHK-PEDES/imgs \
  --checkpoint /home/lzf/ldx/cache/clip/ViT-B-16.pt \
  --tta-root /home/lzf/ldx/external/Transform_to_Transfer_Attack \
  --output /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-vanilla-tta.json
/home/lzf/ldx/envs/tbps-image/bin/python scripts/compare_tta_baselines.py \
  --attribute-report /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-official-params.json \
  --vanilla-report /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-vanilla-tta.json \
  --output /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-five-way.json
```

原报告的 `tta_only` 实际是 **Attribute-guided TTA-only**：每轮从 A* 取属性，经 Stage 05 同时把 caption 和属性词送入图像攻击。五项平均首次正确 ID 名次为 Clean `2.33`、Vanilla TTA `3.00`、Attribute-guided TTA-only `3.00`、Text-only `3.33`、Attribute-guided TTA + Text `3.67`。Vanilla 与属性引导 TTA-only 的三个离散名次相同，但对抗图不相同；在 `cuhk:20:0` 上两者像素最大差约 `0.06275`，配对图与原 caption 的相似度分别约 `-0.0319` 和 `-0.0009`。这组 3 条样本不足以判断属性引导的稳定增益。

### 同样三条样本的 Vanilla TTA 完整图文对照

`scripts/run_full_tta_poc.py` 从原始 CUHK 标注独立重建上述固定 3 条 query 和同一 29 张图库，不读取 Stage 04 记录、A*、S(a_i)、Dynamic Top-K 或 provenance，也不调用 Stage 05。它以完整原始 caption 和配对图像调用官方 `TTAttacker.attack`，保留 **Image_1（10 步）→ Text_1（1 步）→ Image_2（10 步）**。文本攻击采用官方 GloVe 近邻与 BERT masked LM 候选机制，允许词级改写；它与本项目的属性词内单字符替换是不同攻击。图像侧使用四个附加尺度、每尺度 6 个变换、`2/255` 步长，并将最终图像相对原图投影到 `L∞≤8/255`。

装载器按 SHA-256 校验官方 `attacker_TTA.py` 后只编译完整流程所需定义，避免执行上游模块顶层的硬编码 GloVe 加载。上游自带 BERT tokenizer 与当前 Transformers 不兼容，此处使用 `transformers==4.44.2` 的 `BertTokenizer`；CLIP 文本桥按上游方式先解码 BERT IDs 再用 Frozen CLIP tokenizer 编码。此处 GloVe 使用 [fse/glove-wiki-gigaword-300](https://huggingface.co/fse/glove-wiki-gigaword-300) 提供的 Stanford Wikipedia+Gigaword 6B/300d 向量；官方 Google Drive 的 `glove2word2vev300d.model` 无法从服务器获取，因此无法证明两个 Gensim 文件逐字节相同。BERT 使用 `bert-base-uncased`。本机资源 SHA-256：BERT `model.safetensors` 为 `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`，GloVe `*.model` 为 `05e50b69f0722ca06b91edba82e043b0e35e1af81c6d94a2e6d90c4d674f2c9c`、`*.vectors.npy` 为 `20dfb1f44719e2d934bfee5d39a6ffb4f248bae2a00a0d59f953ab7d0a39c879`。模型与实验输出均不提交 Git。

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1
/home/lzf/ldx/envs/tbps-image/bin/python -m pip install -e '.[attack,attack-full]'
HF_HOME=/home/lzf/ldx/cache/huggingface PYTHONPATH=src \
  /home/lzf/ldx/envs/tbps-image/bin/python scripts/run_full_tta_poc.py \
  --annotation /home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json \
  --image-root /home/lzf/TBPS/Datasets/CUHK-PEDES/imgs \
  --checkpoint /home/lzf/ldx/cache/clip/ViT-B-16.pt \
  --tta-root /home/lzf/ldx/external/Transform_to_Transfer_Attack \
  --bert /home/lzf/ldx/cache/tta/bert-base-uncased \
  --glove /home/lzf/ldx/cache/tta/glove-wiki-gigaword-300/glove-wiki-gigaword-300.model \
  --output /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-full-tta.json
/home/lzf/ldx/envs/tbps-image/bin/python scripts/compare_tta_baselines.py \
  --attribute-report /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-official-params.json \
  --vanilla-report /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-vanilla-tta.json \
  --full-report /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-full-tta.json \
  --output /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-six-way.json
```

六项首次正确 ID 名次：

| query | Clean | Vanilla TTA (image-only) | Vanilla TTA (full) | Attribute-guided TTA-only | Text-only | Attribute-guided TTA + Text |
|---|---:|---:|---:|---:|---:|---:|
| `cuhk:5:1` | 1 | 1 | 1 | 1 | 1 | 1 |
| `cuhk:11:0` | 5 | 5 | 4 | 5 | 8 | 7 |
| `cuhk:20:0` | 1 | 3 | 5 | 3 | 1 | 3 |
| 平均 | 2.33 | 3.00 | 3.33 | 3.00 | 3.33 | 3.67 |

完整 Vanilla TTA 三条均发生官方词级文本替换，最终配对图像的最大扰动均为约 `8/255`。该表只描述固定 3 条 train query、29 张图库的配对图像开发诊断；样本太少，不能据此推断攻击方法的整体效果。原报告中的 `tta_only` 是 **Attribute-guided TTA-only**，并非 Vanilla。

## 同批样本补充 AP-Attack 对照

AP-Attack 来自[官方仓库](https://github.com/yuanbianGit/AP-Attack)及其 [ICCV 2025 论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Bian_Prompt-driven_Transferable_Adversarial_Attack_on_Person_Re-Identification_with_Attribute-aware_Textual_ICCV_2025_paper.pdf)。服务器已有本地复现仓库 `/home/lzf/ldx/projects/AP-Attack`（提交 `db0aac946828ea4a15f2a31d349ccc7303855d4a`）和 DukeMTMC-reID 的 stage2 ReID+attribute-semantic 10/10 生成器权重 `/home/lzf/ldx/outputs/AP-Attack/stage2_reid_semantic_10_10/best_G_V.pth.tar`，权重 SHA-256 为 `9dd2afe66afedc7ad17b43ebd14bfc2349e519bd74dbc5addacf5fad13434692`。生成器源码 `advers/GD.py` SHA-256 为 `a5b5a8a5f3df8bf4137b95ed6e98a1ee2667a6f2b51cbcb4f4c82759546c9342`。原论文中的属性语义机制用于训练，已冻结的 `Generator.forward(image)` 推理时**没有属性或 caption 输入**。该权重是本机按官方代码复现训练的产物，不是作者发布权重。

`AP-Attack (full)` 在独立运行模式下只从原始 CUHK 标注重建固定 3 条 query、29 图库，对每条配对图像调用一次上述完整权重的官方生成器，不读取 Stage 04 或 provenance，也不运行 Stage 05。图像按 AP 官方验证预处理缩放到 256×128、以 `[0.5,0.5,0.5]` 均值/标准差归一化；生成器输出的像素扰动截断到 `±8/255`。为与旧六项对照保持完全相同的 Frozen CLIP 224×224 干净图库，适配层把 AP 原生坐标扰动映射回原图坐标，再使用同一 CLIP resize/crop 映射扰动，加在旧评测的干净 CLIP 图像上；两处都检查累计 `L∞≤8/255`。这是跨 ReID→TBPS retrieval 的图像空间适配，不能当作 AP 原论文目标任务成绩。

`Attribute-guided AP-Attack-only` 和 `Attribute-guided AP-Attack + Text` 读取 Stage 04 有序 A*，经 Stage 05 每属性一轮。图像回调在当前 AP 原生图上逐轮重新调用同一个冻结生成器，每轮对同一原图投影，保证累计预算；**属性只决定调用次数和顺序，生成器并不接收当前属性，故这两组是属性调度 AP 生成器，并非属性条件化 AP 生成器**。前者文本不变；后者固定 Image → Text，并复用 M、现有 Unicode confusable 候选与开发期 source soft rank 选择，每属性最多改一个字符，可跳过。三组都只攻击当前 query 的配对图库图片，其他 28 张保持干净。

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1
PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python scripts/run_ap_attack_poc.py \
  --mode full \
  --annotation /home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json \
  --image-root /home/lzf/TBPS/Datasets/CUHK-PEDES/imgs \
  --checkpoint /home/lzf/ldx/cache/clip/ViT-B-16.pt \
  --ap-root /home/lzf/ldx/projects/AP-Attack \
  --generator /home/lzf/ldx/outputs/AP-Attack/stage2_reid_semantic_10_10/best_G_V.pth.tar \
  --output /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-ap-full.json
PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python scripts/run_ap_attack_poc.py \
  --mode guided \
  --annotation /home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json \
  --stage04 /home/lzf/ldx/outputs/idea-TBPS-test1/stage04/cuhk_all_topk.jsonl \
  --image-root /home/lzf/TBPS/Datasets/CUHK-PEDES/imgs \
  --checkpoint /home/lzf/ldx/cache/clip/ViT-B-16.pt \
  --ap-root /home/lzf/ldx/projects/AP-Attack \
  --generator /home/lzf/ldx/outputs/AP-Attack/stage2_reid_semantic_10_10/best_G_V.pth.tar \
  --output /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-ap-guided.json
/home/lzf/ldx/envs/tbps-image/bin/python scripts/compare_ap_baselines.py \
  --six-report /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-six-way.json \
  --ap-full-report /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-ap-full.json \
  --ap-guided-report /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-ap-guided.json \
  --output /home/lzf/ldx/outputs/idea-TBPS-test1/joint-baseline/poc-3-nine-way.json
```

固定三条 query 的首次正确 ID 名次（越大表示正确结果越靠后）：

| 方法 | `cuhk:5:1` | `cuhk:11:0` | `cuhk:20:0` | 平均 |
|---|---:|---:|---:|---:|
| Clean | 1 | 5 | 1 | 2.33 |
| Vanilla TTA (image-only) | 1 | 5 | 3 | 3.00 |
| Vanilla TTA (full) | 1 | 4 | 5 | 3.33 |
| AP-Attack (full) | 1 | 2 | 1 | 1.33 |
| Text-only | 1 | 8 | 1 | 3.33 |
| Attribute-guided TTA-only | 1 | 5 | 3 | 3.00 |
| Attribute-guided TTA + Text | 1 | 7 | 3 | 3.67 |
| Attribute-guided AP-Attack-only | 1 | 2 | 1 | 1.33 |
| Attribute-guided AP-Attack + Text | 1 | 2 | 1 | 1.33 |

AP+Text 三条分别选中 2、2、1 个字符替换；相较 AP-only，soft rank 变化而首次正确 ID 的整数名次未变。当前三条样本上的 AP 结果使平均名次低于 Clean，不能解读为攻击增益。所有方法共用有序图库指纹 `e1505afef38cb62aa5e22dacf9d60a4e62871243763669e4a50e0a492fd88788`；这只是同一 Frozen CLIP source 上的配对图像开发诊断，不代表原论文的图像 ReID 性能、未知查询攻击或完整图库迁移效果。
