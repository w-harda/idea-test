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

每条 JSONL 记录保留 dataset、split、稳定 row_id、原始 annotation_row_index、caption_index、原 id、image、原始 caption、完整 13 槽 attributes，以及 shared_attributes、gallery_count、valid_gallery_count、excluded_gallery_count、candidate_count 和 scores。ICFG 的原 id 可重复，应以 row_id 唯一定位。scores[slot] = S(a_i)，等于删除该属性后新进入候选集的图片数。upper_clothing_type 当前不参与匹配与评分。输出文件属于实验产物，不提交 Git。
