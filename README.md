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
