# 阶段 01：文本属性提取（已冻结）

## 目标与最终方案

将 TBPS caption 映射为固定的 13 个单值 canonical 属性，并可追踪最终属性在原始 caption 中的文本位置。实现采用 `ontology/alias_map.yaml`、规则短语匹配及局部衣物实体绑定；提取器与数据集标注格式分离，不调用 LLM 或第三方 NLP 模型。本阶段只处理文本侧，不负责图像属性、跨模态匹配、重要性评分或攻击。

## 13-slot canonical space

权威定义为 `ontology/upar_attribute_space.yaml`。每个槽位都允许字符串值 `"null"`；它表示未提及、无法可靠判定或冲突，不等于明确的 `no`。两种衣物颜色共用颜色集合 `C = {black, blue, brown, green, grey, orange, pink, purple, red, white, yellow}`。

| 槽位 key | 非 `null` canonical 值 |
| --- | --- |
| `age` | `young`, `adult`, `elderly` |
| `gender` | `male`, `female` |
| `hair_length` | `short`, `long`, `bald` |
| `upper_clothing_length` | `short`, `long` |
| `upper_clothing_color` | `C` |
| `upper_clothing_type` | `t_shirt_shirt`, `hoodie_sweater`, `jacket_coat`, `vest_sleeveless` |
| `lower_clothing_length` | `short`, `long` |
| `lower_clothing_color` | `C` |
| `lower_clothing_type` | `trousers_shorts`, `skirt_dress` |
| `backpack` | `yes`, `no` |
| `bag` | `yes`, `no` |
| `glasses` | `normal_glasses`, `sunglasses`, `no_glasses` |
| `hat` | `yes`, `no` |

## 源码与公开接口

- `src/attributes/text_extractor.py`：`TextAttributeExtractor`、`extract(caption: str) -> dict[str, str]`、`extract_with_provenance(caption: str) -> dict`；包入口 `src/attributes/__init__.py` 导出这三个接口。
- `ontology/alias_map.yaml`：原始短语到上述 canonical 值的映射；不得把别名当作新的 canonical 值。
- `src/attributes/dataset_adapter.py`：`CaptionRecord(id, caption)` 与各数据集 adapter。`scripts/batch_extract.py` 负责 JSON/JSONL 批量输出，提取器本身不读取标注文件。

`extract()` 始终返回全部 13 个槽位。`extract_with_provenance()` 复用同一次属性提取，返回 `{"attributes": {...}, "provenance": {...}}`，其中 `attributes` 与 `extract()` 一致。批处理默认记录仍为 `id`、`caption`、`attributes`；仅传入 `--with-provenance` 时额外输出 `provenance`。

## Provenance / span 契约

`provenance` 同样包含 13 个槽位。最终属性为字符串 `"null"` 时，对应 provenance 是 JSON `null`，不提供可替换 span；否则为 `{"canonical": value, "mentions": [...]}`。每个 mention 至少有 `raw`、`start`、`end`，满足原始 Python 字符串 `caption[start:end] == raw`，保留原文大小写。重复出现且支持同一最终值的 mention 可分别记录。衣物颜色和长度 mention 附有 `linked_object: {raw, start, end}`，指向被选中的衣物实体；内搭不会取代外层衣物的 provenance。后续文本修改应先检查 `mentions` 是否非空，并再次验证切片。

## 数据集接入

| Adapter / CLI 值 | 标注输入 | 输出 ID |
| --- | --- | --- |
| `CuhkPedesAdapter` / `cuhk` | CUHK-PEDES `reid_raw.json` | `file_path#caption_index` |
| `IcfgPedesAdapter` / `icfg` | ICFG-PEDES caption JSON | `file_path#caption_index` |
| `RstpReidAdapter` / `rstp` | RSTPReid caption JSON | `img_path#caption_index` |
| `JsonCaptionAdapter` / `flat` | 通用 JSON 数组或 JSONL | 标注 `id`，缺省为序号 |

前三种 adapter 可按标注中的 `split` 筛选，并将每张图像的每条 caption 展开为独立记录；四种格式均调用同一个文本提取器。UFineBench adapter 尚未接入。

## 已验证结果与冻结边界

- 文本阶段最终提交 `e5da05b`：`155 passed`；与冻结属性实现比较 148 条代表性字符串，13-slot 结果差异为 0；285 个 mention 的原文切片校验全部通过。
- 阶段内已记录三个 `train` split 的批量处理规模：CUHK-PEDES 68,126、ICFG-PEDES 34,674、RSTPReid 37,010 条 caption。这些处理记录不代表 provenance 版本已在三个完整数据集上重新跑过。
- 当前分支仍使用同一套文本 ontology、alias、extractor、adapter 和批处理接口；未纳入本阶段后续图像模块的改动。
- 设计冻结：保持 13 槽位及候选值、单值冲突返回 `"null"`、明确否定与未知分离；仅提取明示信息，不从 `pants` 推断长度；颜色绑定衣物实体，携带的衣物不参与穿着属性，外层衣物优先；多颜色冲突仍返回 `"null"`。后续阶段直接消费接口，不应为个别 caption 随意改变规则。
- 归一化是有限空间内的固定映射，例如 `navy/teal → blue`、`medium/shoulder-length hair → long`、`young adult → adult`；`suit pants/trousers` 按下装处理，不额外生成 `suit` 上装。

## 关键提交

| Commit | 内容 |
| --- | --- |
| `22c18ea` | 13 槽位 ontology、初版文本提取与 CUHK adapter |
| `eb965e4` | ICFG-PEDES、RSTPReid adapter |
| `f5720c8` | 最后一轮文本规则歧义收口，属性逻辑冻结 |
| `e5da05b` | 原文 provenance/span、公开接口与批处理可选开关 |

## 后续模块的调用与剩余事项

```python
from attributes import extract, extract_with_provenance

attributes = extract(caption)
traced = extract_with_provenance(caption)  # traced["attributes"], traced["provenance"]
```

批处理可用 `python scripts/batch_extract.py --adapter cuhk --split train --input reid_raw.json --output captions.jsonl --with-provenance`；`icfg`、`rstp` 使用相同接口。后续模块应以 canonical key/value 处理属性，以 `linked_object` 和已校验的原文 span 定位文本；不要依据 `caption.find(raw)` 重新猜测位置。

已知边界：规则提取不做多人目标识别或复杂句法消歧；冲突属性没有可替换 span；否定配饰的 mention 是物品词，极性由 canonical 值表达，执行替换前需检查否定上下文；完整三数据集的 provenance 版全量复验及 UFineBench adapter 尚待后续阶段处理。
