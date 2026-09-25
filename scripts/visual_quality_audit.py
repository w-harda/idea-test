"""Select image-only UPAR audit samples and evaluate independent human labels."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attributes.visual_canonicalizer import SLOTS, UPAR40_NAMES, VisualCanonicalizer

DATASETS = ("cuhk", "icfg", "rstp")
EVAL_SLOTS = tuple(slot for slot in SLOTS if slot != "upper_clothing_type")
BANDS = ("low", "medium", "high")
COLORS = ("Black", "Blue", "Brown", "Green", "Grey", "Orange",
          "Pink", "Purple", "Red", "White", "Yellow", "Other")


def ontology_values() -> dict[str, set[str]]:
    path = Path(__file__).resolve().parents[1] / "ontology" / "upar_attribute_space.yaml"
    with path.open(encoding="utf-8") as handle:
        space = yaml.safe_load(handle)
    values = {slot["key"]: {value["key"] for value in slot["values"]}
              for slot in space["slots"]}
    if tuple(values) != SLOTS:
        raise ValueError("Ontology slot order differs from VisualCanonicalizer")
    return values


def _choice_score(probabilities: dict[str, float], names: tuple[str, ...]) -> float:
    first, second = sorted((probabilities[name] for name in names), reverse=True)[:2]
    return (first + 1.0 - second) / 2.0


def review_confidence(probabilities: dict[str, float],
                      attributes: dict[str, str]) -> dict[str, float]:
    """Heuristic decision strength, not a calibrated correctness probability."""
    p = probabilities
    scores = {
        "age": _choice_score(p, ("Age-Young", "Age-Adult", "Age-Old")),
        "gender": max(p["Gender-Female"], 1 - p["Gender-Female"]),
        "hair_length": _choice_score(
            p, ("Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald")),
        "upper_clothing_length": max(
            p["UpperBody-Length-Short"], 1 - p["UpperBody-Length-Short"]),
        "lower_clothing_length": max(
            p["LowerBody-Length-Short"], 1 - p["LowerBody-Length-Short"]),
        "lower_clothing_type": _choice_score(
            p, ("LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress")),
    }
    for side, slot in (("UpperBody", "upper"), ("LowerBody", "lower")):
        names = tuple(f"{side}-Color-{color}" for color in COLORS)
        scores[f"{slot}_clothing_color"] = _choice_score(p, names)
    for slot, name in (("backpack", "Backpack"), ("bag", "Bag"), ("hat", "Hat")):
        value = p[f"Accessory-{name}"]
        scores[slot] = max(value, 1 - value)
    normal = p["Accessory-Glasses-Normal"]
    sun = p["Accessory-Glasses-Sun"]
    scores["glasses"] = (1 - max(normal, sun) if attributes["glasses"] == "no_glasses"
                         else _choice_score(
                             p, ("Accessory-Glasses-Normal", "Accessory-Glasses-Sun")))
    return {slot: round(scores[slot], 6) for slot in EVAL_SLOTS}


def confidence_band(score: float) -> str:
    return "high" if score >= 0.85 else ("medium" if score >= 0.70 else "low")


def iter_results(path: Path):
    canonicalizer = VisualCanonicalizer()
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty line")
            row = json.loads(line)
            image = row.get("image")
            attributes = row.get("attributes")
            probabilities = row.get("upar40")
            if not isinstance(image, str) or not image or image in seen:
                raise ValueError(f"{path}:{line_number}: invalid or duplicated image")
            if not isinstance(attributes, dict) or tuple(attributes) != SLOTS:
                raise ValueError(f"{path}:{line_number}: expected 13 ordered slots")
            if not isinstance(probabilities, dict) or tuple(probabilities) != UPAR40_NAMES:
                raise ValueError(f"{path}:{line_number}: expected 40 ordered UPAR scores")
            if attributes["upper_clothing_type"] != "null":
                raise ValueError(f"{path}:{line_number}: upper_clothing_type must be null")
            actual = canonicalizer.canonicalize(
                [probabilities[name] for name in UPAR40_NAMES])
            if actual != attributes:
                raise ValueError(f"{path}:{line_number}: stored canonical mapping differs")
            seen.add(image)
            yield line_number, row


def resolve_image(root: Path, image: str) -> Path:
    relative = PurePosixPath(image.replace("\\", "/"))
    if (relative.is_absolute() or not relative.parts or
            any(part in ("..", ".") for part in relative.parts) or ":" in image):
        raise ValueError(f"Unsafe relative image path: {image}")
    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError(f"Image missing or outside root: {resolved}")
    return resolved


@dataclass(frozen=True)
class Candidate:
    line_number: int
    image: str
    features: tuple[tuple[str, str, str], ...]
    mean_confidence: float


def collect_candidates(path: Path):
    candidates: list[Candidate] = []
    buckets: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for line_number, row in iter_results(path):
        confidence = review_confidence(row["upar40"], row["attributes"])
        features = tuple(
            (slot, row["attributes"][slot], confidence_band(confidence[slot]))
            for slot in EVAL_SLOTS)
        index = len(candidates)
        candidates.append(Candidate(
            line_number, row["image"], features,
            sum(confidence.values()) / len(confidence)))
        for feature in features:
            buckets[feature].append(index)
    return candidates, buckets


def choose_candidates(candidates: list[Candidate],
                      buckets: dict[tuple[str, str, str], list[int]],
                      count: int, seed: int) -> list[tuple[int, str]]:
    if count < 1 or count > len(candidates):
        raise ValueError(f"Requested {count} samples from {len(candidates)} images")
    rng = random.Random(seed)
    ranked = sorted(range(len(candidates)),
                    key=lambda index: (candidates[index].mean_confidence,
                                       candidates[index].image))
    groups: dict[str, list[int]] = {band: [] for band in BANDS}
    image_band: dict[int, str] = {}
    for rank, index in enumerate(ranked):
        band = BANDS[min(2, 3 * rank // len(candidates))]
        groups[band].append(index)
        image_band[index] = band
    pool: set[int] = set()
    for feature in sorted(buckets):
        members = buckets[feature]
        pool.update(rng.sample(members, min(24, len(members))))
    for band in BANDS:
        members = groups[band]
        pool.update(rng.sample(members, min(max(250, count), len(members))))
    weights = {feature: 1 / math.sqrt(len(members))
               for feature, members in buckets.items()}
    usage: Counter[tuple[str, str, str]] = Counter()
    selected: list[tuple[int, str]] = []
    selected_indices: set[int] = set()
    quotas = {band: count // 3 + (index < count % 3)
              for index, band in enumerate(BANDS)}
    for band in BANDS:
        for _ in range(quotas[band]):
            available = [index for index in sorted(pool)
                         if index not in selected_indices and image_band[index] == band]
            if not available:
                raise ValueError(f"Not enough {band} candidates")
            best = max(available, key=lambda index: (
                sum(weights[feature] / (1 + usage[feature])
                    for feature in candidates[index].features),
                -index))
            selected.append((best, band))
            selected_indices.add(best)
            usage.update(candidates[best].features)
    return selected


def selected_rows(results_path: Path, image_root: Path,
                  candidates: list[Candidate],
                  selected: list[tuple[int, str]], dataset: str):
    by_line = {candidates[index].line_number: (index, band)
               for index, band in selected}
    rows: dict[int, dict] = {}
    with results_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number not in by_line:
                continue
            index, band = by_line[line_number]
            source = json.loads(line)
            confidence = review_confidence(source["upar40"], source["attributes"])
            rows[index] = {
                "dataset": dataset,
                "image": source["image"],
                "image_path": str(resolve_image(image_root, source["image"])),
                "sample_band": band,
                "attributes": source["attributes"],
                "review_confidence": confidence,
                "confidence_band": {slot: confidence_band(confidence[slot])
                                    for slot in EVAL_SLOTS},
                "upar40": source["upar40"],
            }
    return [rows[index] for index, _ in selected]


def thumbnail_data(path: Path) -> str:
    from PIL import Image, ImageOps

    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((180, 360))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=78)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def gallery_html(dataset: str, rows: list[dict], values: dict[str, set[str]]) -> str:
    esc = html.escape
    parts = ["""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>UPAR visual audit</title><style>
body{font:14px/1.45 system-ui,sans-serif;margin:24px;background:#f6f7f9;color:#202329}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(350px,1fr));gap:16px}
.card{background:white;border:1px solid #d9dce2;border-radius:10px;padding:14px}
img{max-width:180px;max-height:360px;display:block;margin:8px auto}
code{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:12px}
td,th{border-bottom:1px solid #e4e6eb;padding:3px 5px;text-align:left}
details{margin-top:8px}summary{cursor:pointer}
</style></head><body>""",
             f"<h1>{esc(dataset.upper())} 图像属性人工验收</h1>",
             "<p>先看图像并在 labels.csv 填写 truth_*。空白表示未标注；"
             "无法从图像判断时填 null。预测与概率在折叠区，减少标注时的提示偏差。"
             "upper_clothing_type 本阶段不参与评估。</p>",
             "<details><summary>合法 canonical 值</summary><ul>"]
    for slot in SLOTS:
        parts.append(f"<li><code>{esc(slot)}</code>: "
                     f"{esc(', '.join(sorted(values[slot])))}</li>")
    parts.append("</ul></details><div class='grid'>")
    for row in rows:
        image = thumbnail_data(Path(row["image_path"]))
        parts.append("<article class='card'>")
        parts.append(f"<h2>{esc(row['sample_id'])} · {esc(row['sample_band'])}</h2>")
        parts.append(f"<code>{esc(row['image'])}</code>")
        parts.append(f"<img alt='person' src='data:image/jpeg;base64,{image}'>")
        parts.append("<details><summary>13-slot 预测与审核分数</summary><table>"
                     "<tr><th>slot</th><th>prediction</th><th>score</th><th>band</th></tr>")
        for slot in SLOTS:
            score = row["review_confidence"].get(slot)
            score_text = "—" if score is None else f"{score:.3f}"
            band = row["confidence_band"].get(slot, "not evaluated")
            parts.append(f"<tr><td>{esc(slot)}</td>"
                         f"<td>{esc(row['attributes'][slot])}</td>"
                         f"<td>{score_text}</td><td>{esc(band)}</td></tr>")
        parts.append("</table></details><details><summary>UPAR40 原始概率</summary><table>"
                     "<tr><th>attribute</th><th>probability</th></tr>")
        for name, probability in row["upar40"].items():
            parts.append(f"<tr><td>{esc(name)}</td><td>{float(probability):.4f}</td></tr>")
        parts.append("</table></details></article>")
    parts.append("</div></body></html>")
    return "\n".join(parts)


def sample(args) -> None:
    values = ontology_values()
    output_dir = args.output_dir
    if output_dir.exists():
        raise ValueError(f"Output directory already exists; labels will not be overwritten: {output_dir}")
    roots = {"cuhk": args.cuhk_image_root, "icfg": args.icfg_image_root,
             "rstp": args.rstp_image_root}
    all_rows: list[dict] = []
    summary: dict[str, dict] = {}
    for dataset_index, dataset in enumerate(DATASETS):
        path = args.results_dir / f"{dataset}.jsonl"
        candidates, buckets = collect_candidates(path)
        selected = choose_candidates(
            candidates, buckets, args.per_dataset, args.seed + dataset_index)
        rows = selected_rows(path, roots[dataset], candidates, selected, dataset)
        for number, row in enumerate(rows, 1):
            row["sample_id"] = f"{dataset}-{number:03d}"
        all_rows.extend(rows)
        summary[dataset] = {
            "source_images": len(candidates),
            "selected_images": len(rows),
            "image_confidence_tertiles": dict(Counter(
                row["sample_band"] for row in rows)),
            "slot_confidence_bands": dict(Counter(
                band for row in rows for band in row["confidence_band"].values())),
            "slot_values": {slot: sorted({row["attributes"][slot] for row in rows})
                            for slot in EVAL_SLOTS},
        }
        print(f"{dataset}: {len(rows)} selected from {len(candidates)} images; "
              f"tertiles {summary[dataset]['image_confidence_tertiles']}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = ["sample_id", "dataset", "image", "image_path", "sample_band"]
    fields += [f"truth_{slot}" for slot in SLOTS]
    fields.append("notes")
    with (output_dir / "labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    for dataset in DATASETS:
        rows = [row for row in all_rows if row["dataset"] == dataset]
        (output_dir / f"{dataset}.html").write_text(
            gallery_html(dataset, rows, values), encoding="utf-8")
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created {output_dir} (manifest, labels, 3 offline galleries, summary)")


@dataclass
class Metrics:
    known: int = 0
    predicted: int = 0
    correct: int = 0
    truth_null: int = 0
    bands: dict[str, list[int]] = field(
        default_factory=lambda: {band: [0, 0, 0] for band in BANDS})

    def add(self, prediction: str, truth: str, band: str) -> None:
        if truth == "null":
            self.truth_null += 1
            return
        self.known += 1
        group = self.bands[band]
        group[0] += 1
        if prediction != "null":
            self.predicted += 1
            group[1] += 1
            if prediction == truth:
                self.correct += 1
                group[2] += 1

    def report(self) -> dict:
        def ratio(numerator: int, denominator: int):
            return numerator / denominator if denominator else None
        return {
            "known_truth_count": self.known,
            "truth_null_count": self.truth_null,
            "predicted_non_null_count": self.predicted,
            "correct_count": self.correct,
            "accuracy": ratio(self.correct, self.predicted),
            "overall_accuracy": ratio(self.correct, self.known),
            "non_null_coverage": ratio(self.predicted, self.known),
            "null_ratio": ratio(self.known - self.predicted, self.known),
            "confidence_bands": {
                band: {
                    "known_truth_count": group[0],
                    "predicted_non_null_count": group[1],
                    "correct_count": group[2],
                    "accuracy": ratio(group[2], group[1]),
                    "non_null_coverage": ratio(group[1], group[0]),
                }
                for band, group in self.bands.items()
            },
        }


def evaluate(args) -> None:
    values = ontology_values()
    manifest: dict[str, dict] = {}
    with args.manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in manifest or row["dataset"] not in DATASETS:
                raise ValueError(f"Invalid or duplicate manifest row {line_number}")
            if tuple(row["attributes"]) != SLOTS or tuple(row["upar40"]) != UPAR40_NAMES:
                raise ValueError(f"Invalid prediction schema at manifest row {line_number}")
            for slot in SLOTS:
                if row["attributes"][slot] not in values[slot]:
                    raise ValueError(f"Invalid {slot} prediction at manifest row {line_number}")
            manifest[sample_id] = row
    metrics = {dataset: {slot: Metrics() for slot in EVAL_SLOTS}
               for dataset in DATASETS}
    overall = {slot: Metrics() for slot in EVAL_SLOTS}
    seen: set[str] = set()
    annotated_rows = Counter()
    with args.labels.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "dataset", "image"} | {
            f"truth_{slot}" for slot in SLOTS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("labels.csv is missing required columns")
        for line_number, label in enumerate(reader, 2):
            sample_id = label["sample_id"]
            if sample_id not in manifest or sample_id in seen:
                raise ValueError(f"Unknown or duplicate sample_id at CSV line {line_number}")
            row = manifest[sample_id]
            if label["dataset"] != row["dataset"] or label["image"] != row["image"]:
                raise ValueError(f"Dataset or image changed at CSV line {line_number}")
            seen.add(sample_id)
            any_truth = False
            for slot in SLOTS:
                truth = (label[f"truth_{slot}"] or "").strip()
                if not truth:
                    continue
                if truth not in values[slot]:
                    raise ValueError(
                        f"Invalid truth_{slot}={truth!r} at CSV line {line_number}")
                any_truth = True
                if slot == "upper_clothing_type":
                    continue
                prediction = row["attributes"][slot]
                band = confidence_band(
                    review_confidence(row["upar40"], row["attributes"])[slot])
                metrics[row["dataset"]][slot].add(prediction, truth, band)
                overall[slot].add(prediction, truth, band)
            if any_truth:
                annotated_rows[row["dataset"]] += 1
    if seen != set(manifest):
        raise ValueError(f"Missing {len(set(manifest) - seen)} sample rows in labels.csv")
    if not any(item.known for item in overall.values()):
        raise ValueError("No concrete truth labels yet; fill truth_* columns first")
    report = {
        "metric_definition": {
            "accuracy": "correct / non-null predictions with concrete truth",
            "overall_accuracy": "correct / all concrete truth labels; null predictions count as incorrect",
            "non_null_coverage": "non-null predictions / concrete truth labels",
            "null_ratio": "null predictions / concrete truth labels",
            "truth_null": "human-unobservable; excluded from accuracy and coverage",
            "empty_truth": "not annotated; excluded",
            "confidence": "UPAR decision-strength heuristic, not calibrated probability",
            "confidence_bands": "low < 0.70; medium 0.70-<0.85; high >= 0.85",
            "upper_clothing_type": "excluded",
        },
        "datasets": {
            dataset: {
                "sample_count": sum(row["dataset"] == dataset for row in manifest.values()),
                "annotated_rows": annotated_rows[dataset],
                "overall": aggregate_metrics(metrics[dataset].values()).report(),
                "slots": {slot: metrics[dataset][slot].report() for slot in EVAL_SLOTS},
            }
            for dataset in DATASETS
        },
        "overall": {
            "sample_count": len(manifest),
            "annotated_rows": sum(annotated_rows.values()),
            "overall": aggregate_metrics(overall.values()).report(),
            "slots": {slot: overall[slot].report() for slot in EVAL_SLOTS},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for dataset, section in list(report["datasets"].items()) + [("all", report["overall"])]:
        result = section["overall"]
        accuracy = result["accuracy"]
        coverage = result["non_null_coverage"]
        print(f"{dataset}: annotated_rows={section['annotated_rows']}, "
              f"known_slots={result['known_truth_count']}, "
              f"accuracy={'N/A' if accuracy is None else f'{accuracy:.3f}'}, "
              f"coverage={'N/A' if coverage is None else f'{coverage:.3f}'}")
    print(f"Report: {args.output}")


def aggregate_metrics(items) -> Metrics:
    result = Metrics()
    for item in items:
        result.known += item.known
        result.predicted += item.predicted
        result.correct += item.correct
        result.truth_null += item.truth_null
        for band in BANDS:
            for index in range(3):
                result.bands[band][index] += item.bands[band][index]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="UPAR image-attribute quality audit")
    commands = parser.add_subparsers(dest="command", required=True)
    sample_parser = commands.add_parser("sample", help="select images and make labels.csv")
    sample_parser.add_argument("--results-dir", type=Path, required=True)
    for dataset in DATASETS:
        sample_parser.add_argument(f"--{dataset}-image-root", type=Path, required=True)
    sample_parser.add_argument("--output-dir", type=Path, required=True)
    sample_parser.add_argument("--per-dataset", type=int, default=100)
    sample_parser.add_argument("--seed", type=int, default=20260925)
    sample_parser.set_defaults(func=sample)
    eval_parser = commands.add_parser("evaluate", help="score filled human labels")
    eval_parser.add_argument("--manifest", type=Path, required=True)
    eval_parser.add_argument("--labels", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.set_defaults(func=evaluate)
    args = parser.parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
