"""Score every caption in one dataset against that dataset's complete image gallery."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attributes.attribute_scoring import AttributeGallery
from attributes.image_dataset_adapter import IMAGE_KEYS
from attributes.text_extractor import TextAttributeExtractor


DATASET_NAMES = {"cuhk": "CUHK-PEDES", "icfg": "ICFG-PEDES", "rstp": "RSTPReid"}


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            yield value


def normalize_image(value: object, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"annotation row {index}: missing image path")
    image = value.replace("\\", "/")
    relative = PurePosixPath(image)
    if (relative.is_absolute() or not relative.parts
            or any(part == ".." for part in relative.parts) or ":" in image):
        raise ValueError(f"annotation row {index}: invalid image path: {image}")
    return relative.as_posix()


def load_annotation(path: Path, dataset: str) -> tuple[list[dict], set[str]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("annotation must be a JSON array")
    image_key = IMAGE_KEYS[dataset]
    images = set()
    prepared = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"annotation row {index}: expected an object")
        image = normalize_image(row.get(image_key), index)
        split = row.get("split")
        captions = row.get("captions")
        if split not in ("train", "val", "test"):
            raise ValueError(f"annotation row {index}: unknown split: {split}")
        if not isinstance(captions, list) or not captions or any(
            not isinstance(caption, str) for caption in captions
        ):
            raise ValueError(f"annotation row {index}: invalid captions")
        images.add(image)
        prepared.append({
            "annotation_row_index": index,
            "image": image,
            "split": split,
            "captions": captions,
        })
    return prepared, images


def validate_gallery(annotation_images: set[str], gallery: AttributeGallery) -> None:
    missing = annotation_images - gallery.images
    extra = gallery.images - annotation_images
    if missing or extra:
        raise ValueError(
            "query annotation and gallery contain different image sets "
            f"(missing={len(missing)}, extra={len(extra)})"
        )


def score_dataset(
    dataset: str, annotation: Path, gallery_path: Path, output: Path,
    limit: int | None = None,
) -> int:
    rows, images = load_annotation(annotation, dataset)
    gallery = AttributeGallery(read_jsonl(gallery_path))
    validate_gallery(images, gallery)
    extractor = TextAttributeExtractor()
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            for caption_index, caption in enumerate(row["captions"]):
                traced = extractor.extract_with_provenance(caption)
                attributes = traced["attributes"]
                scored = gallery.score(attributes)
                result = {
                    "dataset": DATASET_NAMES[dataset],
                    "split": row["split"],
                    "row_id": f"{dataset}:{row['annotation_row_index']}:{caption_index}",
                    "annotation_row_index": row["annotation_row_index"],
                    "caption_index": caption_index,
                    "id": f"{row['image']}#{caption_index}",
                    "image": row["image"],
                    "caption": caption,
                    "attributes": attributes,
                    "provenance": traced["provenance"],
                    **scored,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
                if limit is not None and count >= limit:
                    return count
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute all Stage 03 attribute scores")
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_NAMES))
    parser.add_argument("--annotation", required=True, type=Path, help="raw dataset JSON")
    parser.add_argument("--gallery", required=True, type=Path, help="Stage 02 JSONL")
    parser.add_argument("--output", required=True, type=Path, help="score JSONL")
    parser.add_argument("--limit", type=int, help="first N captions for inspection")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.suffix.lower() != ".jsonl":
        parser.error("--output must end in .jsonl")
    if args.output.resolve() in (args.annotation.resolve(), args.gallery.resolve()):
        parser.error("output must differ from inputs")
    count = score_dataset(
        args.dataset, args.annotation, args.gallery, args.output, args.limit
    )
    print(f"{DATASET_NAMES[args.dataset]}: scored {count} captions")


if __name__ == "__main__":
    main()
