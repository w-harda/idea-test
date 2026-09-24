import json
import sys
from pathlib import Path

import pytest

from scripts import batch_extract_image
from attributes.visual_canonicalizer import UPAR40_NAMES, VisualCanonicalizer


def test_resume_skips_existing_images_and_does_not_reload_completed_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    annotation = tmp_path / "images.json"
    annotation.write_text(json.dumps([
        {"file_path": "a.jpg", "captions": ["ignored"]},
        {"file_path": "a.jpg", "captions": ["also ignored"]},
        {"file_path": "b.jpg", "captions": ["ignored"]},
    ]), encoding="utf-8")
    scores = [0.1] * 40
    existing = {
        "image": "a.jpg",
        "attributes": VisualCanonicalizer().canonicalize(scores),
        "upar40": dict(zip(UPAR40_NAMES, scores, strict=True)),
    }
    output = tmp_path / "results.jsonl"
    output.write_text(json.dumps(existing) + "\n", encoding="utf-8")
    predicted_paths: list[list[Path]] = []

    class FakePredictor:
        def __init__(self, *_args):
            predicted_paths.append([])

        def predict_paths(self, paths):
            predicted_paths[-1].extend(paths)
            return [scores for _ in paths]

    monkeypatch.setattr(batch_extract_image, "Upar40Predictor", FakePredictor)
    monkeypatch.setattr(sys, "argv", ["batch_extract_image.py", "--dataset", "cuhk",
                                   "--annotation", str(annotation), "--image-root", str(tmp_path),
                                   "--upar-source", str(tmp_path), "--checkpoint", str(tmp_path / "x"),
                                   "--output", str(output), "--resume", "--batch-size", "2",
                                   "--progress-every", "1"])
    batch_extract_image.main()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["image"] for row in rows] == ["a.jpg", "b.jpg"]
    assert rows[0] == existing
    assert predicted_paths == [[tmp_path / "b.jpg"]]

    batch_extract_image.main()
    assert predicted_paths == [[tmp_path / "b.jpg"]]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_resume_rejects_corrupt_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    output.write_text('{"image": "incomplete', encoding="utf-8")
    with pytest.raises(ValueError, match="不是完整 JSON"):
        batch_extract_image.completed_images(output)
