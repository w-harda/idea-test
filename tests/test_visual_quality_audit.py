"""End-to-end checks for the image-only visual audit workflow."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import visual_quality_audit as audit
from attributes.visual_canonicalizer import UPAR40_NAMES, VisualCanonicalizer

PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None


@unittest.skipUnless(PIL_AVAILABLE, "Pillow is required for offline galleries")
class VisualQualityAuditTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        project = Path(__file__).resolve().parents[1]
        self.temp = tempfile.TemporaryDirectory(dir=project / "outputs")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.results.mkdir()
        self.image_roots = {}
        canonicalizer = VisualCanonicalizer()
        for dataset in audit.DATASETS:
            image_root = self.root / dataset
            image_root.mkdir()
            self.image_roots[dataset] = image_root
            with (self.results / f"{dataset}.jsonl").open("w", encoding="utf-8") as handle:
                for number, level in enumerate(("low", "medium", "high"), 1):
                    image = f"person_{number}.jpg"
                    Image.new("RGB", (24, 48), (number * 60, 40, 80)).save(image_root / image)
                    value = {"low": 0.5, "medium": 0.76, "high": 0.98}[level]
                    p = dict.fromkeys(UPAR40_NAMES, 0.04)
                    if level == "low":
                        p = dict.fromkeys(UPAR40_NAMES, 0.5)
                    else:
                        p.update({
                            "Age-Adult": value,
                            "Gender-Female": value,
                            "Hair-Length-Long": value,
                            "UpperBody-Length-Short": value,
                            "UpperBody-Color-Red": value,
                            "LowerBody-Length-Short": value,
                            "LowerBody-Color-Blue": value,
                            "LowerBody-Type-Skirt&Dress": value,
                            "Accessory-Backpack": value,
                            "Accessory-Glasses-Normal": value,
                        })
                    attrs = canonicalizer.canonicalize([p[name] for name in UPAR40_NAMES])
                    handle.write(json.dumps({
                        "image": image, "attributes": attrs, "upar40": p,
                    }) + "\n")
        self.output = self.root / "audit"
        self.sample_args = argparse.Namespace(
            results_dir=self.results, cuhk_image_root=self.image_roots["cuhk"],
            icfg_image_root=self.image_roots["icfg"],
            rstp_image_root=self.image_roots["rstp"],
            output_dir=self.output, per_dataset=3, seed=7)

    def test_sample_and_evaluate_partial_human_labels(self):
        audit.sample(self.sample_args)
        rows = [json.loads(line) for line in
                (self.output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 9)
        for dataset in audit.DATASETS:
            selected = [row for row in rows if row["dataset"] == dataset]
            self.assertEqual({row["sample_band"] for row in selected},
                             {"low", "medium", "high"})
            self.assertTrue((self.output / f"{dataset}.html").read_text(
                encoding="utf-8").count("data:image/jpeg;base64,") == 3)
        labels_path = self.output / "labels.csv"
        with labels_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            labels = list(reader)
        cuhk = {row["sample_band"]: row for row in rows if row["dataset"] == "cuhk"}
        for label in labels:
            if label["sample_id"] == cuhk["high"]["sample_id"]:
                label["truth_gender"] = "female"
                label["truth_upper_clothing_type"] = "jacket_coat"
            if label["sample_id"] == cuhk["medium"]["sample_id"]:
                label["truth_gender"] = "null"
            if label["sample_id"] == cuhk["low"]["sample_id"]:
                label["truth_gender"] = "female"
        with labels_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(labels)
        report_path = self.output / "report.json"
        audit.evaluate(argparse.Namespace(
            manifest=self.output / "manifest.jsonl",
            labels=labels_path, output=report_path))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        gender = report["datasets"]["cuhk"]["slots"]["gender"]
        self.assertEqual(gender["known_truth_count"], 2)
        self.assertEqual(gender["truth_null_count"], 1)
        self.assertEqual(gender["predicted_non_null_count"], 1)
        self.assertEqual(gender["correct_count"], 1)
        self.assertEqual(gender["accuracy"], 1.0)
        self.assertEqual(gender["non_null_coverage"], 0.5)
        self.assertEqual(gender["null_ratio"], 0.5)
        self.assertEqual(gender["confidence_bands"]["high"]["accuracy"], 1.0)
        self.assertEqual(report["datasets"]["cuhk"]["overall"]["known_truth_count"], 2)
        with self.assertRaisesRegex(ValueError, "already exists"):
            audit.sample(self.sample_args)

    def test_invalid_canonical_truth_is_rejected(self):
        audit.sample(self.sample_args)
        labels_path = self.output / "labels.csv"
        with labels_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            labels = list(reader)
        labels[0]["truth_gender"] = "not_a_gender"
        with labels_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(labels)
        with self.assertRaisesRegex(ValueError, "Invalid truth_gender"):
            audit.evaluate(argparse.Namespace(
                manifest=self.output / "manifest.jsonl",
                labels=labels_path, output=self.output / "report.json"))


if __name__ == "__main__":
    unittest.main()
