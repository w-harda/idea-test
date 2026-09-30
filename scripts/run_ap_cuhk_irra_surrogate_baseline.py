#!/usr/bin/env python3
"""Baseline 3 冻结 IRRA 图像代理：复用原 AP 全图库和 IRRA TBPS 评测。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from scripts import run_ap_duke_irra_baseline as baseline
from attributes.ap_gallery_baseline import sha256
from attributes.irra_victim import OFFICIAL_CUHK_SHA256

DEFAULT_G = Path("/home/lzf/ldx/outputs/AP-Attack/cuhk_irra_semantic_10_10/G_CUHK_IRRA_AP.pth.tar")
DEFAULT_OUTPUT = Path("/home/lzf/ldx/outputs/idea-TBPS-test1/baseline-ap-cuhk-irra-surrogate")


def irra_surrogate_provenance(checkpoint, expected_sha=None):
    metadata = json.loads(checkpoint.with_suffix(checkpoint.suffix + ".json").read_text())
    actual = sha256(checkpoint)
    if (metadata.get("mode") != "formal"
            or metadata.get("training_dataset") != "CUHK-PEDES/train"
            or metadata.get("surrogate") != "official_IRRA_image_encoder"
            or metadata.get("irra_checkpoint_sha256") != OFFICIAL_CUHK_SHA256
            or metadata.get("training_uses_captions") is not False
            or metadata.get("training_uses_irra_text") is not False
            or metadata.get("epoch") != 60
            or metadata.get("checkpoint_sha256") != actual
            or (expected_sha is not None and expected_sha != actual)):
        raise ValueError("G_CUHK_IRRA_AP 来源不匹配或试图用 smoke 权重评测")
    return metadata


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--g-checkpoint", type=Path, default=DEFAULT_G)
    parser.add_argument("--g-sha256")
    args, _ = parser.parse_known_args()
    if any(x in sys.argv for x in ("-h", "--help")):
        baseline.DEFAULT_G = DEFAULT_G
        baseline.DEFAULT_OUTPUT = DEFAULT_OUTPUT
        baseline.main()
        return
    metadata = irra_surrogate_provenance(args.g_checkpoint, args.g_sha256)
    baseline.DEFAULT_G = DEFAULT_G
    baseline.DEFAULT_G_SHA = metadata["checkpoint_sha256"]
    baseline.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    baseline.BASELINE = "AP-Attack (G_CUHK^IRRA-image → CUHK-PEDES test → IRRA)"
    original_spec = baseline.generator_spec

    def generator_spec(parsed):
        if parsed.g_origin != "local_reproduction":
            raise ValueError("G_CUHK_IRRA_AP 必须标记 local_reproduction")
        paths, spec = original_spec(parsed)
        spec["training_dataset"] = "CUHK-PEDES/train"
        spec["surrogate"] = metadata["surrogate"]
        spec["training_provenance"] = metadata
        return paths, spec

    baseline.generator_spec = generator_spec
    baseline.main()


if __name__ == "__main__":
    main()
