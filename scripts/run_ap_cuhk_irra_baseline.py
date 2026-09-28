#!/usr/bin/env python3
"""复用 Baseline 1 全部 TBPS 生成/评测能力，只替换 G_CUHK 及可核验训练来源。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from scripts import run_ap_duke_irra_baseline as baseline
from attributes.ap_gallery_baseline import sha256

DEFAULT_G = Path("/home/lzf/ldx/outputs/AP-Attack/cuhk_reid_semantic_10_10/G_CUHK_AP.pth.tar")
DEFAULT_OUTPUT = Path("/home/lzf/ldx/outputs/idea-TBPS-test1/baseline-ap-cuhk-irra")


def cuhk_provenance(checkpoint, expected_sha=None):
    metadata = json.loads(checkpoint.with_suffix(checkpoint.suffix + ".json").read_text())
    actual = sha256(checkpoint)
    if (metadata["mode"] != "formal" or metadata["training_dataset"] != "CUHK-PEDES/train"
            or actual != metadata["checkpoint_sha256"]
            or (expected_sha is not None and expected_sha != actual)):
        raise ValueError("G_CUHK 来源不匹配或试图用 smoke 权重评测")
    return metadata


def main():
    baseline.__doc__ = __doc__
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--g-checkpoint", type=Path, default=DEFAULT_G)
    parser.add_argument("--g-sha256")
    args, _ = parser.parse_known_args()
    if any(x in sys.argv for x in ("-h", "--help")):
        baseline.DEFAULT_G = DEFAULT_G
        baseline.DEFAULT_OUTPUT = DEFAULT_OUTPUT
        baseline.main()
        return
    metadata = cuhk_provenance(args.g_checkpoint, args.g_sha256)
    baseline.DEFAULT_G = DEFAULT_G
    baseline.DEFAULT_G_SHA = metadata["checkpoint_sha256"]
    baseline.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    baseline.BASELINE = "AP-Attack (G_CUHK^AP → CUHK-PEDES test → IRRA)"
    original_spec = baseline.generator_spec

    def generator_spec(parsed):
        if parsed.g_origin != "local_reproduction":
            raise ValueError("G_CUHK 必须标记 local_reproduction")
        paths, spec = original_spec(parsed)
        spec["training_dataset"] = "CUHK-PEDES/train"
        spec["surrogate"] = metadata["surrogate"]
        spec["training_provenance"] = metadata
        return paths, spec

    baseline.generator_spec = generator_spec
    baseline.main()


if __name__ == "__main__":
    main()
