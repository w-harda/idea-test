"""仅G-v2评测输出可写；不导入G-v1或训练入口。"""
import sys,json,hashlib
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import cache_io as io
EVAL_OUTPUTS=ROOT/"outputs/evaluation_phase29a"
OFFICIAL=Path("/home/lzf/Attack/IRRA-main/IRRA-main")
RAW=Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json")
IMAGES=RAW.parent/"imgs"
TRAIN=ROOT/"outputs/train_v2"
CP_SHA={1:"876547ffeb827cbb4691bbf5b5d473da495590d15df2adba12c9e1415e52a27a",
3:"a42a6dc0fcae269ee2e8efd379b4a3f3c947f22c50181a56d15acf82a8450d02",
10:"f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e"}
REFERENCE=Path("/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_selection_full_test_20261011/outputs/full")
REF_SHA={"clean_features.pt":"3ef23d81e097f88975d9f9151cba2a58ca50b2a20547fba9c908027674bf1800",
"selection_manifest.json":"5c1bbd04f878bf20419ddf7062d74c661445f3bf8be603a21577c7cee2740b40",
"protocol.json":"22bd4cb1a232a4e6e318e04627ff0ada565ad6351969a2711c9c31531862dfa2",
"summary.json":"dc706b04b242ffe4ac8ac0645819199cadbf3ed100b4219c2fb8d8691c76acac"}
METRICS=("mAP","R@1","R@5","R@10","mINP")
GROUPS=("clean","epoch1","epoch3","epoch10")
FEATURE_ATOL=2e-5
CLEAN_METRIC_ATOL_PP=.001
def safe(path):
 p=io.safe(path)
 if not p.is_relative_to(EVAL_OUTPUTS.resolve()):raise ValueError("evaluation outputs must be isolated from training")
 return p
def official_imports():
 sys.path.insert(0,str(OFFICIAL))
 from datasets.cuhkpedes import CUHKPEDES
 from datasets.bases import TextDataset
 from datasets.build import build_transforms
 return CUHKPEDES,TextDataset,build_transforms
