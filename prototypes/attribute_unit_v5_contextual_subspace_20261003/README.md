# v5 独立 feasibility prototype

无训练，不接线正式pipeline。默认smoke；--mode full为用户手动全量执行。

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_unit_v5_contextual_subspace_20261003
PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest -v test_subspace test_protocol
PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -u run.py --mode full --features pooled
```

--features auto优先经SHA核验的pooled，再block8，再extract。pooled显式要求原cache；存在但hash不匹配则报错，不静默回退。block8只读取原H并重新pool；extract在原caption提取block8。所有模式只在本目录outputs/<mode>/创建产物，拒绝覆盖。失败重跑需先在服务器保存/重命名对应输出目录；程序不会自动删除或替换。

输出：protocol/provenance/corpus/validation/geometry/diagnostic/audit/completed JSON，corpus.pt以及中文summary.md。corpus.pt为统计centroid与basis，非新训练checkpoint。smoke固定每个非null class前2条的union，仅65 train/67 validation；原9句45属性不变。

log map仅构造tangent basis；所有分类统一在raw z/raw centroid投影坐标中进行Euclidean nearest-centroid。完整Euclidean between-value span分类与full理论等价；同rank random才是必要对照。rank0属性记录零响应，不伪造epsilon方向。全量结论不能由smoke推断。
