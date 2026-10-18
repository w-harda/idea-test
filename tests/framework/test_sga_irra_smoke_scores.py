"""两图Smoke的限定评分桥；不放宽Full Ranking契约。"""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import torch
from tbps.artifacts import ArtifactError
from tbps.baselines.sga_irra_smoke_scores import config,official_rank,small_ranking,smoke_engine,main
from tbps.baselines import sga_irra as runner
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"configs/framework/sga_irra_smoke_scores.server.json"

@pytest.mark.parametrize("matrix,r1,ap",[
 ([[1.,0.],[.8,.1],[0.,1.],[.1,.9]],100.,100.),
 ([[0.,1.],[.1,.8],[1.,0.],[.9,.1]],0.,50.),
 ([[1.,0.],[.1,.8],[0.,1.],[.9,.1]],50.,75.),
])
def test_actual_official_small_rank(matrix,r1,ap):
 c=config(CONFIG);rank=official_rank(c)
 identity=SimpleNamespace(gallery_pids=("a","b"),query_pids=("a","a","b","b"))
 got,order=small_ranking(torch.tensor(matrix),identity,rank)
 assert got=={"R@1":r1,"mAP":ap} and order.shape==(4,2)

@pytest.mark.parametrize("x",[torch.zeros(4,3),torch.zeros(10,2),torch.zeros(4,2,dtype=torch.float64),torch.full((4,2),float("nan"))])
def test_fixed_scope_and_finite(x):
 c=config(CONFIG)
 with pytest.raises(ArtifactError,match="fixed two-gallery"):
  small_ranking(x,SimpleNamespace(),official_rank(c))

def test_full_engine_and_metric_constants_restored():
 c=config(CONFIG);old=runner.EvaluationEngine;metrics=runner.METRICS
 with smoke_engine(c):
  assert runner.EvaluationEngine is not old and runner.METRICS==("R@1","mAP")
 assert runner.EvaluationEngine is old and runner.METRICS==metrics

def test_no_unauthorized_target_load():
 with pytest.raises(ArtifactError,match="explicit only fixed smoke"):
  main(["--config",str(CONFIG),"--action","evaluate","--model","aptm"])
 assert not torch.cuda.is_initialized()
