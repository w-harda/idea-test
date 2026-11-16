"""新方法的原子账本、反事实特征保留、教师Checkpoint和拒绝损坏。"""
import copy
import json
from pathlib import Path
import sys
import torch
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from tbps.artifacts import ArtifactStore,ArtifactError,FileRef,sha_file,json_bytes
from tbps.alternating.state import tree_sha
from tbps.alternating.formal_state import tensor_bytes
from tbps.attribute_response.store import FreshJournal,read_ledger
from tbps.attribute_response.teacher import ResponseTeacher
from tbps.attribute_response.settings import Settings
from tbps.attribute_response.memory import semantic_key

def test_ten_boundaries_preserve_counterfactual_features_and_all_teacher_state(tmp_path):
    base=tmp_path/"tbps-framework-gh-attribute-response-v1-20261116";run=base/"train"
    j=FreshJournal(run,"f"*64,base);teacher=ResponseTeacher(Settings(width=8))
    state={"G":{"w":torch.ones(2)},"H":{"w":torch.ones(3)},"parents":{"source":"fixture"},
           "cursor":{"round":0},"attribute_teacher":teacher.state_dict(),
           "attribute_runtime":{"virtual_counts":{"G":0,"H":0},"probe_cursor":0,"response_visits":{"G":0,"H":0}}}
    j.initialize({"fixture":True},state)
    first_feature=None
    for i in range(1,11):
        j.begin(i,state);state=copy.deepcopy(state);state["cursor"]["round"]=i;state["G"]["w"].add_(1)
        key=semantic_key({"semantic_axis":"color","canonical_value":"blue","region":"upper"})
        batch=teacher.features([key],[[.1]*12],version=i);teacher.observe(batch,[.3],0)
        teacher.remember(key,[.1]*12,[.3,0.],version=i)
        state["attribute_teacher"]=teacher.state_dict()
        views=torch.randn(2,3,4);bank=ArtifactStore(run/"B"/f"bank_{i}",new_base=base)
        bank.publish("completed",{"features.pt":tensor_bytes(views[:,0].clone()),"view_features.pt":tensor_bytes(views),
          "view_inputs.json":json_bytes({"entries":[{"row":"r"}]}),"identity.json":json_bytes({"round":i})},
          {"G":tree_sha(state["G"])})
        io=ArtifactStore(run/"round_io"/f"round_{i:04d}",new_base=base)
        io.publish("rollout_0000",{"action.json":json_bytes({"original":"blue","adversarial":"blue","edits":[]})},{})
        feature=torch.randn(3,4);io.publish("h_features_0000",{"features.pt":tensor_bytes(feature),
                      "texts.json":json_bytes({"texts":["a","b","c"]})},{"row":"r"})
        if i==1:first_feature=feature.clone()
        folder=io.root/"h_features_0000"
        h={"attribute_response":{"counterfactual_features":{"artifact":str(folder),
             "feature_sha256":sha_file(folder/"features.pt"),"texts_sha256":sha_file(folder/"texts.json")}}}
        ledger={"round":i,"G":[],"H":[h],"conditions":[{"row":"r"}],"g_orders":[["r"]],
                "bank":{"g_state_sha256":tree_sha(state["G"])}}
        manifest=bank.root/"completed/manifest.json";j.commit(i,state,ledger,FileRef(str(manifest),sha_file(manifest)))
    assert not (run/"round_io/round_0001").exists()
    first=read_ledger(run/"ledgers/round_0001/ledger.json")
    ref=list(first["H"])[0]["attribute_response"]["counterfactual_features"]
    file=run/"ledgers/round_0001"/ref["feature_file"]
    assert torch.equal(torch.load(file,map_location="cpu",weights_only=True),first_feature)
    assert sha_file(file)==ref["feature_sha256"]
    restored,_=j.load_latest(state["parents"]);assert tree_sha(restored)==tree_sha(state)
    exports=ArtifactStore(run/"checkpoints",new_base=base);exports.verify("round_10");exports.verify("teacher_10")
    saved=torch.load(exports.root/"teacher_10/teacher.pt",map_location="cpu",weights_only=False)
    other=ResponseTeacher(Settings(width=8));other.load_state_dict(saved)
    assert tree_sha(saved)==tree_sha(teacher.state_dict())==tree_sha(other.state_dict())
    file.write_bytes(b"corrupt")
    with pytest.raises(ArtifactError):read_ledger(run/"ledgers/round_0001/ledger.json")

def test_real_fail_prevents_boundary_recovery(tmp_path):
    base=tmp_path/"tbps-framework-gh-attribute-response-v1-20261116";run=base/"train";run.mkdir(parents=True)
    (run/"FAIL_initial").write_text("retained actual failure")
    j=FreshJournal(run,"a"*64,base)
    with pytest.raises((ValueError,ArtifactError)):j.load_latest({})
