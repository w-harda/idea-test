"""缓存评分只能引用同一不可变原生编码会话；仅CPU元数据Fixture。"""
from dataclasses import asdict,replace
from pathlib import Path
from unittest.mock import patch
import pytest
from tbps.artifacts import ArtifactError,FileRef,checked,json_bytes,sha_file
from tbps.models.irra_source import OFFICIAL_IRRA_SHA
from tbps.engines.materialized_cache import CachedIRRA
from test_materialized_irra_evaluation import cache_fixture

@pytest.mark.parametrize("change",[None,"tokenizer","generation","feedback","checkpoint","missing"])
def test_encoding_session_required_and_bound(tmp_path,change):
    _,_,space,ck=cache_fixture(tmp_path)
    space=replace(space,model_id="irra",checkpoint_sha256=OFFICIAL_IRRA_SHA)
    ck=FileRef(ck.path,OFFICIAL_IRRA_SHA) # 权重校验只在此合成Fixture中Mock，不反序列化。
    root=tmp_path/"session";root.mkdir()
    session={"schema":"tbps-materialized-irra-cache-1","model_space":asdict(space),
             "G_H_generation":False,"ranking_feedback_to_attack":False}
    if change=="tokenizer":session["model_space"]["tokenizer_sha256"]="0"*64
    if change=="generation":session["G_H_generation"]=True
    if change=="feedback":session["ranking_feedback_to_attack"]=True
    data=root/"protocol.json";data.write_bytes(json_bytes(session))
    manifest=root/"manifest.json"
    manifest.write_bytes(json_bytes({"schema":"tbps-artifact-1",
        "files_sha256":{"protocol.json":sha_file(data)},
        "sources":[] if change=="checkpoint" else [asdict(ck)]}))
    proof=FileRef(str(manifest),sha_file(manifest))
    original=checked
    def fixture_checked(ref):
        return {} if ref==ck else original(ref)
    with patch("tbps.engines.materialized_cache.checked",new=fixture_checked):
        if change is None:
            assert CachedIRRA(space,ck,cache_protocol=proof).source_verified
        else:
            with pytest.raises(ArtifactError):
                CachedIRRA(space,ck,cache_protocol=None if change=="missing" else proof)
