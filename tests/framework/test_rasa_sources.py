"""真实来源只读CPU校验；不加载模型或初始化CUDA。"""
from pathlib import Path
from unittest.mock import patch
import json,sys
import pytest
import torch
from tbps.artifacts import ArtifactError
from tbps.models.rasa import RasaSource,native_modules,official_tokenizer
from tbps import rasa_eval

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"configs/framework/rasa_sources.server.json"

@pytest.mark.parametrize("kind",["missing","tamper","runtime"])
def test_source_missing_tamper_or_environment_rejected(tmp_path,kind):
    data=json.loads(SOURCE.read_text());key=data["root"]+"/models/vit.py"
    if kind=="missing":data["files"].pop(key)
    elif kind=="tamper":data["files"][key]="0"*64
    else:data["runtime_versions"]["transformers"]="0.0"
    p=tmp_path/"sources.json";p.write_text(json.dumps(data))
    with patch.object(torch,"load",side_effect=AssertionError("no weights")),patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("no CUDA")):
        with pytest.raises(ArtifactError):RasaSource(p).verify()

def test_actual_official_import_tokenizer_and_namespace_restore():
    source=RasaSource(SOURCE);before=list(sys.path)
    prior={k:v for k,v in sys.modules.items() if k=="models" or k.startswith("models.")}
    with patch.object(torch,"load",side_effect=AssertionError("no weights")),patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("no CUDA")):
        with native_modules(source) as (native,tok,utils):
            tokenizer=official_tokenizer(tok,source)
            ids=tokenizer([utils.pre_caption(t,50) for t in ["A blue shirt.","A blυe shirt."]],
                padding="max_length",truncation=True,max_length=50,return_tensors="pt").input_ids
            assert ids.shape==(2,50) and not torch.equal(ids[0],ids[1])
            assert native.ALBEF.__module__=="models.model_person_search"
    assert sys.path==before and all(sys.modules.get(k)is v for k,v in prior.items())
    assert not torch.cuda.is_initialized()

def test_CLI_bootstrap_selects_frozen_producer_paths_only():
    import tbps.attacks,tbps.materialization
    paths=[list(p.__path__) for p in [tbps.attacks,tbps.materialization]]
    try:
        with patch("tbps.engines.materialized_rasa.main",return_value=0):
            assert rasa_eval.main()==0
        assert list(tbps.attacks.__path__)==[str(rasa_eval.FROZEN/"attacks")]
        assert list(tbps.materialization.__path__)==[str(rasa_eval.FROZEN/"materialization")]
    finally:
        for p,old in zip([tbps.attacks,tbps.materialization],paths):p.__path__[:]=old
