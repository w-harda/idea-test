"""真实环境CPU预检回归：不Mock preflight，不加载模型，不初始化CUDA。"""
from pathlib import Path
from importlib.metadata import version
from unittest.mock import patch
import ast
import builtins
import pytest
import torch
from tbps.artifacts import ArtifactError
from tbps.baselines import dra_full as full

ROOT=Path(__file__).parents[2]
CONFIG=ROOT/"configs/framework/dra_irra_full_v2.server.json"
SOURCE=version("torch")=="2.1.0+cu121"
TARGET=version("torch")=="2.5.1+cu124"

def forbidden(*a,**kw):
    raise AssertionError("CPU preflight must not load models or initialize CUDA")

def source_only_import(name,*a,**kw):
    if name in ("tbps.models.rasa","tbps.models.aptm"):
        raise AssertionError("Source must not enter Target runtime modules")
    return REAL_IMPORT(name,*a,**kw)
REAL_IMPORT=builtins.__import__

@pytest.mark.skipif(not SOURCE,reason="真实dra-irra-source环境专用")
def test_source_actual_preflight_no_target_import_or_model_load():
    with patch("builtins.__import__",side_effect=source_only_import),patch.object(torch,"load",side_effect=forbidden),patch.object(torch.cuda,"_lazy_init",side_effect=forbidden):
        c,ds,raw,m,report=full.preflight(CONFIG,stage="source")
    assert report["stage"]=="source" and report["Gallery"]==3074 and report["Query"]==6156
    assert report["batches"]==385 and report["ten_caption_gallery"]=="cuhk:38920:0"
    assert report["model_loads"]==0 and not report["GPU"] and not torch.cuda.is_initialized()

@pytest.mark.skipif(not SOURCE,reason="真实dra-irra-source环境专用")
def test_actual_source_run_reaches_store_after_real_source_preflight():
    class EndCPU(Exception):pass
    with patch("builtins.__import__",side_effect=source_only_import),patch.object(torch,"load",side_effect=forbidden),patch.object(torch.cuda,"_lazy_init",side_effect=forbidden),patch.object(full,"resources",return_value={"CPU_fixture":True}),patch.object(full,"ArtifactStore",side_effect=EndCPU) as store,patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":"0"}):
        with pytest.raises(EndCPU):full.source_run(CONFIG)
        store.assert_called_once()
    assert not Path(full.configuration(CONFIG)["output_base"]).exists()
    assert not torch.cuda.is_initialized()

@pytest.mark.skipif(not SOURCE,reason="真实dra-irra-source环境专用")
def test_source_version_mismatch_still_refused():
    with patch.object(full,"version",return_value="wrong"):
        with pytest.raises(ArtifactError,match="pinned dedicated DRA runtime"):
            full.preflight(CONFIG,stage="source")

@pytest.mark.skipif(not TARGET,reason="真实tbps-image环境专用")
def test_target_actual_native_verify_and_import_contexts():
    import tbps.models.rasa as rasa
    import tbps.models.aptm as aptm
    calls=[];rverify=rasa.RasaSource.verify;averify=aptm.APTMSource.verify
    def rv(self,*a,**kw):calls.append("rasa");return rverify(self,*a,**kw)
    def av(self,*a,**kw):calls.append("aptm");return averify(self,*a,**kw)
    with patch.object(rasa.RasaSource,"verify",rv),patch.object(aptm.APTMSource,"verify",av),patch.object(torch,"load",side_effect=forbidden),patch.object(torch.cuda,"_lazy_init",side_effect=forbidden):
        c,ds,raw,m,report=full.preflight(CONFIG,stage="target")
    assert "rasa" in calls and "aptm" in calls and report["stage"]=="target"
    assert report["model_loads"]==0 and not report["GPU"] and not torch.cuda.is_initialized()

@pytest.mark.skipif(not TARGET,reason="真实tbps-image环境专用")
@pytest.mark.parametrize("model",["rasa","aptm"])
def test_target_dependency_error_not_swallowed(model):
    mod=__import__("tbps.models."+model,fromlist=["RasaSource"])
    cls=mod.RasaSource if model=="rasa" else mod.APTMSource
    with patch.object(cls,"verify",side_effect=ArtifactError("injected "+model+" dependency mismatch")) as verify:
        with pytest.raises(ArtifactError,match="dependency mismatch"):
            full.preflight(CONFIG,stage="target")
        verify.assert_called_once()

def test_invalid_stage_rejected():
    with pytest.raises(ArtifactError,match="preflight stage"):full.preflight(CONFIG,stage="all")

def test_source_entry_explicitly_selects_source_stage():
    tree=ast.parse((ROOT/"src/tbps/baselines/dra_full.py").read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="source_run")
    call=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="preflight")
    assert [(k.arg,ast.literal_eval(k.value)) for k in call.keywords]==[("stage","source")]
