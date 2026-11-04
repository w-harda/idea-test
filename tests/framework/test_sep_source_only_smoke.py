import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from operations.sep_source_only import attack, common, parity, phases
from tbps.artifacts import ArtifactError
from tbps.baselines.cpgc_sep.sep_bridge import PIDPlan, prepare_caption
from tbps.contracts import PersonID

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("CPU Smoke test forbids CUDA/weights/backward")
    for obj, name in ((torch.cuda, "_lazy_init"), (torch.cuda, "init"), (torch, "load"),
                      (torch.Tensor, "cuda"), (torch.Tensor, "backward"), (torch.autograd, "grad")):
        monkeypatch.setattr(obj, name, forbidden)
    torch.set_num_threads(2)
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def config():
    return common.source_config(common.settings()[1])


class Source(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.cache = {}
        self.calls = self.hits = 0
        self.values = values
        self.pid_plan = PIDPlan.build(tuple(PersonID("cuhk_train", str(i)) for i in range(2)))

    def forward(self, texts):
        result = []
        for text in texts:
            self.calls += 1
            if text in self.cache:
                self.hits += 1
            else:
                value = sum(self.values.get(w, 0.) for w in text.split())
                self.cache[text] = torch.tensor([value, -.5])
            result.append(self.cache[text])
        return SimpleNamespace(logits=torch.stack(result))


def test_original_dpp_degenerates_but_consumes_original_rng(config):
    common.seed()
    before = torch.get_rng_state()
    model = Source({})
    ensemble = attack.s1_ensemble(config, model, "cpu")
    ensemble.init_weights()
    actual = torch.get_rng_state()
    torch.set_rng_state(before)
    torch.empty(100, 1).uniform_(0, 1)
    assert torch.equal(actual, torch.get_rng_state())
    assert ensemble.len_weights == 1 and ensemble.next_weights().tolist() == [1.]
    assert tuple(ensemble.models) == ("irra",)


@pytest.mark.parametrize("noop", [False, True])
def test_official_full_caption_path_fixed_source_pid_and_boundary(config, monkeypatch, noop):
    import nltk
    monkeypatch.setattr(nltk.data, "path", [config["nltk_root"]])
    common.seed()
    raw = "A man wearing a BLUE shirt."
    vals = {"man":.40, "wearing":.33, "blue":.25, "shirt":.12,
            "person":.10, "dressed":.09, "azure":.08, "top":.04}
    alternatives = {"man":"person", "wearing":"dressed", "blue":"azure", "shirt":"top"}
    terms = list(vals)
    mapping = {w:i for i,w in enumerate(terms)}
    neighbors = [[(1.,i),(.9,i if noop else mapping[alternatives.get(w,w)])] for i,w in enumerate(terms)]
    model = Source(vals)
    row, logits = attack.attack_caption(config, raw, model, "cpu",
                                      (mapping, dict(enumerate(terms)), neighbors), lambda *_, **__:None)
    assert row["pseudo_pid_index"] == logits[0].argmax().item() == 0
    assert row["source_oracle"][0] == {"texts":[prepare_caption(config, raw)], "source_top1_changed":[False]}
    assert row["noop"] is noop
    assert (row["adversarial"] == raw) is noop
    assert row["text_forwards"] == len(model.cache)
    assert len(row["candidates"]) == int(not noop)


def test_source_config_has_no_target_data(config):
    assert set(config["victims"]) == {"irra"}
    assert set(config["victims"]["irra"]["datasets"]) == {"cuhk"}
    data = json.dumps(config)
    for term in ("reid_raw.json", "data_captions.json", "ICFG-PEDES.json", "rasa_cuhk_checkpoint", "aptm_cuhk_checkpoint",
                 "/RSTPReid/best.pth", "/ICFG-PEDES/best.pth"):
        assert term not in data


def test_landlock_blocks_target_and_label_files_and_writes(tmp_path):
    allowed = tmp_path/"allow.txt"
    forbidden = tmp_path/"target-label.txt"
    weights = tmp_path/"target-checkpoint.pt"
    output = tmp_path/"new"
    output.mkdir()
    for p in (allowed, forbidden, weights):
        p.write_text("fixture")
    script = """
from pathlib import Path
from operations.sep_source_only.isolation import restrict_source, must_deny
import sys
allow, target, checkpoint, output = map(Path, sys.argv[1:])
assert restrict_source([allow], [output]) >= 3
assert allow.read_text() == 'fixture'
must_deny([target, checkpoint])
(output/'proof.txt').write_text('allowed')
try:
    allow.write_text('change')
except PermissionError:
    pass
else:
    raise AssertionError('Source original changed')
"""
    subprocess.run([sys.executable, "-B", "-c", script, *map(str, (allowed, forbidden, weights, output))],
                   cwd=ROOT, env=os.environ.copy(), check=True)
    assert (output/"proof.txt").read_text() == "allowed" and allowed.read_text() == "fixture"


@pytest.mark.parametrize("replacement,pos", [("banned","NOUN"),("person","DET")])
def test_lexical_or_pos_violation_stops(replacement, pos):
    with pytest.raises(ArtifactError):
        attack.verify_edits("man", replacement, ({"man":0}, {0:"man",1:"person"}, [[(1,0),(.9,1)]]), [pos])


def test_final_word_budget_stops():
    with pytest.raises(ArtifactError, match="final word budget"):
        attack.verify_edits("man man", "person person",
                            ({"man":0}, {0:"man",1:"person"}, [[(1,0),(.9,1)]]), ["NOUN","NOUN"])


def test_deadline_raises_and_restores_handler():
    import signal
    import time
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError, match="attempt stops"):
        with attack.deadline(.01):
            time.sleep(.1)
    assert signal.getsignal(signal.SIGALRM) == previous


def test_immutable_cpu_replay_uses_old_and_new_production(config):
    pids = tuple(PersonID("cuhk_pedes", str(i)) for i in (0,1,1,2,0))
    manifest = SimpleNamespace(gallery=tuple(SimpleNamespace(pid=p) for p in pids),
                               queries=(SimpleNamespace(pid=pids[0]),))
    base = common.settings()[1]
    matrix = torch.tensor([[.9,.8,.7,.6,.5], [.3,.1,.2,.5,.4], [.4,.2,.1,.5,.3]])
    result = parity.compare(base, manifest, ["clean","candidate one","candidate two"], {"irra":matrix, "rasa":matrix*.5})
    assert result["status"] == "EXACT" and result["rng"]
    assert all(x["cache_hits"] == 2 for x in result["models"].values())


def test_bitwise_near_difference_stops():
    value = torch.tensor([1.])
    other = torch.nextafter(value, torch.tensor([2.]))
    with pytest.raises(ValueError, match="element 0"):
        parity.assert_bits("fixture", value, other)


def test_target_load_is_after_complete_freeze(monkeypatch):
    calls = []
    monkeypatch.setattr(phases, "eval_sets", lambda _: {"cuhk":"private"})
    monkeypatch.setattr(phases, "eval_manifest", lambda value: value)
    def fail(*_):
        raise ArtifactError("unfrozen")
    monkeypatch.setattr(phases, "frozen_texts", fail)
    monkeypatch.setattr(phases.native, "native", lambda *a: calls.append(a))
    with pytest.raises(ArtifactError, match="unfrozen"):
        phases.target({"output":"/unwritten","control":"/unwritten"}, {}, None, "cuhk-rasa")
    assert calls == []


def test_real_native_source_tokenizer_cpu_only(config):
    from tbps.baselines.cpgc_sep.victims import VictimSource
    from tbps.models.irra_io import official_tokenizer
    tokenize = official_tokenizer(VictimSource(config, "irra", "cuhk"))
    result = phases.token_receipt(tokenize, ["a man with blue pants", "person " * 90])
    assert not result["reports"][0]["truncated"] and result["reports"][1]["truncated"]
    assert result["reports"][0]["ids"][0] == 49406
