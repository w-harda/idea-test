"""发布门禁使用合成小型Full回执；不加载Tensor、模型或GPU。"""
from pathlib import Path
import csv
import hashlib
import io
import json

import pytest

from tbps.artifacts import ArtifactStore,ArtifactError,json_bytes
from tbps.baselines.cpgc_sep.runtime import ref
from operations import sep_word_budget1_publish as p


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/"tbps-framework-publisher-fixture"
    config_file=tmp_path/"experiment.json";config_file.write_text("{}")
    checkpoint={"path":"/source.pth","sha256":"0"*64}
    target_checkpoint={"path":"/target.pth","sha256":"1"*64}
    qids=["q0","q1"]
    values={"R@1":50.,"R@5":75.,"R@10":100.,"mAP":45.125}
    c={"output":str(root),"source_checkpoint":checkpoint,
       "victim_checkpoints":{"cuhk_irra":target_checkpoint},
       "experiment_config":{"path":str(config_file),"sha256":p.sha_file(config_file)},
       "control":str(tmp_path/"control"),"report_path":"docs/results",
       "branch":"codex/fixture","repo_url":"https://github.com/example/project",
       "push_retry_seconds":300}
    Path(c["control"]).mkdir()
    monkeypatch.setitem(p.SIZES,"cuhk",(3,2))
    def store(part):return ArtifactStore(root/part,new_base=root)
    def publish(s,key,data,sources=()):
        return s.publish(key,{name:json_bytes(value) if name.endswith(".json") else value
                              for name,value in data.items()},{"kind":"synthetic fixture"},sources=sources)
    evaluation=store("inputs")
    publish(evaluation,"evaluation-cuhk",{"manifest-data.json":{"queries":[{"query_id":x} for x in qids],"gallery":[{}, {}, {}]}})
    source=store("source/cuhk")
    source_report={"status":"PASS","scope":"full","dataset":"cuhk","queries":2,"checkpoint":checkpoint}
    index=[{"key":hashlib.sha256(("cuhk\0"+q).encode()).hexdigest()} for q in qids]
    publish(source,"completed-full",{"report.json":source_report,"index.json":index})
    target=store("full/target/cuhk/irra")
    summary={"status":"PASS","scope":"full","dataset":"cuhk","victim":"irra","mode":"Text-only",
             "Gallery":3,"Query":2,"checkpoint":target_checkpoint,"metrics":values,"query_ids":qids,
             "frozen_text_sha256":["2"*64,"3"*64],"elapsed_seconds":3.}
    csv_file=io.StringIO()
    w=csv.writer(csv_file);w.writerow(["Dataset","Victim","Attack",*p.METRICS])
    w.writerow(["cuhk","irra","SEP Text-only",*(values[k] for k in p.METRICS)])
    publish(target,"completed",{"summary.json":summary,"metrics.csv":csv_file.getvalue().encode(),
                                "scores.pt":b"opaque synthetic tensor; publisher must never open"},
            sources=(ref(source.root/"completed-full/manifest.json"),))
    audit=store("full/audit/cuhk/irra")
    report={"status":"PASS","scope":"full","dataset":"cuhk","victim":"irra","CUDA":False,
            "Gallery":3,"Query":2,"word_budget":1,"budget_violations":0,"noop_queries":1,"one_word_queries":1,
            "noop_reasons":{"NO_LEGAL_SYNONYM":1},"checkpoint":target_checkpoint,"source_checkpoint":checkpoint,
            "metrics":values,"result":{},
            "source_total_seconds":4.,"mean_source_query_seconds":2.,"target_seconds":3.,"source_forward_count":20}
    # FileRef uses dataclass slots. Keep exact path/SHA via the public representation.
    from dataclasses import asdict
    report["result"]=asdict(ref(target.root/"completed/manifest.json"))
    def publish_audit():
        publish(audit,"completed",{"report.json":report},sources=(ref(target.root/"completed/manifest.json"),))
    return c,root,report,summary,publish_audit


def test_target_without_completed_audit_waits(fixture):
    c,root,*_=fixture
    assert p.read_group(c,"cuhk","irra") is None
    assert not (root/"REPORT.md").exists()


def test_valid_group_readonly_and_tensor_never_opened(fixture,monkeypatch):
    c,root,_,_,ready=fixture
    ready()
    before={str(x):p.sha_file(x) for x in root.rglob("*") if x.is_file()}
    original=Path.open
    def protected(self,*args,**kwargs):
        if self.suffix==".pt":raise AssertionError("large Tensor opened by publisher")
        return original(self,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path,"open",protected)
        row=p.read_group(c,"cuhk","irra")
        text=p.render_group(row)
    after={str(x):p.sha_file(x) for x in root.rglob("*") if x.is_file()}
    assert before==after
    assert row["noop"]==row["one_word"]==1 and row["metrics"]["mAP"]==45.125
    assert "50.0000%" in text and "Word Budget违规 0" in text and "CPU审计 PASS" in text


@pytest.mark.parametrize("field,value",[
    ("scope","smoke"),("Query",1),("budget_violations",1),("word_budget",2),
    ("noop_queries",2),("metrics",{"R@1":0.,"R@5":75.,"R@10":100.,"mAP":45.125}),
    ("checkpoint",{"path":"/wrong.pth","sha256":"4"*64})])
def test_invalid_or_partial_artifact_cannot_be_published(fixture,field,value):
    c,_,report,_,ready=fixture
    report[field]=value;ready()
    with pytest.raises(ArtifactError):
        p.read_group(c,"cuhk","irra")


def test_summary_waits_for_all_six_groups():
    with pytest.raises(ArtifactError,match="all six"):
        p.render_summary({})


def test_failed_github_push_preserves_local_report_and_state(tmp_path,monkeypatch):
    control=tmp_path/"control";control.mkdir()
    report=tmp_path/"REPORT.md";report.write_text("local report")
    c={"control":str(control),"branch":"codex/fixture","push_retry_seconds":300,"repo_url":"https://github.com/a/b"}
    record={"publication":"COMMITTED_LOCAL","commit":"0"*40,"path":"REPORT.md"}
    state={"groups":{"cuhk_irra":record}}
    def fail(*_):raise RuntimeError("network unavailable")
    monkeypatch.setattr(p,"git",fail)
    p.push(c,state,force=True)
    assert record["publication"]=="COMMITTED_LOCAL" and "network unavailable" in state["push_error"]
    assert report.read_text()=="local report" and "PUSH_FAILED" in (control/"events.jsonl").read_text()


def test_group_committed_once_and_no_early_summary(fixture,monkeypatch):
    c,_,_,_,ready=fixture
    ready()
    data=p.read_group(c,"cuhk","irra")
    monkeypatch.setattr(p,"read_group",lambda _c,d,v:data if (d,v)==("cuhk","irra") else None)
    calls=[]
    monkeypatch.setattr(p,"commit_report",lambda *args: calls.append(args) or "5"*40)
    def publish(_c,state,force=False):
        for r in state["groups"].values():r["publication"]="PUBLISHED"
    monkeypatch.setattr(p,"push",publish)
    state={"groups":{},"validation_errors":{}}
    p.cycle(c,state);p.cycle(c,state)
    assert len(calls)==1 and "summary" not in state and state["status"]=="MONITORING"
