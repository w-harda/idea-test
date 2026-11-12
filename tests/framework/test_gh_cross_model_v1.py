"""迁移接线、实际缓存/评分/独立审计和拒绝路径的CPU验证。"""
from pathlib import Path
from dataclasses import asdict,replace
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util,json,sys
import pytest,torch
from tbps.artifacts import ArtifactStore,ArtifactError,json_bytes,digest,sha_file
from tbps.gh_transfer import common,evaluation,audit as audit_module
from tbps.models.base import ModelSpace
from tbps.models.rasa import RasaAdapter,PREPROCESS,SCORING,validate_payload
import test_rasa_adapter as fixtures
import test_aptm_adapter as aptm_fixtures
from tbps.models.aptm import APTMAdapter
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/framework/gh_cross_model_v1.server.json'

@pytest.mark.parametrize('key,value',[('round',40),('image_epsilon',16/255),('text_budget','one word'),('modes',['Joint']),('retry_real_failure',True),('no_training',False)])
def test_config_rejects_changed_science(tmp_path,key,value):
    c=json.loads(CONFIG.read_text());c[key]=value;p=tmp_path/'config.json';p.write_text(json.dumps(c))
    with pytest.raises(ArtifactError):common.config(p)

def test_permission_checked_before_any_input_or_model():
    with patch.object(evaluation,'preflight',side_effect=AssertionError('must not preflight')):
        with pytest.raises(ArtifactError):evaluation.run('none','cooperative','rasa','full')

@pytest.mark.parametrize('kind',['budget','delta','digest','bytes'])
def test_input_tensor_corruption_rejected(tmp_path,kind):
    from tbps.engines.provenance import payload_digest
    original=torch.full((1,3,384,128),.5);adv=original.clone();adv[0,0,0,0]+=.01
    delta=adv-original
    if kind=='budget':adv[0,0,0,0]=.6;delta=adv-original
    if kind=='delta':delta.zero_()
    p=tmp_path/'data.pt';torch.save({'original':original,'adversarial':adv,'effective_delta':delta},p)
    item={'data':asdict(common.ref(p)),'adversarial_sha256':payload_digest(adv)}
    if kind=='digest':item['adversarial_sha256']='0'*64
    if kind=='bytes':p.write_bytes(b'bad')
    with pytest.raises((ValueError,ArtifactError)):common.image_value(item)

@pytest.mark.parametrize('method',['cooperative','attraction'])
def test_real_native_Unicode_rule_and_budget_tamper(method):
    c=json.loads(CONFIG.read_text());h=c['historical'][method];f=json.loads(Path(h['configuration']['path']).read_text())
    ns={k:f['native_template'][k] for k in ('managed_roots','import_paths')};ns.update(runtime_pins=f['runtime_pins'],producers=f['producers'])
    from tbps.alternating.native_sources import native_session
    from tbps.alternating.fresh_audit import validate_action
    p=Path(h['output_base'])/'test/round_60/attack/text_0000/action.json';action=json.loads(p.read_text())['native_action']
    with native_session(ns) as s:
        contracts=s.import_module('contracts');constraints=s.import_module('constraints')
        n=validate_action(action,contracts,constraints);assert n<=action['budget_k']
        if n:
            damaged=dict(action,budget_k=0)
            with pytest.raises(ValueError):validate_action(damaged,contracts,constraints)

class Batch(SimpleNamespace):
    def to(self,device):return Batch(**{k:v.to(device) for k,v in vars(self).items()})
class NativeFixture(RasaAdapter):
    @property
    def neural_counts(self):return {k:self.counts[k] for k in ('image_samples','text_samples','fusion_pairs')}

class APTMFixture(APTMAdapter):
    @property
    def neural_counts(self):return {k:self.counts[k] for k in ('image_samples','text_samples','fusion_pairs')}

def integration_fixture(tmp_path,monkeypatch,model='rasa'):
    f,_,m,_=fixtures.native_fixture(tmp_path)
    m=replace(m,gallery=m.gallery[:10])
    base=tmp_path/'tbps-framework-gh-transfer-fixture';base.mkdir()
    cp=f.ref;sourcefile=tmp_path/'target-source.json';sourcefile.write_text('{}');source_ref=common.ref(sourcefile)
    configfile=tmp_path/'transfer.json';configfile.write_text('{}')
    space=ModelSpace(model,cp.sha256,*(digest(i) for i in range(4)),'fp32/math/no_tf32')
    src=SimpleNamespace(space=space,checkpoint=cp,ref=source_ref)
    def tokenize(values,**kw):
        x=torch.tensor([[len(t)%31+2]*(50 if model=='rasa' else 56) for t in values]);return Batch(input_ids=x,attention_mask=torch.ones_like(x))
    adapter=(NativeFixture if model=='rasa' else APTMFixture)((fixtures.Backend if model=='rasa' else aptm_fixtures.Backend)(),tokenize,lambda text,n,**kw:text,space,'cpu',smoke=True)
    @contextmanager
    def loader(*a,**kw):yield adapter
    from importlib import import_module
    native=import_module('tbps.models.'+model)
    mod=SimpleNamespace(load_rasa=loader,load_aptm=loader,PREPROCESS=native.PREPROCESS,SCORING=native.SCORING,validate_payload=native.validate_payload)
    h={'configuration':asdict(common.ref(configfile)),'G':asdict(cp),'H':asdict(cp)}
    c={'output_base':str(base),'historical':{'cooperative':h,'attraction':h}}
    images=[];texts=[]
    from tbps.engines.provenance import payload_digest
    for i,g in enumerate(m.gallery):
        raw=torch.full((1,3,384,128),.2+i*.01);adv=raw+.005
        data=tmp_path/('image'+str(i)+'.pt');torch.save({'original':raw,'adversarial':adv,'effective_delta':adv-raw},data)
        images.append({'id':g.image_id,'data':asdict(common.ref(data)),'original_sha256':g.image_sha256,'adversarial_sha256':payload_digest(adv)})
    for i,q in enumerate(m.queries):
        data=tmp_path/('text'+str(i)+'.json');data.write_bytes(json_bytes({'original':q.caption,'adversarial':q.caption+'x'}))
        texts.append({'id':q.query_id,'data':asdict(common.ref(data)),'original_sha256':common.utf8(q.caption),'adversarial_sha256':common.utf8(q.caption+'x')})
    coll={'images':images,'texts':texts};store=ArtifactStore(base/'input',new_base=base);store.publish('completed',{'report.json':b'{}'},{'fixture':True})
    for module in (evaluation,audit_module):
        monkeypatch.setattr(module,'preflight',lambda *a:(c,mod,src));monkeypatch.setattr(module,'collection',lambda *a:(coll,store));monkeypatch.setattr(module,'manifest',lambda *a:m)
    monkeypatch.setattr(audit_module,'protection',lambda *a,**kw:0)
    from tbps.materialization.irra_startup import CPUStartupGate
    monkeypatch.setattr(CPUStartupGate,'initialize_cuda',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0);monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    return configfile,c,coll,m

@pytest.mark.parametrize('model',['rasa','aptm'])
def test_real_runner_cache_scores_and_independent_cpu_audit(tmp_path,monkeypatch,model):
    path,c,coll,m=integration_fixture(tmp_path,monkeypatch,model)
    report=evaluation.run(path,'cooperative',model,'smoke',allow=True)
    assert report['G_H_loads']==report['optimizer']==report['JVP']==0
    result=audit_module.audit(path,'cooperative',model,'smoke')
    assert result['native_ranking_exact'] and result['groups']==report['groups']
    with pytest.raises(ArtifactError):evaluation.run(path,'cooperative',model,'smoke',allow=True)

@pytest.mark.parametrize('kind',['order','digest'])
def test_bound_input_order_or_content_mismatch(tmp_path,monkeypatch,kind):
    path,c,coll,m=integration_fixture(tmp_path,monkeypatch)
    if kind=='order':coll['images'][0]['id']='wrong'
    else:coll['images'][0]['adversarial_sha256']='0'*64
    with pytest.raises(ArtifactError):evaluation.run(path,'cooperative','rasa','smoke',allow=True)

def test_supervisor_four_full_pairs_no_training():
    spec=importlib.util.spec_from_file_location('transfer_supervisor',ROOT/'operations/gh_cross_model_v1_supervisor.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    plan=module.plan();gpu=[args for _,args in plan if '--allow-gpu' in args]
    assert len(gpu)==8
    assert len([args for args in gpu if args[args.index('--scope')+1]=='full'])==4
    assert not any('train' in args for _,args in plan)
    assert all('--resume' not in args for _,args in plan)

@pytest.mark.parametrize('model',['rasa','aptm'])
def test_independent_audit_rejects_corrupted_native_score(tmp_path,monkeypatch,model):
    path,c,coll,m=integration_fixture(tmp_path,monkeypatch,model)
    evaluation.run(path,'cooperative',model,'smoke',allow=True)
    root=Path(c['output_base'])/'evaluation/cooperative'/model/'smoke'
    name='score-clean_text--clean_gallery-000000' if model=='rasa' else 'aptm-raw-clean_text--clean_gallery-000000'
    (root/name/'score.pt').write_bytes(b'corrupted')
    with pytest.raises(ArtifactError):audit_module.audit(path,'cooperative',model,'smoke')
