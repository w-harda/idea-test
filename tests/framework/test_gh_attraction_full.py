"""Full生命周期：独立KL重算、历史Bank保留、冻结SHA和原生Joint评分。"""
import copy,importlib.util,json,math
from pathlib import Path
from dataclasses import asdict
import pytest
import torch
from torch.nn import functional as F
from tbps.artifacts import ArtifactStore,ArtifactError,FileRef,sha_file,json_bytes,digest
from tbps.alternating.state import tree_sha
from tbps.alternating.formal_state import tensor_bytes
from tbps.attraction.objectives import ProfileKL,Settings,scalars
from tbps.attraction.audit_math import audit_attraction
ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    def deny(*a,**kw):raise AssertionError('Full CPU test entered CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',deny);torch.set_num_threads(2)

def fixture_ledger():
    torch.manual_seed(47);anchors=F.normalize(torch.randn(8,4),dim=1);bank=F.normalize(torch.randn(8,4),dim=1)
    texts=F.normalize(torch.randn(3,4),dim=1);cfg=asdict(Settings());profile=ProfileKL(anchors,Settings(**cfg))
    gv=scalars(profile.terms(bank[:1],anchors[:1],texts[1:2]))
    rec={'phase':'joint','global':gv['loss'],'local':0.,'retain':0.,'total':gv['loss'],
         'finite_gradients':True,'G_gradient_l2':.5,'cumulative_linf':.01,
         'attraction':{'phase':'joint','image_feature':bank[0].tolist(),**gv}}
    ident={'original_caption_sha256':'a'*64,'context_caption_sha256':'b'*64,'selected_g_ids':[],
        'scores':[],'source_space_sha256':'c'*64,'anchor_mapping_sha256':'d'*64}
    g={'row_id':'r','G_before':'G0','G_after':'G1','K':0,'optimizer_steps':1,'records':[rec],
        'weights':{'lambda_global':1.,'lambda_local':1.,'lambda_retain':1.},
        'gradient_observations':[{'record':{k:v for k,v in rec.items() if k!='attraction'},'gradient_sha256':'f'*64}],
        'attraction':{'teacher_H_state':'H0','anchor_sha256':tree_sha(anchors),'candidate_count':8,'settings':cfg,
            'condition_identity':digest(ident),'partner_text_feature':texts[1].tolist()}}
    hv=[scalars(profile.terms(v[None],texts[:1],bank[:1])) for v in texts[1:]]
    h={'row_id':'r','optimizer_steps_this_step':1,'IRRA_text_samples':3,'gradient_norm':.1,
        'H_state_sha256_before':'H0-legacy','H_state_sha256_after':'H1-legacy','sequence_log_prob':-3.,
        'attraction':{'teacher_G_state':'G1','settings':cfg,'anchor_sha256':tree_sha(anchors),'candidate_count':8,
            'R_sim_optimized':False,'R_rank_optimized':False,'text_features':texts.tolist(),
            'partner_image_feature':bank[0].tolist(),'sampled':hv[0],'greedy':hv[1]}}
    for name,v in zip(('sampled','greedy'),hv):h[name+'_reward']={'total':v['reward'],**{k:v[k] for k in ('pull_kl','push_kl','baseline_pull_kl','pull_gain')}}
    h['advantage']=hv[0]['reward']-hv[1]['reward'];h['SCST_loss']=-h['advantage']*h['sequence_log_prob']
    ledger={'G_before':'G0','G_after':'G1','condition_H_state':'H0','G':[g],'H':[h],
        'conditions':[{'row':'r','identity':ident}],'g_orders':[['r']],'h_orders':[['r']],
        'bank':{'ordered_train_ids':['r']+['other-'+str(i) for i in range(7)]}}
    return ledger,cfg,anchors,bank

def test_independent_FP64_recomputation_matches_production_KL():
    log,cfg,anchors,bank=fixture_ledger();r=audit_attraction(log,cfg,None,None,anchors,bank)
    assert r['G_steps_independently_recomputed']==r['H_steps_independently_recomputed']==1
    assert r['image_budget_tolerance']==0 and r['neural_forward_backward_optimizer_JVP']==0

@pytest.mark.parametrize('case',['G_loss','G_feature','G_teacher','H_reward','H_feature','H_teacher','H_scst','pixel_budget'])
def test_independent_audit_rejects_corruption(case):
    log,cfg,anchors,bank=fixture_ledger()
    if case=='G_loss':log['G'][0]['records'][0]['global']+=.1
    elif case=='G_feature':log['G'][0]['records'][0]['attraction']['image_feature'][0]+=.2
    elif case=='G_teacher':log['G'][0]['attraction']['teacher_H_state']='stale'
    elif case=='H_reward':log['H'][0]['sampled_reward']['total']+=.1
    elif case=='H_feature':log['H'][0]['attraction']['partner_image_feature'][0]+=.2
    elif case=='H_teacher':log['H'][0]['attraction']['teacher_G_state']='stale'
    elif case=='H_scst':log['H'][0]['SCST_loss']+=.1
    elif case=='pixel_budget':log['G'][0]['records'][0]['cumulative_linf']=8/255+1e-9
    with pytest.raises(ArtifactError):audit_attraction(log,cfg,None,None,anchors,bank)

def test_new_journal_retains_original_Bank_features_and_G_rollouts(tmp_path):
    from tbps.attraction.store import FreshJournal,read_ledger
    base=tmp_path/'tbps-framework-gh-attraction-v1-20261109';run=base/'train';j=FreshJournal(run,'f'*64,base)
    state={'G':{'v':torch.ones(1)},'H':{'v':torch.ones(1)},'parents':{'source':'fixed'},'cursor':{'round':0}}
    j.initialize({'fixture':True},state)
    for i in (1,2):
        j.begin(i,state);state=copy.deepcopy(state);state['cursor']['round']=i;state['G']['v'].add_(1)
        bank=ArtifactStore(run/'B'/f'bank_{i}',new_base=base)
        bank.publish('completed',{'features.pt':tensor_bytes(torch.eye(2,4)),'identity.json':json_bytes({'round':i})},{'G':tree_sha(state['G'])})
        io=ArtifactStore(run/'round_io'/f'round_{i:04d}',new_base=base)
        action={'original':'caption','adversarial':'caption','K':0,'edits':[],'native_action':{'fixed_NOOP':True}}
        io.publish('rollout_0000',{'action.json':json_bytes(action)},{})
        ledger={'round':i,'G':[],'H':[],'conditions':[{'row':'r'}],'g_orders':[['r']],
            'bank':{'g_state_sha256':tree_sha(state['G'])}}
        ref=bank.root/'completed/manifest.json';j.commit(i,state,ledger,FileRef(str(ref),sha_file(ref)))
    first=read_ledger(run/'ledgers/round_0001/ledger.json')
    assert list(first['conditions'])[0]['G_rollout']==action and not (run/'B/bank_1').exists()
    assert sha_file(run/'ledgers/round_0001/bank_features.pt')==first['bank_features_sha256']
    assert torch.equal(torch.load(run/'ledgers/round_0001/bank_features.pt',map_location='cpu',weights_only=True),torch.eye(2,4))
    value,summary=j.load_latest(state['parents']);assert tree_sha(value)==tree_sha(state)

def test_attraction_Joint_engine_fixture(tmp_path):
    # 复用既有原生Engine数值fixture，注入本版本的评分入口。
    spec=importlib.util.spec_from_file_location('attraction_native_joint_fixture',ROOT/'tests/framework/test_gh_alternating_v2.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    from tbps.alternating import fresh_evaluation
    from tbps.attraction.evaluation import joint_score
    from unittest.mock import patch
    with patch.object(fresh_evaluation,'joint_score',joint_score):fixture.test_Joint_only_native_engine_fixture(tmp_path)

def test_full_plan_contains_six_predeclared_pairs_and_no_retry():
    import ast
    source=(ROOT/'operations/gh_attraction_v1_supervisor.py').read_text();ast.parse(source)
    assert 'for n in (10,20,30,40,50,60)' in source
    assert "expected=3 if n<60 else 0" in source and "FAIL_STOPPED_NO_RETRY" in source
    assert "audit_train_" in source and "audit_test_" in source


def test_final_report_six_results_SVG_and_current_state(tmp_path,monkeypatch):
    from tbps.attraction import report
    root=tmp_path/'repo';base=tmp_path/'tbps-framework-report-fixture';cfg=tmp_path/'configuration.json';cfg.write_text('{}')
    (root/'docs').mkdir(parents=True);(root/'docs/PROJECT_STATE.md').write_text('<!-- GH_ATTRACTION_CURRENT_BEGIN -->pending<!-- GH_ATTRACTION_CURRENT_END -->')
    f={'test_rounds':[10,20,30,40,50,60],'output_base':str(base),'test_output':str(base/'test'),
        'audits':{'train':str(base/'audit')},'attraction':asdict(Settings()),
        'report_runtime':json.loads((ROOT/'configs/framework/gh_attraction_v1.server.json').read_text())['report_runtime']}
    train=base/'audit/round_60/completed';train.mkdir(parents=True)
    (train/'report.json').write_bytes(json_bytes({'status':'GH_ATTRACTION_TRAIN_CPU_AUDIT_PASS','fresh_initialization':{'CPU_fixture':True}}))
    for n in f['test_rounds']:
        d=base/'test'/f'round_{n:02d}';(d/'cpu_audit/completed').mkdir(parents=True);(d/'completed').mkdir()
        metrics={'R@1':float(80-n),'R@5':float(90-n),'R@10':float(95-n),'mAP':float(70-n)}
        (d/'completed/summary.json').write_bytes(json_bytes({'metrics':metrics}))
        (d/'cpu_audit/completed/report.json').write_bytes(json_bytes({'status':'GH_ATTRACTION_JOINT_TEST_CPU_AUDIT_PASS',
            'scope':'official_test','metrics':metrics,'round':n,'gallery':3074,'query':6156,'CPU_fixture':True}))
    monkeypatch.setattr(report,'load',lambda *a:(f,{}));monkeypatch.setattr(report,'__file__',str(root/'src/tbps/attraction/report.py'))
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'mpl'))
    result=report.publish(cfg);out=root/'docs/framework/results/gh_attraction_v1'
    assert len(result['rows'])==6 and len(list(out.glob('*.svg')))==2 and len(list(out.glob('*.png')))==2
    for svg in out.glob('*.svg'):assert all(line==line.rstrip() for line in svg.read_text().splitlines())
    assert 'GH_ATTRACTION_V1.md' in (out/'REPORT.md').read_text()
    assert '六Joint Test完成' in (root/'docs/PROJECT_STATE.md').read_text()
