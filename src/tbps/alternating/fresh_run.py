"""v2执行生命周期；真实失败保留停止，正式训练仅由新Smoke审计放行。"""
import argparse,json,os,signal,time,traceback,fcntl
from pathlib import Path
from contextlib import contextmanager
from tbps.artifacts import ArtifactStore,read_json,json_bytes,sha_file,FileRef,checked
from .contracts import require
from .fresh_sources import load,preflight,Grant
from .fresh_store import FreshJournal
from .state import tree_sha
from .configured_cli import resources

@contextmanager
def no_historical_weights(base):
    import torch
    native=torch.load;events=[]
    banned=('/attribute_image_generator_v2_20261014/outputs/train_v2/', '/outputs/tbps-framework-gv2-epoch','/attribute_text_policy_h_v1_20261015/outputs/formal_v1/training/',
        '/outputs/tbps-framework-hv1-g60-','/outputs/tbps-framework-gh-formal-b/','/outputs/tbps-framework-gh-alternating-v1-')
    def load_file(path,*a,**kw):
        if isinstance(path,(str,Path)):
            p=str(Path(path).resolve())
            require(p.startswith(str(Path(base).resolve())+'/') or not any(x in p for x in banned),'historical G/H state load forbidden: '+p)
            events.append(p)
        return native(path,*a,**kw)
    torch.load=load_file
    try:yield events
    finally:torch.load=native

def dynamic_budget_probe(bundle,c):
    """固定Train公开输入、独立采样器，验证原策略允许多词；不使用奖励选候选。"""
    import torch
    entries=read_json(Path(c['prototype']['history_root'])/'qualification/train.json')['entries']
    from .native_sources import native_session
    from .parents import no_optimizer_steps
    before=tree_sha(bundle.h.state_dict())
    with native_session(c,reuse=bundle.modules) as s,no_optimizer_steps(),torch.no_grad():
        data=s.import_module('formal_training.data');constraints=s.import_module('constraints');contracts=s.import_module('contracts')
        for e in entries[:64]:
            if not e['eligible'] or len(e['attributes'])<2:continue
            example=data.example_from(e)
            for n in range(64):
                out=bundle.h.rollout(example,'sample',torch.Generator().manual_seed(1000+n))
                require(constraints.replay(example,out.edits,out.budget_k)==out.adversarial,'original Dynamic-K replay')
                words=[contracts.word_at(example.caption,x.position) for x in out.edits]
                require(len(words)==len(set(words)) and len(words)<=out.budget_k,'original one-character/shared-word budget')
                if len(words)>=2:
                    require(before==tree_sha(bundle.h.state_dict()),'budget probe changed H')
                    bundle.modules=s.modules
                    return {'status':'ORIGINAL_DYNAMIC_K_MULTIWORD_PASS','row_id':e['identity']['row_id'],
                        'independent_sampler_seed':1000+n,'K':out.budget_k,'changed_word_positions':words,'action':out.record(),
                        'not_training_or_Test_selection':True}
    raise ValueError('fresh original policy multiword capability not demonstrated')

def execute(path,scope,allow=False,resume=False):
    require(allow,'explicit GPU capability required')
    f,c,bundle,gate,pre=preflight(path,scope)
    root=Path(f['outputs'][scope]);control=Path(f['controls'][scope]);control.mkdir(parents=True,exist_ok=True)
    if scope=='train':
        proof=read_json(Path(f['audits']['smoke'])/'completed/report.json')
        require(proof['status']=='GH_V2_SMOKE_CPU_AUDIT_PASS' and proof['configuration_sha256']==sha_file(path),'new matching Smoke audit required')
    require(resume or not root.exists(),'Attempt consumed; no automatic retry')
    with (control/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        snap=resources(f,scope);require(snap['resource_eligible'],'RAM/disk/inode insufficient')
        journal=FreshJournal(root,sha_file(path),f['output_base']);previous=None;parent=None
        if resume:
            parent,previous=journal.load_latest(bundle.state['parents'])
            from .configured_audit import verify_bank
            verify_bank(root,previous,c)
        store=ArtifactStore(root,new_base=f['output_base'])
        if not resume:store.publish('claim',{'preflight.json':json_bytes(pre),'resources.json':json_bytes(snap)},
            {'configuration':sha_file(path),'scope':scope,'fresh':True})
        identity={'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgid(0),'sid':os.getsid(0),'tty':os.ttyname(0) if os.isatty(0) else None}
        (control/('process_resume.json' if resume else 'process_initial.json')).write_bytes(json_bytes(identity))
        stop=[False]
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
        start=time.monotonic()
        try:
            import torch
            from tbps.models.irra import fp32_runtime
            from .fresh_training import loaded
            from .configured_training import run_rounds
            from .formal_state import tensor_bytes
            torch.set_num_threads(2)
            if scope=='smoke':
                proof=dynamic_budget_probe(bundle,c)
                store.publish('dynamic_budget_probe',{'report.json':json_bytes(proof)},{'fresh_H':bundle.report['initial_H']})
            gate.initialize_cuda()
            with no_historical_weights(f['output_base']) as load_events,fp32_runtime(),loaded(c,bundle,Grant(c,True)) as backend:
                backend.load_reward_inputs();backend.resume_ledger=previous
                def progress(label,index,visited,total):
                    data={'phase':label,'round':index,'visited':visited,'total':total,'cursor':dict(backend.cursor),
                        'counts':dict(backend.counts),'elapsed_seconds':time.monotonic()-start,'resources':resources(f,scope),
                        'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),**identity}
                    (control/'status.json').write_bytes(json_bytes(data))
                    with (control/'resources.jsonl').open('a') as out:out.write(json.dumps(data,ensure_ascii=False)+'\n')
                    print('GH_V2_STAGE',label,index,visited,total,backend.cursor,flush=True)
                backend.progress=progress
                if parent is None:
                    parent=backend.complete_state(0)
                    require(parent['cursor']=={'round':0,'g_updates':0,'h_visits':0,'h_updates':0}
                        and not parent['G_Adam']['state'] and not parent['H_Adam']['state'],'fresh start proof')
                    journal.initialize({'configuration':sha_file(path),'initialization':bundle.report},parent)
                    store.publish('initial_state',{'state.pt':tensor_bytes(parent),'proof.json':json_bytes(bundle.report)},
                        {'state_sha':tree_sha(parent),'fresh':True})
                state,complete=run_rounds(backend,journal,parent,c['prototype']['rounds'],start=parent['cursor']['round'],stop_requested=lambda:stop[0])
                result={'status':'GH_V2_GENERATED' if complete else 'STOPPED_COMMITTED_BOUNDARY','complete':complete,'scope':scope,
                    'configuration_sha256':sha_file(path),'cursor':state['cursor'],'frozen_sources':backend.final_frozen(),
                    'counts':dict(backend.counts),'elapsed_seconds':time.monotonic()-start,
                    'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
                    'historical_G_H_loads':0,'Scheduler':None,'resources':resources(f,scope),'weight_load_event_count':len(load_events)}
            load(path,scope)
            require(sha_file(path)==c['fresh_config_sha256'],'configuration changed during execution')
            store.publish('generated' if complete else 'stopped_'+str(state['cursor']['round']),{'report.json':json_bytes(result)},
                {'configuration':sha_file(path),'scope':scope})
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True);return complete
        except BaseException:
            error=traceback.format_exc();name='FAIL_resume' if resume else 'FAIL_initial'
            with (root/name).open('x') as out:out.write(error)
            (control/'status.json').write_bytes(json_bytes({'phase':'FAIL','error':error,**identity}));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    ap.add_argument('--action',choices=('preflight','smoke','train','audit','test','test-smoke','test-audit','test-smoke-audit','report'),default='preflight')
    ap.add_argument('--scope',choices=('smoke','train'),default='smoke');ap.add_argument('--allow-gpu',action='store_true')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--round',type=int);a=ap.parse_args()
    if a.action in ('smoke','train'):
        if not execute(a.config,a.action,a.allow_gpu,a.resume):raise SystemExit(3)
    elif a.action=='preflight':
        f,c,b,g,p=preflight(a.config,a.scope);print(json.dumps(p,ensure_ascii=False,indent=2));print(p['status'])
    elif a.action=='audit':
        from .fresh_audit import audit
        print(json.dumps(audit(a.config,a.scope),ensure_ascii=False,indent=2))
    elif a.action in ('test-audit','test-smoke-audit'):
        from .fresh_audit import audit_test
        print(json.dumps(audit_test(a.config,a.round,smoke=a.action=='test-smoke-audit'),ensure_ascii=False,indent=2))
    elif a.action in ('test','test-smoke'):
        from .fresh_evaluation import evaluate
        print(json.dumps(evaluate(a.config,a.round,smoke=a.action=='test-smoke',allow=a.allow_gpu),ensure_ascii=False,indent=2))
    else:
        from .fresh_report import publish
        print(publish(a.config))
