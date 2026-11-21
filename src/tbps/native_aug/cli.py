"""显式授权作业入口：两个训练进程，首节点后各自派生三个评测进程。"""
import argparse,sys
from .common import *
def preflight(c):
    import torch,types
    from importlib import import_module
    from transformers import BertTokenizerFast
    from .training import native_bundle
    from .native import Source
    n,bundle=native_bundle(c)
    q=read_json(Path(n['prototype']['history_root'])/'qualification/train.json')['entries']
    need(len(q)==2048 and [e['identity']['row_id'] for e in q]==n['prototype']['train_ids'],'Train 2048 membership/order')
    raw=json.loads(Path(n['prototype']['annotation']).read_text())
    for e in q:
        row=e['identity'];item=raw[int(row['row_id'].split(':')[1])]
        need(item['split']==row['split']=='train' and row['role']=='train' and item['id']==row['pid'] and item['captions'][row['caption_index']]==e['caption'],'official Train identity')
        checked(FileRef(str(Path(n['prototype']['image_root'])/row['image_rel_path']),row['image_sha256']))
    report={'configuration_sha256':sha_file(c['_config_path']),'status':'NATIVE_AUG_CPU_PREFLIGHT_PASS','models':{},'Train':2048,'fresh_initialization':bundle.report,'CUDA':False}
    for name in ('rasa','aptm'):
        mod=import_module('tbps.models.'+name)
        source=(mod.RasaSource if name=='rasa' else mod.APTMSource)(c['targets'][name]['path']);source.verify()
        with mod.native_modules(source) as (_,tok,utils):
            tokenizer=mod.official_tokenizer(tok,source)
            adapter=types.SimpleNamespace(tokenizer=tokenizer,pre_caption=utils.pre_caption)
            limit=50 if name=='rasa' else 56
            adapter.tokenize=lambda texts,ad=adapter,nn=name,ll=limit:ad.tokenizer([ad.pre_caption(t,ll) if nn=='rasa' else ad.pre_caption(t,ll,icfg_rstp=False,is_eda=False) for t in texts],padding='max_length',truncation=True,max_length=ll,return_tensors='pt')
            stub=types.SimpleNamespace(adapter=adapter,name=name,max_tokens=limit,fast=BertTokenizerFast.from_pretrained(source.data['bert'],local_files_only=True))
            stub.normalized=types.MethodType(Source.normalized,stub)
            for e in q:Source.aligned(stub,e['caption'])
            # 官方命名空间与旧H/G命名空间必须能同时装载，不构造受害者模型。
            from tbps.alternating.native_sources import native_session
            from .compat import source_session
            with source_session(n,bundle,c,name):pass
        report['models'][name]={'native_tokenizer_exact_rows':2048,'source':asdict(source.ref),'weight':asdict(source.checkpoint),'legacy_namespace_closure':True}
    from .recovery import validate_prepared_payload
    prepared=FileRef(**c['preparation_reuse']['rasa']['manifest'])
    checked(prepared)
    original=Path(prepared.path).parents[1]
    ArtifactStore(original,new_base=c['previous_output_base']).verify('completed')
    checked(FileRef(**c['previous_configuration']))
    old=read_json(c['previous_configuration']['path'])
    need(old['targets']['rasa']==c['targets']['rasa'],'same RaSa official target sources')
    prepared_report=read_json(original/'completed/report.json')
    from tbps.models.rasa import RasaSource
    need(prepared_report['source_model']=='rasa'
         and prepared_report['source_space']==asdict(RasaSource(c['targets']['rasa']['path']).space)
         and prepared_report['reader_epochs']==20 and prepared_report['train_count']==2048,
         'RaSa preparation identity')
    torch.set_num_threads(2)
    validate_prepared_payload(original,n,prepared_report)
    report['reused_preparation']=asdict(prepared)
    report['expected_G_updates_per_round']=prepared_report['expected_G_updates_per_round']
    need(not torch.cuda.is_initialized(),'CPU preflight touched CUDA')
    d=CONTROL/'preflight';d.mkdir(parents=True,exist_ok=True);(d/'report.json').write_bytes(json_bytes(report));print(json.dumps(report,ensure_ascii=False),flush=True)
    return report

def check_launch_gate(c):
    receipt = read_json(CONTROL / 'preflight/report.json')
    need(receipt['status'] == 'NATIVE_AUG_CPU_PREFLIGHT_PASS'
         and receipt['configuration_sha256'] == sha_file(c['_config_path']),
         'matching CPU preflight gate')
    proof = read_json(CONTROL / 'cpu_probes/rasa/report.json')
    need(proof['status'] == 'NATIVE_AUG_REAL_CPU_BACKEND_PROBE_PASS'
         and proof['source_configuration'] == c['targets']['rasa'],
         'real RaSa CPU backend gate')
    for source in proof['tested_sources']:
        checked(FileRef(**source))
    need(set(p['path'] for p in proof['tested_sources']) == {
        str(ROOT / 'src/tbps/native_aug' / n)
        for n in ('native.py', 'compat.py', 'guard.py', 'startup.py')},
        'probe bound to this worktree')

def launch(c):
    check_launch_gate(c)
    need(not (OUTPUT / 'rasa').exists(), 'source Attempt already consumed')
    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / 'launch.json').open('xb') as receipt, (CONTROL / 'supervisor.log').open('xb') as log:
        process = subprocess.Popen(command('pipeline'), cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES':'0', 'PYTHONUNBUFFERED':'1',
                 'TOKENIZERS_PARALLELISM':'false', 'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
        receipt.write(json_bytes({'pid':process.pid,'configuration_sha256':sha_file(c['_config_path'])}))
    print('STARTED_RASA_PIPELINE', process.pid, flush=True)

def stage(c, phase, job, *args):
    d = CONTROL / job
    d.mkdir(parents=True, exist_ok=True)
    state('pipeline', phase, active_job=job, arguments=list(args))
    with (d / 'launcher.log').open('xb') as log:
        p = subprocess.Popen(command(*args), cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT)
        (d / 'started.json').write_bytes(json_bytes({'pid':p.pid,'args':list(args),
                           'configuration_sha256':sha_file(c['_config_path'])}))
        code = p.wait()
    (d / 'exit.json').write_bytes(json_bytes({'returncode':code}))
    need(code == 0, f'{job}: child exit {code}')

def pipeline(c):
    check_launch_gate(c)
    stage(c, 'TRAINING_RASA', 'train_rasa', 'train', 'rasa')
    report = read_json(CONTROL / 'train_rasa/status.json')
    need(report['phase'] == 'TRAINING_COMPLETE' and report['round'] == 60,
         '60-round training completion required')
    from .evaluation import checkpoint, attack_root
    for r in c['execution']['evaluation_order']:
        audited = OUTPUT / 'rasa' / 'train' / f'round_{r:02d}' / 'cpu_audit'
        ArtifactStore(audited,new_base=OUTPUT).verify('completed')
        need(read_json(audited / 'completed/report.json')['status'] == 'NATIVE_AUG_ROUND_CPU_AUDIT_PASS',
             'training audit gate')
        cp=checkpoint('rasa',r)
        ArtifactStore(cp.parent,new_base=OUTPUT).verify(cp.name)
        stage(c, 'GENERATING_SHARED_INPUTS', f'generate_{r:02d}', 'generate', 'rasa', r)
        proof = attack_root('rasa',r) / 'cpu_audit'
        ArtifactStore(proof,new_base=OUTPUT).verify('completed')
        need(read_json(proof / 'completed/report.json')['status'] == 'NATIVE_AUG_INPUT_CPU_AUDIT_PASS',
             'shared input audit gate')
        for target in c['evaluation_targets']:
            stage(c, 'EVALUATING_JOINT', f'eval_rasa_to_{target}/round_{r:02d}',
                  'evaluate-round', 'rasa', target, r)
    from .rasa_report import summarize
    summarize(c)
    state('pipeline','SUCCESS',source='rasa',completed_rounds=list(ROUNDS),audited_joint_results=18)

def status():
    def alive(pid):
        try:
            stat=(Path('/proc')/str(pid)/'stat').read_text()
            return stat.rsplit(')',1)[1].split()[0] != 'Z'
        except OSError:
            return False
    for job in ('pipeline','train_rasa'):
        p=CONTROL/job/'status.json'
        if p.exists():
            value=read_json(p)
            print(json.dumps({**value,'process_alive':alive(value['pid'])},ensure_ascii=False))
            if job=='pipeline' and value.get('active_job'):
                source=CONTROL/value['active_job']/'status.json'
                if source.exists() and source!=CONTROL/'train_rasa/status.json':
                    d=read_json(source)
                    print(json.dumps({**d,'process_alive':alive(d['pid'])},ensure_ascii=False))
    for p in sorted((OUTPUT/'rasa'/'evaluation').glob('*/round_*/cpu_audit/completed/report.json')):
        d=read_json(p)
        print('AUDITED_RESULT',json.dumps({k:d[k] for k in ('round','source','target','metrics')},ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','launch','train','audit','evaluate','audit-inputs','audit-evaluation','probe-cpu','status','pipeline','generate','evaluate-round'])
    ap.add_argument('args',nargs='*');a=ap.parse_args()
    if a.action=='status':return status()
    c=config();job=a.action+'_'+'_'.join(a.args)
    if a.action=='train':job='train_'+a.args[0]
    if a.action=='pipeline':job='pipeline'
    if a.action=='generate':job=f'generate_{int(a.args[1]):02d}'
    if a.action=='evaluate-round':job=f'eval_rasa_to_{a.args[1]}/round_{int(a.args[2]):02d}'
    if a.action=='evaluate':job='eval_'+a.args[0]+'_to_'+a.args[1]
    try:
        if a.action=='preflight':return preflight(c)
        if a.action=='launch':return launch(c)
        if a.action=='pipeline':return pipeline(c)
        if a.action in ('train','generate','evaluate-round'):
            need(a.args[0]=='rasa','this experiment authorizes only RaSa source')
        if a.action=='generate':
            from .evaluation import make_attack
            return make_attack(c,'rasa',int(a.args[1]),job)
        if a.action=='evaluate-round':
            from .evaluation import evaluate_one
            return evaluate_one(c,'rasa',a.args[1],int(a.args[2]),job)
        if a.action=='probe-cpu':
            from .probe import run
            return run(c,a.args[0])
        if a.action=='train':
            from .training import run
            return run(c,a.args[0])
        if a.action=='audit':
            from .audit import run
            return run(c,a.args[0],int(a.args[1]))
        if a.action=='evaluate':
            from .evaluation import worker
            return worker(c,*a.args)
        if a.action=='audit-inputs':
            from .evaluation_audit import audit_inputs
            return audit_inputs(c,a.args[0],int(a.args[1]))
        if a.action=='audit-evaluation':
            from .evaluation_audit import audit_evaluation
            return audit_evaluation(c,a.args[0],a.args[1],int(a.args[2]))
    except BaseException:
        failed(job);raise
