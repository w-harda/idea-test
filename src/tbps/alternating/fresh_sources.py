"""v2独立来源与seed42初始化；不允许加载任何历史G/H训练状态。"""
from pathlib import Path
from types import SimpleNamespace
import copy
from tbps.artifacts import read_json,sha_file,checked,FileRef,digest
from .contracts import require
from .native_sources import native_session,bindings,strict_import_preflight,source_key
from .parents import cpu_restore_only,ParentBundle
from .state import snapshot,capture_rng,clone_tree,tree_sha

SCHEMA='gh-alternating-v2'

def load(path,scope):
    f=read_json(path,SCHEMA);require(scope in ('smoke','train','test'),'scope')
    require(f['rounds']==60 and f['test_rounds']==[10,20,30,40,50,60] and f['seed']==42,'fixed sixty/six endpoints')
    require(f['text_budget']=='original Natural-K; one Unicode character per original native word'
        and f['initialization']=='fresh G/H; empty Adam; no historical dynamic bank','v2 method')
    require(f['g_passes_per_round']==f['h_passes_per_round']==1 and f['checkpoint_interval']==10
        and f['g_learning_rate']==.0002 and f['h_learning_rate']==.0001
        and f['image_epsilon']==8/255 and f['precision']=='fp32/math/no_tf32','fixed original training settings')
    require(f['g_scheduler'] is None and f['h_scheduler'] is None,'original scripts have no Scheduler')
    root=Path(__file__).resolve().parents[3]
    require(f['producers']=={str(p):sha_file(p) for p in (root/'src/tbps').rglob('*.py')},'complete current source SHA')
    for p,h in f['tools'].items():checked(FileRef(p,h))
    for value in f['fixed_resources'].values():checked(FileRef(**value))
    for p,h in f['runtime_pins'].items():checked(FileRef(p,h))
    old=read_json(f['v1_configuration']['path']);checked(FileRef(**f['v1_configuration']))
    oldroot=Path(f['v1_configuration']['path']).parents[2]
    for p,h in old['producers'].items():require(sha_file(root/Path(p).relative_to(oldroot))==h,'frozen v1 source changed')
    c=copy.deepcopy(f['native_template'])
    c.update(producers=f['producers'],runtime_pins=f['runtime_pins'],output_base=f['output_base'])
    c['managed_roots']=[str(root/'src') if x.endswith('/src') and 'idea-test-tbps-gh-' in x else x for x in c['managed_roots']]
    p=c['prototype'];p['samples']=f['smoke_rows'] if scope=='smoke' else p['train_ids']
    p['rounds']=1 if scope=='smoke' else 60;p['output_base']=f['outputs']['train' if scope=='test' else scope]
    c['smoke_output']=p['output_base'];c['formal_binding']={'scope':scope}
    gm={e['row_id']:e for e in read_json(p['g_manifest'])['entries'] if e['role']=='train'}
    q=read_json(Path(p['history_root'])/'qualification/train.json')['entries']
    eligible={e['identity']['row_id']:e['eligible'] for e in q}
    k=sum(gm[row]['K'] for row in p['samples']);n=len(p['samples']);rounds=p['rounds']
    c['smoke']={'max_G_forwards':rounds*(k+n+2048),'max_IRRA_images':rounds*(k+n+2048),
        'max_H_forwards':rounds*3*sum(eligible[row] for row in p['samples']),
        'max_IRRA_texts':rounds*(10*k+20*n),'max_reader_forwards':rounds*n,'max_JVPs':rounds*k}
    c['condition_space_sha256']=digest({'source':p['source_space_sha256'],'reader':c['sources']['reader'],
        'basis':c['sources']['basis'],'mapping':c['byte_alignment_policy'],'producers':f['producers']})
    c['_config_path']=str(Path(path).resolve());c['fresh_config_sha256']=sha_file(path)
    c['training_configuration']={'g_passes_per_round':1,'h_passes_per_round':1,'checkpoint_interval_rounds':10,**{k:f[k] for k in ('rounds','seed','g_scheduler','h_scheduler','text_budget','initialization')}}
    require(not {'G60','H10','old_G60_bank_audit'} & set(c['sources']),'historical trained sources forbidden')
    return f,c

class Grant:
    action='smoke'
    def __init__(self,c,allow):
        require(allow is True,'explicit GPU capability required');self.key=source_key(c)
    def validate(self,c,action):require(action=='smoke' and source_key(c)==self.key,'v2 capability/source mismatch')

def fresh_bundle(f,c):
    import torch,random,numpy as np
    with native_session(c) as session:
        with cpu_restore_only(()) as events:
            b=bindings(session)
            gc=read_json(f['fixed_resources']['G_original_config']['path'])
            hc=read_json(f['fixed_resources']['H_original_config']['path'])
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            g=b.get('image_only_generator').ImageOnlyGenerator(**gc['architecture'])
            ga=torch.optim.Adam(g.parameters(),**gc['optimizer'])
            sampler=torch.Generator().manual_seed(42);gr=capture_rng(sampler)
            tokenizer=b.get('utils.simple_tokenizer').SimpleTokenizer(bpe_path=c['sources']['bpe']['path'])
            probe=b.get('tokenizer_probe').TokenizerProbe(tokenizer)
            training=session.import_module('formal_training.training')
            h=training.fresh_policy(probe,{'seed':42,'architecture':hc['architecture']})
            ha=b.get('phase4_training').optimizer_for(h,hc);hr=capture_rng(sampler)
            require(type(h) is b.get('policy').CharacterPolicy and not getattr(h,'word_budget1',False),'original uncapped H only')
            require(not ga.state_dict()['state'] and not ha.state_dict()['state'],'fresh empty Adam required')
            parents={'G':tree_sha(g.state_dict()),'H':tree_sha(h.state_dict()),'protocol':digest(c['prototype'])}
            state=snapshot(g,h,ga,ha,g_rng=gr,h_rng=hr,cursor={'round':0,'g_updates':0,'h_visits':0,'h_updates':0},parents=parents)
            frontend=b.get('text_frontend').CaptionFrontend()
            report={'status':'FRESH_G_H_SEED42_PASS','initial_G':parents['G'],'initial_H':parents['H'],
                'empty_G_Adam':True,'empty_H_Adam':True,'trained_checkpoint_loads':len(events['loads']),
                'historical_dynamic_bank_loaded':False,'word_budget1_installed':False,'Scheduler':None}
            bundle=ParentBundle(g,h,ga,ha,sampler,state,b,{},report,probe,frontend)
    bundle.modules=session.modules
    require(report['trained_checkpoint_loads']==0,'historical model load forbidden')
    return bundle

def preflight(path,scope):
    f,c=load(path,scope)
    # 原缓存仅作固定S_raw/K/JVP基准；不读取G/H训练权重或旧对抗Bank。
    gm=read_json(c['prototype']['g_manifest'])['entries'];gm={e['row_id']:e for e in gm if e['role']=='train'}
    qualification=read_json(Path(c['prototype']['history_root'])/'qualification/train.json')['entries']
    ids=[e['identity']['row_id'] for e in qualification]
    require(ids==c['prototype']['train_ids'] and set(ids)==set(gm) and len(ids)==2048,'fixed Train membership/order')
    from .contracts import Anchor
    import json
    annotation=json.loads(Path(c['prototype']['annotation']).read_text())
    total=0
    for e in qualification:
        row=e['identity']['row_id'];g=gm[row]
        require(e['identity']['split']=='train' and e['identity']['role']=='train' and e['caption']==g['caption']
            and e['identity']['pid']==g['pid'],'train-only binding')
        raw=annotation[int(row.split(':')[1])]
        require(raw['split']=='train' and raw['id']==g['pid'] and raw['file_path']==g['image_rel_path']
            and raw['captions'][g['caption_index']]==g['caption'],'official annotation identity')
        checked(FileRef(g['tensor_path'],g['tensor_sha256']));checked(FileRef(g['metadata_path'],g['metadata_sha256']))
        md=read_json(g['metadata_path']);attrs=[Anchor(**a) for a in e['attributes']]
        for a in md['source_selection']['selected_attributes']:
            anchor=Anchor(**{k:a[k] for k in Anchor.__dataclass_fields__})
            require(sum(anchor==b for b in attrs)==1,'canonical ambiguity')
        checked(FileRef(str(Path(c['prototype']['image_root'])/e['identity']['image_rel_path']),e['identity']['image_sha256']))
        total+=g['K']
    require(total==6020,'original G K accounting')
    from tbps.materialization.irra_startup import CPUStartupGate
    gate=CPUStartupGate();imports=strict_import_preflight(c);bundle=fresh_bundle(f,c)
    gate.checked();gate.exit_checked()
    report={'status':'GH_V2_CPU_PREFLIGHT_PASS','scope':scope,'members':2048,'G_updates_per_round':8068,
        'H_updates_per_round':sum(e['eligible'] for e in qualification),'initialization':bundle.report,'imports':imports,
        'Validation_updates':0,'Test_feedback':False,'original_order':'fixed ordered Train manifest each G/H pass'}
    return f,c,bundle,gate,report
