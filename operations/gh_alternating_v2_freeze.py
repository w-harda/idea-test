"""新v2配置仅在未消费Smoke/Train/Test时冻结；历史G/H不是固定资源。"""
from pathlib import Path
import sys,argparse,copy,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tbps.artifacts import read_json,sha_file,json_bytes

def freeze(rebind=False):
    out='/home/lzf/ldx/outputs/tbps-framework-gh-alternating-v2-20261107'
    require=__import__('tbps.alternating.contracts',fromlist=['require']).require
    require(not any(Path(out,k).exists() for k in ('smoke','train','test')),'Attempt consumed; no source rebinding')
    cfg=ROOT/'configs/framework/gh_alternating_v2.server.json'
    if cfg.exists():
        require(rebind,'explicit unconsumed rebind required')
        archive=Path('/home/lzf/ldx/tmp/gh_alternating_v2_cpu_20261107/config_revisions')
        d=archive/('v'+str(len(list(archive.glob('v*')))+1));d.mkdir(parents=True);__import__('shutil').copy2(cfg,d/cfg.name)
    oldpath=Path('/home/lzf/ldx/projects/idea-test-tbps-gh-alternating-v1-20261106/configs/framework/gh_alternating_v1.server.json')
    old=read_json(oldpath);native=read_json(old['native_configs']['train']);p=native['prototype']
    # 只取后端所需的固定资源/来源，完全移除历史Parent、恢复游标、旧G60图库。
    keys=('reader','reader_split','basis','dataset_config','bpe')
    sources={k:native['sources'][k] for k in keys}
    def ref(path):return {'path':str(path),'sha256':sha_file(path)}
    groot=Path('/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014')
    hroot=Path('/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015')
    resources={**sources,'G_original_config':ref(groot/'config_phase26.json'),'H_original_config':ref(hroot/'config_phase4.json')}
    history=Path(p['history_root'])
    resources.update(G_manifest=ref(p['g_manifest']),G_condition_protocol=ref(Path(p['g_manifest']).with_name('protocol.json')),
        train_qualification=ref(history/'qualification/train.json'),clean_protocol=ref(history/'gallery_train/protocol.json'),
        clean_completed=ref(history/'gallery_train/completed.json'),annotation=ref(p['annotation']))
    template={k:copy.deepcopy(native[k]) for k in ('runtime','byte_alignment_policy','import_paths','managed_roots')}
    template['sources']=sources
    template['prototype']={k:copy.deepcopy(p[k]) for k in ('train_ids','g_manifest','history_root','image_root','annotation','source_space_sha256','clean_gallery_sha256')}
    template['prototype']['schema']='gh-alternating-v2-protocol-1'
    f={'schema':'gh-alternating-v2','name':'GH-Alternating-v2','rounds':60,'test_rounds':[10,20,30,40,50,60],
        'seed':42,'initialization':'fresh G/H; empty Adam; no historical dynamic bank',
        'text_budget':'original Natural-K; one Unicode character per original native word',
        'g_scheduler':None,'h_scheduler':None,'g_learning_rate':.0002,'h_learning_rate':.0001,
        'g_passes_per_round':1,'h_passes_per_round':1,'data_order':'fixed ordered 2048 Train manifest; no Validation',
        'checkpoint_interval':10,'precision':'fp32/math/no_tf32','image_epsilon':8/255,
        'outputs':{k:out+'/'+k for k in ('smoke','train')},'output_base':out,'test_output':out+'/test',
        'controls':{k:'/home/lzf/ldx/tmp/gh_alternating_v2_'+k+'_20261107' for k in ('smoke','train')},
        'audits':{k:out+'/'+k+'_cpu_audit' for k in ('smoke','train')},
        'smoke_rows':read_json(old['references']['formal_parent']['path'])['pilot']['rows'],
        'resources':{k:{'minimum_ram_bytes':16*1024**3,'minimum_disk_bytes':100*1024**3,'minimum_free_inodes':300000} for k in ('smoke','train')},
        'resource_probe_directory':'/home/lzf/ldx','GPU_policy':'shared_observe_only_no_VRAM_floor',
        'fixed_resources':resources,'runtime_pins':native['runtime_pins'],'native_template':template,
        'v1_configuration':ref(oldpath),'producers':{str(x):sha_file(x) for x in (ROOT/'src/tbps').rglob('*.py')},
        'tools':{str(x):sha_file(x) for x in (ROOT/'operations').glob('gh_alternating_v2*.py')},
        'authorization':{'short_Smoke':True,'train_after_new_Smoke_audit':True,'six_Joint_Tests_after_train_audit':True,
            'extra_Validation':False,'additional_victims_or_datasets':False,'retry_real_failure':False}}
    cfg.write_bytes(json_bytes(f));print(str(cfg),sha_file(cfg))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--rebind-unconsumed',action='store_true');a=ap.parse_args();freeze(a.rebind_unconsumed)
