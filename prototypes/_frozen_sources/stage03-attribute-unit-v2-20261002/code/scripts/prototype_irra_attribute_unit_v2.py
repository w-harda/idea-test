#!/usr/bin/env python3
"""固定9-caption、第8层、fp32、uniform normalized pooling验证。"""
from __future__ import annotations
import argparse,importlib,json,os,sys
from pathlib import Path
from functools import partial
sys.dont_write_bytecode=True
os.environ.setdefault('TORCH_HOME','/home/lzf/ldx/cache/torch')
os.environ.setdefault('HF_HOME','/home/lzf/ldx/cache/huggingface')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from attributes.ap_gallery_baseline import atomic_json,sha256,write_path
from attributes.irra_paired_rank_drop import load_cuhk_paired_split
from attributes.irra_victim import FrozenIRRA
from attributes.irra_attribute_unit_v2_prototype import AttributeUnitV2Probe


def run(args):
    output=write_path(args.output)
    if output.exists(): raise FileExistsError(output)
    prior=json.loads(args.reference.read_text(encoding='utf-8'))
    assert sha256(args.reference)=='2ad1d49e80575d48c2974959d13a0acd994c6f4e16adeed764e47c9c03616a75'
    previous={q['row_id']:q for q in prior['results'] if q['precision']=='fp32_control'}
    protocol=load_cuhk_paired_split(args.annotation,split='train')
    payload=torch.load(args.gallery_cache,map_location='cpu',weights_only=True)
    spec=payload['spec']; root=Path(__file__).resolve().parents[1]
    assert spec==prior['gallery_spec'] and sha256(args.gallery_cache)==prior['gallery_cache_sha256']
    assert spec['gallery_paths']==list(protocol.gallery_paths) and len(protocol.gallery_paths)==34054
    assert spec['annotation_sha256']==sha256(args.annotation)
    assert spec['checkpoint_sha256']==sha256(args.checkpoint)
    assert spec['config_sha256']==sha256(args.config)
    assert spec['torch_version']==str(torch.__version__)
    assert spec['encoder_adapter_sha256']==sha256(root/'src/attributes/irra_victim.py')
    for name,digest in spec['source_sha256'].items(): assert digest==sha256(args.irra_repo/name)
    assert len(prior['row_ids'])==9
    queries={q.row_id:q for q in protocol.captions if q.row_id in previous}
    assert set(queries)==set(previous)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(args.irra_repo.resolve()))
    official=importlib.import_module('model.build')
    official.build_CLIP_from_openai_pretrained=partial(official.build_CLIP_from_openai_pretrained,
                         download_root=str(write_path(Path('/home/lzf/ldx/cache/clip'))))
    victim=FrozenIRRA(args.irra_repo,args.checkpoint,args.config,device='cuda').eval()
    victim.model.float()
    gallery=payload['features'].to('cuda')
    results=[]
    for row_id in prior['row_ids']:
        query=queries[row_id]; old=previous[row_id]
        probe=AttributeUnitV2Probe(victim,gallery,query.paired_gallery_index,tau=.02)
        result=probe.run(query.caption,layer=8,epsilons=(.01,.05,.1))
        assert result['attributes']==old['attributes'] and result['provenance']==old['provenance']
        assert result['F']==old['F'],'baseline F must match previous fp32 control exactly'
        by_slot={r['slot']:r for r in old['attribute_results']}
        for attr in result['attribute_results']:
            previous_attr=by_slot[attr['slot']]
            assert attr['support_token_indices']==previous_attr['support_token_indices']
            assert abs(attr['S_old']-previous_attr['S'])<=max(1e-8,abs(previous_attr['S'])*1e-6)
            attr['previous_fp32_S_old']=previous_attr['S']
            assert attr['direction_degrees_of_freedom']==512
            assert attr['S_new']<=attr['S_old']*(1+1e-6)
            for r in attr['interventions']:
                assert r['effective_budget_relative_error']<1e-4
        result.update(row_id=row_id,paired_image=query.image,paired_gallery_index=query.paired_gallery_index)
        results.append(result)
        print(f'{row_id} old_order={result["rankings"]["S_old"]} new_order={result["rankings"]["S_new"]}',flush=True)
        for a in result['attribute_results']:
            print(f'  {a["slot"]}={a["canonical"]} k={a["k"]} old={a["S_old"]:.9g} new={a["S_new"]:.9g}',flush=True)
    report={'prototype_only':True,'objective':'paired_soft_rank','tau':.02,'tau_status':'prototype, no search',
            'query_count':9,'attribute_count':sum(len(r['attribute_results']) for r in results),
            'gallery_count':34054,'gallery_cache_sha256':sha256(args.gallery_cache),
            'gallery_spec':spec,'row_ids':prior['row_ids'],'layer':8,'precision':'fp32',
            'precision_policy':'same loaded frozen weights promoted to float32; previous fixed image cache unchanged',
            'epsilons':[.01,.05,.1],'support_policy':prior['support_policy'],
            'pooling':'uniform 1/sqrt(k) only on support, no learned/manual weights',
            'direction_dof':512,'image_encodings':0,'new_removal_evaluations':0,
            'reference_sha256':sha256(args.reference),
            'code_sha256':{name:sha256(root/name) for name in
                ['src/attributes/irra_attribute_unit_v2_prototype.py','scripts/prototype_irra_attribute_unit_v2.py',
                 'tests/test_irra_attribute_unit_v2_prototype.py','src/attributes/irra_attribute_unit_prototype.py',
                 'src/attributes/irra_token_alignment.py']},'results':results}
    atomic_json(output,report)
    print(f'report: {output}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('annotation','irra-repo','checkpoint','config','gallery-cache','reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
