"""显式复用已验证准备产物；不覆盖旧文件，不读取旧G/H诊断权重。"""
import math
import numpy as np
import torch
from .common import *
def completed_preparation(c,n,bundle,source,name,root):
    item=c['preparation_reuse'].get(name)
    if not item or item['kind']!='completed':return None
    checked(FileRef(**item['manifest']))
    old=Path(item['manifest']['path']).parents[1]
    ArtifactStore(old,new_base=c['previous_output_base']).verify('completed')
    checked(FileRef(**c['previous_configuration']))
    previous=read_json(c['previous_configuration']['path'])
    need(previous['targets'][name]==c['targets'][name],'same official target source')
    original=read_json(old/'completed/report.json')
    need(original['source_model']==name and original['source_space']==asdict(source.adapter.space)
        and original['reader_epochs']==20 and original['gallery_count']==34054 and original['train_count']==2048,'completed native preparation identity')
    manifest=read_json(old/'completed/manifest.json')
    need(set(manifest['files_sha256'])=={'reader.pt','bases.json','selection.json','candidates.json','clean.pt','gallery.pt','gallery_sources.json','report.json'},'complete preparation payload set')
    validate_prepared_payload(old,n,original)
    files={key:(old/'completed'/key).read_bytes() for key in manifest['files_sha256'] if key!='report.json'}
    report={**original,'reuse_evidence':{'original_manifest':item['manifest'],'previous_configuration':c['previous_configuration'],
        'new_configuration':asdict(ref(c['_config_path'])),'prepared_bytes_preserved':True,'G_H_weights_reused':False}}
    files['report.json']=json_bytes(report)
    publish(root,'completed',files,{'source_model':name,'reused_preparation':True},sources=(FileRef(**item['manifest']),ref(c['_config_path'])))
    split=read_json(n['sources']['reader_split']['path'])
    reader=bundle.bindings.get('learned_reader').SlotReader([len(v) for v in split['values']],dim=source.hidden_dim)
    reader.load_state_dict(torch.load(root/'completed/reader.pt',map_location='cpu',weights_only=True),strict=True)
    reader=reader.cuda().eval().requires_grad_(False);reader.zero_grad(set_to_none=True)
    bases=read_json(root/'completed/bases.json');plans=read_json(root/'completed/selection.json')
    clean=torch.load(root/'completed/clean.pt',map_location='cpu',weights_only=True).clone()
    entries=read_json(Path(n['prototype']['history_root'])/'qualification/train.json')['entries']
    need(clean.shape==(2048,source.feature_dim) and set(plans)==set(n['prototype']['train_ids']),'native dimensions/membership')
    source.check_frozen(final=True)
    return reader,split,bases,plans,clean,entries
def verified_prefix(c,n,bundle,source,name,root):
    item=c['preparation_reuse'].get(name)
    if not item or item['kind']!='verified_prefix':return None
    checked(FileRef(**item['manifest']))
    proof=read_json(item['manifest']['path'],'native-prefix-recovery-1')
    need(proof['source_model']==name and proof['source_configuration']==c['targets'][name] and proof['reader_split']==n['sources']['reader_split']
        and proof['prefix_layer']==source.prefix_index+1 and proof['dtype']=='float32','native prefix identity')
    for key in ('features','mask','original_failure','previous_configuration'):checked(FileRef(**proof[key]))
    need(proof['validated_rows']==proof['rows'] and proof['all_finite'] and proof['tokenizer_masks_exact'],'full recovered coverage')
    H=np.load(proof['features']['path'],mmap_mode='r');mask=np.load(proof['mask']['path'],mmap_mode='r')
    need(H.shape==(proof['rows'],source.max_tokens,source.hidden_dim) and mask.shape==H.shape[:2],'native prefix shape')
    need(not H.flags.writeable and not mask.flags.writeable,'historical cache read-only')
    publish(root,'prefix_reuse',{'proof.json':json_bytes(proof)},{'source_model':name,'historical_arrays_not_rewritten':True},
        sources=(FileRef(**item['manifest']),FileRef(**proof['features']),FileRef(**proof['mask'])))
    return H,mask


def validate_prepared_payload(old,n,report):
    from .audit import strict
    from .native import select
    records=strict((old/'completed/gallery_sources.json').read_text())
    raw=strict(Path(n['prototype']['annotation']).read_text())
    train=[(i,x) for i,x in enumerate(raw) if x['split']=='train']
    need(len(records)==len(train)==34054,'full official Train gallery')
    for rec,(i,x) in zip(records,train):
        need(rec['raw_index']==i and rec['file_path']==x['file_path'],'ordered gallery image identity')
        checked(FileRef(str(Path(n['prototype']['image_root'])/rec['file_path']),rec['sha256']))
    gallery=torch.load(old/'completed/gallery.pt',map_location='cpu',weights_only=True)
    clean=torch.load(old/'completed/clean.pt',map_location='cpu',weights_only=True)
    need(gallery.shape==(34054,256) and clean.shape==(2048,256) and torch.isfinite(gallery).all(),'native feature tensors')
    need(torch.allclose(gallery.norm(dim=1),torch.ones(34054),atol=1e-5,rtol=0),'native feature normalization')
    entries=read_json(Path(n['prototype']['history_root'])/'qualification/train.json')['entries']
    lookup={rec['file_path']:i for i,rec in enumerate(records)}
    need(torch.equal(clean,torch.stack([gallery[lookup[e['identity']['image_rel_path']]] for e in entries])),'paired clean anchors')
    plans=read_json(old/'completed/selection.json');candidates=read_json(old/'completed/candidates.json')
    need(list(plans)==list(candidates) and set(plans)==set(n['prototype']['train_ids']),'selection membership')
    for row,values in candidates.items():
        need(all(math.isfinite(x['S_raw']) and x['S_raw']>=0 for x in values),'finite native scores')
        need(select(values)==plans[row],'gamma3 selection exact')
    need(sum(len(v)+1 for v in plans.values())==report['expected_G_updates_per_round'],'native G budget')
    for b in read_json(old/'completed/bases.json').values():
        Q=torch.tensor(b['Q'],dtype=torch.float64)
        need(Q.shape==(768,b['rank']) and torch.isfinite(Q).all(),'native basis shape')
        need(torch.allclose(Q.T@Q,torch.eye(b['rank'],dtype=torch.float64),atol=1e-8,rtol=1e-8),'basis orthonormality')
