"""同一Canonical标签及PID划分，拟合各代理原生reader/basis与干净Train图库。"""
import numpy as np
import torch
from .common import *
from .native import pool_slots,instances,localization,select
def pixels(bundle,path):
    from PIL import Image
    with Image.open(path) as im:return bundle.bindings.get('irra_image_adapter').pixels_from_pil(im).unsqueeze(0)
def prepare(c,n,bundle,source,name,job):
    root=OUTPUT/name/'preparation';need(not root.exists(),'preparation Attempt consumed')
    root.mkdir(parents=True)
    split=read_json(n['sources']['reader_split']['path']);records=split['records']
    raw=json.loads(Path(n['prototype']['annotation']).read_text())
    for r in records:
        item=raw[int(r['row_id'].split(':')[1])]
        need(item['split']=='train' and item['id']==r['pid'] and item['captions'][int(r['row_id'].split(':')[2])]==r['caption'],'reader source official Train only')
    need(not set(split['train_pids'])&set(split['validation_pids']),'reader PID split')
    N=len(records);L=source.max_tokens;D=source.hidden_dim
    cache=root/'native_prefix.fp32.npy'
    H=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float32,shape=(N,L,D))
    mask=np.lib.format.open_memmap(root/'content_mask.npy',mode='w+',dtype=np.bool_,shape=(N,L))
    for start in range(0,N,16):
        texts=[r['caption'] for r in records[start:start+16]]
        h,m=source.prefix(texts);H[start:start+len(texts)]=h.cpu().numpy();mask[start:start+len(texts)]=m.numpy()
        state(job,'PREPARE_NATIVE_READER_FEATURES',done=start+len(texts),total=N)
    H.flush();mask.flush()
    torch.manual_seed(42)
    reader=bundle.bindings.get('learned_reader').SlotReader([len(v) for v in split['values']],dim=D).to('cuda:0')
    optimizer=torch.optim.Adam(reader.parameters(),lr=.001)
    train=np.array(split['train_indices']);valid=np.array(split['validation_indices']);rng=np.random.default_rng(42)
    labels=torch.tensor([r['labels'] for r in records],dtype=torch.long)
    reader_history=[]
    def batch(ix):
        return torch.from_numpy(np.array(H[ix],copy=True)).cuda(),torch.from_numpy(np.array(mask[ix],copy=True)).cuda(),labels[ix].cuda()
    for epoch in range(1,21):
        reader.train();order=rng.permutation(train);loss_sum=0
        for start in range(0,len(order),512):
            ix=order[start:start+512];h,m,y=batch(ix);optimizer.zero_grad(set_to_none=True)
            _,_,logits=reader(h,m);loss=bundle.bindings.get('learned_reader').ce_sum(logits,y)
            need(torch.isfinite(loss),'native reader finite loss');loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(ix)
        reader.eval();val=0
        with torch.no_grad():
            for start in range(0,len(valid),512):
                ix=valid[start:start+512];h,m,y=batch(ix)
                val+=float(bundle.bindings.get('learned_reader').ce_sum(reader(h,m)[2],y))*len(ix)
        reader_history.append({'epoch':epoch,'train_CE':loss_sum/len(train),'validation_CE':val/len(valid)})
        state(job,'PREPARE_NATIVE_READER_TRAIN',total_epochs=20,**reader_history[-1])
    reader.eval().requires_grad_(False);reader.zero_grad(set_to_none=True)
    sums={};counts={}
    for start in range(0,len(train),128):
        ix=train[start:start+128];h,m,_=batch(ix)
        with torch.no_grad():alpha,z,_=reader(h,m)
        for j,k in enumerate(ix):
            rec=records[int(k)]
            for a in instances(bundle.frontend,rec['caption'],rec['row_id']):
                slots=pool_slots(a,split['slots'])
                if slots is None:
                    span=a.source_span if a.semantic_axis=='garment_type' else a.linked_object
                    if not span:continue
                    ids=source.support(rec['caption'],span)
                    if not ids:continue
                    zz=h[j,ids].double().mean(0).cpu()
                else:zz=z[j,slots].double().mean(0).cpu()
                key=(a.semantic_axis,a.canonical_value)
                sums[key]=sums.get(key,torch.zeros(D,dtype=torch.float64))+zz;counts[key]=counts.get(key,0)+1
        state(job,'PREPARE_NATIVE_BASES',done=min(start+128,len(train)),total=len(train))
    bases={}
    for axis in bundle.frontend.adapter.AXES:
        values=sorted(v for a,v in sums if a==axis)
        if not values:continue
        centroids=torch.stack([sums[(axis,v)]/counts[(axis,v)] for v in values])
        center=centroids.mean(0);matrix=(centroids-center).T
        u,s,_=torch.linalg.svd(matrix,full_matrices=False)
        rank=int((s>max(float(s[0])*1e-8,1e-12)).sum()) if len(s) else 0
        Q=u[:,:rank]
        bases[axis]={'values':values,'center':center.tolist(),'Q':Q.tolist(),'rank':rank,
                     'centroids':centroids.tolist(),'counts':[counts[(axis,v)] for v in values]}
    # 原生全官方Train图库；不是Test，也不是仅2048奖励Bank。
    gallery=[];gallery_index={};gallery_sources=[]
    for raw_index,item in enumerate(raw):
        if item['split']!='train':continue
        path=Path(n['prototype']['image_root'])/item['file_path']
        image=pixels(bundle,path)
        with torch.no_grad():feature=source(image.cuda()).cpu().clone()
        gallery_index[item['file_path']]=len(gallery);gallery.append(feature)
        gallery_sources.append({'raw_index':raw_index,'file_path':item['file_path'],'sha256':sha_file(path)})
        if len(gallery)%64==0:state(job,'PREPARE_NATIVE_TRAIN_GALLERY',done=len(gallery),total=34054)
    need(len(gallery)==34054,'full official Train Gallery cardinality')
    gallery=torch.cat(gallery);need(gallery.shape==(34054,source.feature_dim),'native Gallery feature dimensions')
    q=read_json(Path(n['prototype']['history_root'])/'qualification/train.json')['entries']
    clean=torch.cat([gallery[gallery_index[e['identity']['image_rel_path']]:gallery_index[e['identity']['image_rel_path']]+1] for e in q]).clone()
    plans={};candidates_by_row={}
    # 每条原文一次原检索目标梯度，所有属性共享；Native-K gamma3/beta1，无cap4。
    for position,e in enumerate(q):
        caption=e['caption'];row=e['identity']['row_id']
        aa=instances(bundle.frontend,caption,row);located=localization(source,reader,split,bases,caption,aa)
        h,z=source.linearized(caption);logits=(z@gallery.cuda().T)/.07
        target=gallery_index[e['identity']['image_rel_path']]
        objective=torch.logsumexp(logits,dim=1)[0]-logits[0,target]
        grad,=torch.autograd.grad(objective,h)
        scores=[]
        for aid,item in located.items():
            g=(grad[0]*item['w'][:,None]).sum(0);s=float((item['Q'].T@g.double()).norm())
            scores.append({'instance_id':aid,'S_raw':s,'attribute':item['attribute'].to_dict()})
        chosen=select(scores)
        plans[row]=chosen;candidates_by_row[row]=scores
        state(job,'PREPARE_NATIVE_ATTRIBUTE_SELECTION',done=position+1,total=len(q),K=len(chosen))
        del h,z,grad,logits
    source.check_frozen(final=True)
    report={'source_model':name,'source_space':asdict(source.adapter.space),'hidden_dim':D,'feature_dim':source.feature_dim,
        'prefix_layer':source.prefix_index+1,'native_text_layers':source.text_end,
        'reader_epochs':20,'reader_selection':'fixed final20','reader_train':len(train),'reader_validation':len(valid),
        'reader_PID_split_ref':asdict(ref(n['sources']['reader_split']['path'])),'reader_history':reader_history,
        'gallery_count':34054,'train_count':2048,'Test_feedback':False,'selection_gamma':3,'selection_beta':1,
        'K_distribution':{str(k):sum(len(p)==k for p in plans.values()) for k in sorted(set(map(len,plans.values())))},
        'expected_G_updates_per_round':sum(len(p)+1 for p in plans.values()),'source_frozen':True}
    publish(root,'completed',{'reader.pt':tensor_bytes(reader.state_dict()),'bases.json':json_bytes(bases),'selection.json':json_bytes(plans),
        'candidates.json':json_bytes(candidates_by_row),'clean.pt':tensor_bytes(clean),'gallery.pt':tensor_bytes(gallery),
        'gallery_sources.json':json_bytes(gallery_sources),'report.json':json_bytes(report)},report,
        sources=(ref(c['_config_path']),ref(n['sources']['reader_split']['path']),source.adapter.space and ref(c['targets'][name]['path'])))
    return reader,split,bases,plans,clean,q
