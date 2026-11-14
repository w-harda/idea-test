"""原生256维全局特征与768维BERT中间层；不复用IRRA特征/JVP。"""
from contextlib import contextmanager
from dataclasses import replace
import math,re
import torch
from torch import nn
from torch.nn import functional as F
from .common import *
class Source(nn.Module):
    def __init__(self,adapter,name):
        super().__init__();self.model=adapter.backend;self.adapter=adapter;self.name=name
        self.bert=self.model.text_encoder.bert
        self.text_end=int(self.bert.config.fusion_layer)
        self.prefix_index=math.floor(self.text_end*2/3)-1
        self.block=self.bert.encoder.layer[self.prefix_index]
        self.max_tokens=50 if name=='rasa' else 56
        self.hidden_dim=int(self.bert.config.hidden_size)
        self.feature_dim=self.model.text_proj.out_features
        from transformers import BertTokenizerFast
        self.fast=BertTokenizerFast.from_pretrained(adapter.tokenizer.name_or_path,local_files_only=True)
        self.model.eval().requires_grad_(False)
        self.frozen=tree_sha(self.model.state_dict())
        self.versions=[(id(x),x._version) for x in (*self.model.parameters(),*self.model.buffers())]
    def forward(self,pixels):
        x=self.adapter.prepare_pixels(pixels)
        if self.name=='rasa':h=self.model.visual_encoder(x)
        else:h,_=self.model.get_vision_embeds(x)
        return F.normalize(self.model.vision_proj(h[:,0]),dim=1)
    def text(self,texts):
        batch=self.adapter.tokenize(texts).to(next(self.model.parameters()).device)
        if self.name=='rasa':
            h=self.bert(batch.input_ids,attention_mask=batch.attention_mask,mode='text').last_hidden_state
        else:h=self.model.get_text_embeds(batch.input_ids,batch.attention_mask)
        return F.normalize(self.model.text_proj(h[:,0]),dim=1)
    def prefix(self,texts):
        captured=[]
        class Stop(Exception):pass
        def hook(m,a,out):
            captured.append(out[0].detach().clone());raise Stop()
        h=self.block.register_forward_hook(hook)
        try:
            with torch.no_grad():self.text(texts)
        except Stop:pass
        finally:h.remove()
        need(len(captured)==1,'one native prefix capture')
        batch=self.adapter.tokenize(texts)
        mask=batch.attention_mask.bool();mask[:,0]=False
        for j,row in enumerate(batch.input_ids):mask[j,row==self.adapter.tokenizer.sep_token_id]=False
        need(mask.any(1).all(),'nonempty native content')
        return captured[0],mask
    def transport(self,caption,eta,direction):
        def hook(m,a,out):return (out[0].detach()+eta*direction,*out[1:])
        handle=self.block.register_forward_hook(hook)
        try:return self.text([caption])
        finally:handle.remove()
    def linearized(self,caption):
        H,_=self.prefix([caption]);H=H.requires_grad_(True)
        def hook(m,a,out):return (H,*out[1:])
        h=self.block.register_forward_hook(hook)
        try:z=self.text([caption])
        finally:h.remove()
        return H,z
    def check_frozen(self,final=False):
        need(self.versions==[(id(x),x._version) for x in (*self.model.parameters(),*self.model.buffers())],'native target mutated')
        need(all(not x.requires_grad and x.grad is None for x in self.model.parameters()),'native target gradients')
        need(not self.model.training,'native target training mode')
        if final:need(tree_sha(self.model.state_dict())==self.frozen,'native target SHA changed')
    def normalized(self,caption):
        if self.name=='rasa':return self.adapter.pre_caption(caption,50)
        return self.adapter.pre_caption(caption,56,icfg_rstp=False,is_eda=False)
    def aligned(self,caption):
        """跟随官方预处理逐字符传递来源，Fast仅提供偏移；ID必须等于官方Tokenizer。"""
        if self.name=='rasa':
            seq=[]
            for i,ch in enumerate(caption):
                for v in ch.lower():
                    if v in ",.'!?\"()*#:;~":continue
                    seq.append((' ' if v in '-/' else v,i))
            # 原生<person>替换仅删除尖括号，保留person的真实字符来源。
            txt=''.join(v for v,_ in seq)
            while '<person>' in txt:
                a=txt.index('<person>');seq=seq[:a]+seq[a+1:a+7]+seq[a+8:];txt=''.join(v for v,_ in seq)
            compressed=[];i=0
            while i<len(seq):
                j=i+1
                if seq[i][0].isspace():
                    while j<len(seq) and seq[j][0].isspace():j+=1
                    compressed.extend([(' ',seq[i][1])] if j-i>=2 else [seq[i]])
                else:compressed.append(seq[i])
                i=j
            while compressed and compressed[-1][0]=='\n':compressed.pop()
            while compressed and compressed[0][0]==' ':compressed.pop(0)
            while compressed and compressed[-1][0]==' ':compressed.pop()
            seq=compressed
            words=''.join(v for v,_ in seq).split(' ')
            if len(words)>50:seq=seq[:len(' '.join(words[:50]))]
        else:
            seq=[(ch,i) for i,ch in enumerate(caption)]
            words=list(re.finditer(r'\S+',caption))
            if len(words)>56:
                seq=[]
                for j,m in enumerate(words[:56]):
                    if j:seq.append((' ',m.start()-1))
                    seq.extend((caption[k],k) for k in range(m.start(),m.end()))
        clean=''.join(v for v,_ in seq);need(clean==self.normalized(caption),'native preprocessing character map')
        # 两个官方Tokenizer单句都只加CLS，不加SEP；Fast仅分内容，不采用其默认模板。
        fast=self.fast(clean,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
        ids=[self.adapter.tokenizer.cls_token_id]+fast['input_ids']
        offsets=[(0,0)]+fast['offset_mapping']
        ids=ids[:self.max_tokens];offsets=offsets[:self.max_tokens]
        ids += [self.adapter.tokenizer.pad_token_id]*(self.max_tokens-len(ids))
        offsets += [(0,0)]*(self.max_tokens-len(offsets))
        official=self.adapter.tokenize([caption]);need(torch.equal(torch.tensor([ids]),official.input_ids),'official/Fast WordPiece mismatch')
        support=[set(seq[k][1] for k in range(a,b)) if b>a else set() for a,b in offsets]
        return support
    def support(self,caption,span):
        a,b=span['start'],span['end'];tokens=self.aligned(caption)
        return [i for i,chars in enumerate(tokens) if any(a<=v<b for v in chars)]
@contextmanager
def loaded(c,name,smoke=False):
    from importlib import import_module
    mod=import_module('tbps.models.'+name)
    source=(mod.RasaSource if name=='rasa' else mod.APTMSource)(c['targets'][name]['path']);source.verify()
    loader=mod.load_rasa if name=='rasa' else mod.load_aptm
    with loader(source,smoke=smoke) as adapter:
        wrapper=Source(adapter,name);wrapper.eval().requires_grad_(False)
        yield wrapper
def instances(frontend,caption,row='caption-only'):
    result=frontend.extract(caption)
    aa=frontend.adapter.convert_record({'row_id':row,'caption':caption,**result})
    return frontend.adapter.text_candidates(aa,caption,require_representation=False)
def pool_slots(a,slots):
    direct={'age','gender','hair_length','hat','glasses','backpack','bag'}
    if a.semantic_axis in direct:return [slots.index(a.semantic_axis)]
    if a.region in ('upper','lower'):return [slots.index(a.region+'_clothing_'+x) for x in ('color','type','length')]
    return None
def localization(source,reader,split,bases,caption,attrs):
    H,mask=source.prefix([caption]);H=H.detach()
    with torch.no_grad():alpha,z,_=reader(H,mask.to(H.device))
    out={}
    for a in attrs:
        if a.unresolved or a.semantic_axis not in bases:continue
        basis=bases[a.semantic_axis]
        if not basis['rank'] or a.canonical_value not in basis['values']:continue
        if not source.support(caption,a.source_span):continue
        if a.linked_object and not source.support(caption,a.linked_object):continue
        indices=pool_slots(a,split['slots'])
        if indices is None:
            span=a.source_span if a.semantic_axis=='garment_type' else a.linked_object
            if not span:continue
            ids=source.support(caption,span)
            if not ids:continue
            w=torch.zeros(source.max_tokens,device=H.device);w[ids]=1/len(ids);pooled=(H[0]*w[:,None]).sum(0)
        else:w=alpha[0,indices].mean(0);pooled=z[0,indices].mean(0)
        w=(w/w.norm()).detach()
        Q=torch.tensor(basis['Q'],dtype=torch.float64,device=H.device)
        center=torch.tensor(basis['center'],dtype=torch.float64,device=H.device)
        projected=Q@(Q.T@(pooled.double()-center))
        need(projected.norm()>0,'zero native semantic direction')
        d=(projected/projected.norm()).float()
        B=(w[:,None]*d[None])[None].detach()
        need(abs(float(B.double().norm())-1)<3e-6,'native unit hidden direction')
        out[a.instance_id]={'attribute':a,'w':w,'Q':Q,'B':B}
    return out
def jvp(source,caption,item):
    t=torch.zeros((),device=next(source.parameters()).device,requires_grad=True);B=item['B']
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        z,v=torch.autograd.functional.jvp(lambda eta:source.transport(caption,eta,B),t,torch.ones_like(t),strict=True)
        with torch.no_grad():
            clean=source.text([caption]);zero=source.transport(caption,torch.zeros_like(t),B)
            fd=(source.transport(caption,torch.full_like(t,.01),B)-source.transport(caption,torch.full_like(t,-.01),B))/.02
    need(torch.equal(z,clean) and torch.equal(zero,clean),'native JVP zero exact')
    need(torch.isfinite(v).all() and v.norm()>0,'finite nonzero native JVP')
    error=float((fd-v).double().norm()/v.double().norm())
    need(error<=.02,'native JVP finite difference exceeds original 2% threshold')
    return v[0].detach().cpu(),{'zero_exact':True,'FD_relative_error':error,'unit_direction_norm':float(B.double().norm())}
def select(scores):
    ordered=sorted((v for v in scores if v['S_raw']>0),key=lambda v:-v['S_raw'])
    if not ordered:return []
    peak=math.log(ordered[0]['S_raw']);u=[math.exp(3*(math.log(x['S_raw'])-peak)) for x in ordered];total=math.fsum(u)
    p=[v/total for v in u];n=math.exp(-math.fsum(x*math.log(x) for x in p if x>0))
    return ordered[:min(len(ordered),math.ceil(n))]
