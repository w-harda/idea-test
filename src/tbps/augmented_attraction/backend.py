"""三视图G共享更新、完整当前G多视图库、H三头SCST共同引导。"""
from contextlib import contextmanager
from pathlib import Path
from dataclasses import asdict
import torch
from tbps.artifacts import ArtifactStore,FileRef,read_json,sha_file,json_bytes
from tbps.alternating.fresh_training import FreshBackend
from tbps.alternating.native_sources import native_session,source_key
from tbps.alternating.state import clone_tree,tree_sha
from tbps.alternating.contracts import BankIdentity
from tbps.alternating.formal_state import tensor_bytes
from tbps.alternating.native_backend import assert_pixels
from .objectives import Settings,ProfileKL,require
from .h_reward import train_step
from .views import ViewSettings,recipe,apply_views

class AttractionBackend(FreshBackend):
    def load_reward_inputs(self):
        super().load_reward_inputs()
        self.profile_cpu=ProfileKL(self.clean,Settings(**self.c['attraction']))
        self.profile_g=self.profile_cpu.to(next(self.g.parameters()).device)
        self.anchor_sha=tree_sha(self.clean);self.view_settings=ViewSettings(**self.c['augmentation'])
    def g_update(self,row,condition):
        from .multiview_training import train_single_sample
        self.grant.validate(self.c,'smoke');self.g.requires_grad_(True)
        index=self.p['train_ids'].index(row);image=self.image(row)
        plan=recipe(self.view_settings,row,self.round_index,*image.shape[-2:]);views=apply_views(image.detach().cpu(),plan).to(image.device)
        with torch.no_grad():clean=self.encoder(views).detach().clone()
        clean[0]=self.clean[index].to(clean)
        gradients=[];before=self.g_fingerprint();teacher=self.h_fingerprint()
        def observe(event):
            gradients.append({'record':event['record'],'gradient_sha256':tree_sha({n:p.grad for n,p in self.g.named_parameters() if p.grad is not None})})
        raw=train_single_sample(self.g,self.encoder,self.ga,views,condition.payload,observer=observe,profile=self.profile_g,clean_features=clean)
        result={k:v for k,v in raw.items() if k not in ('serial_adversarial','joint_adversarial')}
        result.update(G_before=before,G_after=self.g_fingerprint(),gradient_observations=gradients,
            G_input_view_samples=3*result['optimizer_steps'],encoder_adv_view_samples=3*result['optimizer_steps'],encoder_clean_view_samples=3)
        result['attraction']={'teacher_H_state':teacher,'condition_identity':condition.identity.key,
            'anchor_sha256':self.anchor_sha,'candidate_count':len(self.clean),'settings':dict(self.c['attraction']),
            'partner_text_feature':condition.payload['z_t'].detach().cpu().reshape(-1).tolist(),
            'view_recipe':plan,'clean_view_sha256':tree_sha(views.detach().cpu())}
        self.cursor['g_updates']+=result['optimizer_steps'];return result
    def h_update(self,row,bank):
        self.grant.validate(self.c,'smoke');require(bank==self.last_bank,'current G reward bank required')
        require(bank.g_state_sha256==self.g_fingerprint(),'stale G parameters invalidate H partner bank')
        self.cursor['h_visits']+=1;e=self.entries[row]
        if not e['eligible']:return {'row_id':row,'fixed_NOOP':True,'optimizer_steps_this_step':0,'IRRA_text_samples':0}
        self.h.requires_grad_(True);example=self.native.get('formal_training.data').example_from(e);query=self.catalog.get_train(row)
        index=self.p['train_ids'].index(row)
        def bind(k):return self.reward.bind(query,self.runtime,example,k)
        log,_=train_step(self.native.get('phase4_training'),self.h,self.ha,example,self.sampler,bind,
            self.cursor['h_visits'],row,self.profile_cpu,self.current_view_features[index])
        self.cursor['h_updates']+=1;log['bank_sha256']=bank.adversarial_sha256
        log['attraction'].update(teacher_G_state=bank.g_state_sha256,anchor_sha256=self.anchor_sha,
            settings=dict(self.c['attraction']),view_bank_sha256=self.view_bank_sha)
        return log
    def refresh_bank(self,ids,arm,index):
        require(list(ids)==self.p['train_ids'] and len(ids)==2048,'full original ordered Train identities')
        root=self.root/arm/f'bank_{index}';require(not root.exists(),'new immutable view bank only')
        store=ArtifactStore(root,new_base=self.c['output_base']);self.g.eval().requires_grad_(False);self.g.zero_grad(set_to_none=True)
        gsha=self.g_fingerprint();store.publish('G_state',{'state.pt':tensor_bytes(self.g.state_dict())},{'G':gsha,'arm':arm,'round':index})
        full=[];view_inputs=[]
        for position,row in enumerate(ids):
            store.publish(f'claim_{position:04d}',{'claim.json':json_bytes({'row':row,'index':position,'G':gsha})},{'source':source_key(self.c),'G':gsha})
            image=self.image(row);plan=recipe(self.view_settings,row,index,*image.shape[-2:]);views=apply_views(image.detach().cpu(),plan).to(image.device)
            with torch.no_grad():
                generated=[self.native.get('inference_once').generate_once(self.g,views[j:j+1]) for j in range(3)]
                adv=torch.cat([v['adversarial'] for v in generated]);delta=torch.cat([v['effective_delta'] for v in generated])
                for j in range(3):assert_pixels(views[j:j+1],adv[j:j+1],delta[j:j+1])
                feature=self.encoder(adv).detach().cpu().clone()
            payload={'original':views[:1].detach().cpu().clone(),'adversarial':adv[:1].detach().cpu().clone(),
                'effective_delta':delta[:1].detach().cpu().clone(),'feature':feature[:1].clone(),
                'augmented_original':views[1:].detach().cpu().clone(),'augmented_adversarial':adv[1:].detach().cpu().clone(),
                'augmented_effective_delta':delta[1:].detach().cpu().clone(),'view_features':feature}
            md={'row':row,'index':position,'G':gsha,'content_sha256':tree_sha(payload),'budget':8/255,'view_recipe':plan,
                'view_linf':[float(delta[j].detach().double().abs().max()) for j in range(3)]}
            view_inputs.append({'row':row,'input_sha256':tree_sha(views.detach().cpu()),'recipe_sha256':__import__('tbps.artifacts',fromlist=['digest']).digest(plan),'view_linf':md['view_linf'],'feature_sha256':tree_sha(feature)})
            store.publish(f'image_{position:04d}',{'data.pt':tensor_bytes(payload),'metadata.json':json_bytes(md)},md)
            restored=torch.load(root/f'image_{position:04d}/data.pt',map_location='cpu',weights_only=True)
            require(tree_sha(restored)==tree_sha(payload),'all view pixels atomic readback')
            full.append(feature);del image,views,adv,delta,feature,payload,restored,generated
            print('GH_AUG_BANK_PROGRESS',index,position+1,2048,'views',3,flush=True)
        tensor=torch.stack(full).clone();identity_features=tensor[:,0].clone()
        self.previous_features=self.current_features;self.current_features=identity_features;self.current_view_features=tensor;self.view_bank_sha=tree_sha(tensor)
        identity=BankIdentity(gsha,tuple(ids),self.p['clean_gallery_sha256'],tree_sha(identity_features),self.p['source_space_sha256'],index,arm)
        store.publish('completed',{'features.pt':tensor_bytes(identity_features),'view_features.pt':tensor_bytes(tensor),'view_inputs.json':json_bytes({'entries':view_inputs}),'identity.json':json_bytes(asdict(identity))},
            {'source':source_key(self.c),'bank':asdict(identity),'view_bank_sha256':tree_sha(tensor),'augmentation':self.c['augmentation']})
        require(gsha==self.g_fingerprint(),'frozen G during full view bank');self.last_bank=identity;return identity
    def restore_complete(self,value):
        super().restore_complete(value)
        if value['cursor']['round']:
            p=self.root/'B'/f"bank_{value['cursor']['round']}"/'completed/view_features.pt'
            self.current_view_features=torch.load(p,map_location='cpu',weights_only=True).clone()
            self.view_bank_sha=tree_sha(self.current_view_features)
            require(self.current_view_features.shape==(2048,3,512) and torch.equal(self.current_view_features[:,0],self.current_features),'resume view bank exact identity')

@contextmanager
def loaded(c,bundle,grant):
    from tbps.models.irra import IRRAAdapter
    from tbps.models.irra_source import IRRASource
    torch.cuda.manual_seed_all(42);cuda=clone_tree(torch.cuda.get_rng_state_all())
    for side in ('G','H'):bundle.state[side+'_RNG']['cuda']=clone_tree(cuda)
    source=IRRASource.from_dict(read_json(c['sources']['dataset_config']['path'])['irra'])
    adapter=IRRAAdapter.from_source(source,device='cuda:0',allow_model_load=True)
    with native_session(c,reuse=bundle.modules) as session:
        backend=AttractionBackend(c,bundle,grant,adapter);backend.native_session=session
        try:yield backend
        finally:backend.close()
