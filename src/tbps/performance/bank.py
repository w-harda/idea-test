"""原Bank原子发布和审计保持；仅加速输入准备。"""
from pathlib import Path
from dataclasses import asdict
import torch
from tbps.artifacts import ArtifactStore,FileRef,read_json,sha_file,json_bytes
from tbps.alternating.native_sources import source_key
from tbps.alternating.state import tree_sha
from tbps.alternating.contracts import BankIdentity
from tbps.alternating.formal_state import tensor_bytes
from tbps.alternating.native_backend import assert_pixels
from tbps.augmented_attraction.objectives import require
from tbps.augmented_attraction.views import recipe
def refresh_bank(self,ids,arm,index):
    require(list(ids)==self.p['train_ids'] and len(ids)==2048,'full original ordered Train identities')
    root=self.root/arm/f'bank_{index}';require(not root.exists(),'new immutable view bank only')
    store=ArtifactStore(root,new_base=self.c['output_base']);self.g.eval().requires_grad_(False);self.g.zero_grad(set_to_none=True)
    gsha=self.g_fingerprint();store.publish('G_state',{'state.pt':tensor_bytes(self.g.state_dict())},{'G':gsha,'arm':arm,'round':index})
    full=[];view_inputs=[]
    for position,row in enumerate(ids):
        store.publish(f'claim_{position:04d}',{'claim.json':json_bytes({'row':row,'index':position,'G':gsha})},{'source':source_key(self.c),'G':gsha})
        image=self._performance_images.cpu(row);plan=recipe(self.view_settings,row,index,*image.shape[-2:]);views=self._performance_images.views(row,index).to(next(self.g.parameters()).device,non_blocking=True)
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
