"""批量候选奖励及等值历史复用；GPU仅在显式调用时使用。"""
import json, weakref
import torch
from torch.nn import functional as F
from tbps.attribute_response.settings import require
from tbps.attribute_response.memory import key_bytes, STATE_DIM, ACTION_DIM
_ANCHORS = weakref.WeakKeyDictionary()

def candidate_rewards(profile, features, clean, partners, *, device="cuda:0", chunk=128):
    require(features.ndim == 2 and clean.shape == (1,features.shape[1])
            and partners.shape == (3,features.shape[1]), "full candidate/clean/three-view shapes")
    require(type(chunk) is int and chunk > 0, "positive chunk")
    for x in (features,clean,partners):
        require(x.dtype==profile.anchors.dtype and x.device==profile.anchors.device,"original candidate space/dtype/device")
        require(torch.isfinite(x).all() and (x.norm(dim=1)>0).all(), "finite nonzero candidate features")
    device = torch.device(device)
    key=(str(device),id(profile.anchors),profile.anchors._version)
    cached=_ANCHORS.get(profile)
    if cached is None or cached[0]!=key:
        cached=(key,profile.anchors.detach().to(device));_ANCHORS[profile]=cached
    anchors=cached[1];tau=profile.settings.temperature
    out=[]
    with torch.no_grad():
        lc=F.log_softmax(F.normalize(clean.to(device),dim=1)@anchors.T/tau,dim=1)
        lp=F.log_softmax(F.normalize(partners.to(device),dim=1)@anchors.T/tau,dim=1)
        baseline=F.kl_div(lc.expand_as(lp),lp,log_target=True,reduction="none").sum(1)
        for start in range(0,len(features),chunk):
            x=features[start:start+chunk].to(device)
            lq=F.log_softmax(F.normalize(x,dim=1)@anchors.T/tau,dim=1)
            n=len(lq)
            pull=F.kl_div(lq[:,None,:].expand(n,3,-1),
                lp[None,:,:].expand(n,3,-1),log_target=True,reduction="none").sum(2)
            push=F.kl_div(lq,lc.expand_as(lq),log_target=True,reduction="none").sum(1)
            value=(profile.settings.pull_weight*(baseline[None,:]-pull)
                   +profile.settings.push_weight*push[:,None]).mean(1)
            require(torch.isfinite(value).all(),"finite all-candidate rewards")
            out.extend(value.cpu().tolist())
    require(len(out)==len(features),"every original candidate evaluated")
    return out

def fast_predict(self,batch):
    if not batch["keys"]:return torch.empty((0,2))
    model=self.model
    with torch.no_grad():
        keys=batch["keys"];states=torch.tensor(batch["states"]);actions=torch.tensor(batch["actions"])
        require(model.semantic.weight.device.type=="cpu" and states.device.type=="cpu",
                "training-only CPU response teacher")
        require(states.shape==(len(keys),STATE_DIM) and actions.shape==(len(keys),ACTION_DIM),
                "teacher feature shapes")
        semantic=model.semantic(torch.tensor([key_bytes(k) for k in keys])).mean(1)
        cache={};rows=[]
        for events,version in zip(batch["histories"],batch["versions"]):
            identity=json.dumps([events,version],sort_keys=True,ensure_ascii=False,allow_nan=False)
            if identity not in cache:
                hidden=states.new_zeros((1,model.semantic.embedding_dim))
                for event in events:
                    age=float(max(0,version-event["version"]))/(1+max(0,version-event["version"]))
                    value=states.new_tensor(event["state"]+event["response"]+[age])[None]
                    hidden=model.history(value,hidden)
                cache[identity]=hidden
            rows.append(cache[identity])
        history=torch.cat(rows) if rows else states.new_empty((0,model.semantic.embedding_dim))
        hidden=model.trunk(torch.cat((semantic,history,states,actions),1))
        out=torch.cat((model.g_to_h(hidden),model.h_to_g(hidden)),1)
    require(torch.isfinite(out).all(),"finite teacher response")
    return out.detach().clone()
