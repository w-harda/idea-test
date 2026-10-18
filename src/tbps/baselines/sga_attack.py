"""SGA image-only TBPS适配；增强直接调用SHA固定官方ImageAttacker。
官方MIT来源见docs/framework/SGA_LICENSE.txt；不含SGAttacker的文本扰动两阶段。
"""
from pathlib import Path
import importlib.util
import hashlib
import numpy as np
import torch
from tbps.artifacts import ArtifactError, checked, FileRef

OFFICIAL_COMMIT="f49e6926cd13148e49bdbc2c3e537660e096230c"
SCALES=(.5,.75,1.25,1.5)
EPSILON=8/255
STEP_SIZE=.5/255
STEPS=10

def need(ok,message):
    if not ok:raise ArtifactError(message)

def official_image_attacker(source):
    checked(FileRef(**source))
    spec=importlib.util.spec_from_file_location("_tbps_official_sga_"+source["sha256"][:12],source["path"])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.ImageAttacker(None,eps=EPSILON,steps=STEPS,step_size=STEP_SIZE)

def sample_seed(image_id,seed=42):
    return int.from_bytes(hashlib.sha256((str(seed)+":"+image_id).encode()).digest()[:4],"big")

def project(clean,value,epsilon=EPSILON):
    """只把FP32边界向原图内侧舍入，严格满足实数8/255，无STE。"""
    low=(clean.double()-epsilon).clamp(0,1).to(clean.dtype)
    high=(clean.double()+epsilon).clamp(0,1).to(clean.dtype)
    low=torch.where(clean.double()-low.double()>epsilon,torch.nextafter(low,clean),low)
    high=torch.where(high.double()-clean.double()>epsilon,torch.nextafter(high,clean),high)
    return torch.maximum(torch.minimum(value,high),low)

def caption_groups(manifest):
    """仅原始paired_image_id；不使用同PID其他图的Caption，不按攻击效果选词。"""
    groups={g.image_id:[] for g in manifest.gallery}
    for q in manifest.queries:
        if q.paired_image_id in groups:groups[q.paired_image_id].append(q)
    need(all(groups.values()),"every attacked Gallery needs its original Caption set")
    return {k:tuple(v) for k,v in groups.items()}

def attack_image(clean,objective,official,*,seed,steps=STEPS,scales=SCALES,epsilon=EPSILON,step_size=STEP_SIZE):
    need(clean.dtype==torch.float32 and clean.ndim==4 and clean.shape[:2]==(1,3)
         and not clean.requires_grad and torch.isfinite(clean).all() and clean.min()>=0 and clean.max()<=1,
         "one original FP32 RGB image required")
    need(callable(objective) and steps>0 and epsilon>0 and step_size>0,"explicit SGA objective/settings")
    # 原官方使用NumPy uniform/normal；每图独立锁定seed，退出恢复外部RNG。
    saved=np.random.get_state();np.random.seed(seed);records=[]
    try:
        adv=project(clean,clean+torch.from_numpy(np.random.uniform(-epsilon,epsilon,clean.shape)).float().to(clean.device))
        for iteration in range(steps):
            adv=adv.detach().requires_grad_(True)
            augmented=official.get_scaled_imgs(adv,list(scales),str(clean.device))
            need(len(augmented)==len(scales)+1 and torch.isfinite(augmented).all(),"official scale set")
            # 顺序microbatch降低峰值：先逐尺度求dL/daug，再一次增强VJP。
            # 不保留历史模型图，不改变集合sum目标或增强随机数。
            derivatives=[];losses=[]
            for scale in augmented.split(1):
                leaf=scale.detach().requires_grad_(True)
                loss=objective(leaf)
                need(loss.ndim==0 and torch.isfinite(loss),"finite scalar SGA set objective")
                grad,=torch.autograd.grad(loss,leaf)
                need(torch.isfinite(grad).all(),"finite scale input gradient")
                derivatives.append(grad.detach());losses.append(float(loss.detach()))
                del loss,leaf,grad
            grad,=torch.autograd.grad(augmented,adv,grad_outputs=torch.cat(derivatives))
            magnitude=grad.abs().mean()
            need(torch.isfinite(grad).all() and magnitude>0,"nonzero finite SGA input gradient; no silent zero fallback")
            norm=float(grad.norm().detach())
            adv=project(clean,adv.detach()+step_size*(grad/magnitude).sign()).detach()
            linf=float((adv.double()-clean.double()).abs().max())
            need(linf<=epsilon and torch.isfinite(adv).all(),"strict global RGB attack budget")
            records.append({"iteration":iteration+1,"objective":sum(losses),"scale_objectives":losses,
                "gradient_l2":norm,"linf":linf})
            del augmented,derivatives,grad
        return adv,records
    finally:np.random.set_state(saved)
