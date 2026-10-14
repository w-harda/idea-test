"""AP-style轻量条件ResNet；不导入IRRA、scorer、JVP、TTA。"""
import math
import torch
from torch import nn
from torch.nn import functional as F
EPSILON=8/255
class ResidualBlock(nn.Module):
 def __init__(self,dim):
  super().__init__()
  self.layers=nn.Sequential(nn.ReflectionPad2d(1),nn.Conv2d(dim,dim,3),nn.InstanceNorm2d(dim),
   nn.ReLU(),nn.ReflectionPad2d(1),nn.Conv2d(dim,dim,3),nn.InstanceNorm2d(dim))
 def forward(self,x):return x+self.layers(x)
class ConditionalGenerator(nn.Module):
 def __init__(self,width=32,blocks=6,condition_hidden=256,epsilon=EPSILON):
  super().__init__()
  if width<1 or blocks<1 or not math.isfinite(epsilon) or not 0<epsilon<=EPSILON:raise ValueError("invalid architecture/budget")
  self.epsilon=float(epsilon);self.architecture={"width":width,"blocks":blocks,"condition_hidden":condition_hidden,"epsilon":float(epsilon)}
  self.enc0=nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(3,width,7),nn.InstanceNorm2d(width),nn.ReLU())
  self.enc1=nn.Sequential(nn.Conv2d(width,width*2,3,2,1),nn.InstanceNorm2d(width*2),nn.ReLU())
  self.enc2=nn.Sequential(nn.Conv2d(width*2,width*4,3,2,1),nn.InstanceNorm2d(width*4),nn.ReLU())
  self.z_norm=nn.LayerNorm(512);self.attr_norm=nn.LayerNorm(512)
  self.condition=nn.Sequential(nn.Linear(1024,condition_hidden),nn.ReLU(),nn.Linear(condition_hidden,width*8))
  self.residual=nn.Sequential(*(ResidualBlock(width*4) for _ in range(blocks)))
  self.dec1=nn.Sequential(nn.Conv2d(width*4,width*2,3,1,1),nn.InstanceNorm2d(width*2),nn.ReLU())
  self.dec0=nn.Sequential(nn.Conv2d(width*2,width,3,1,1),nn.InstanceNorm2d(width),nn.ReLU())
  self.output=nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(width,3,7))
  nn.init.normal_(self.output[1].weight,std=.001);nn.init.zeros_(self.output[1].bias)
 def forward(self,image,z_t,c_attr):
  if image.ndim!=4 or image.shape[1]!=3 or min(image.shape[-2:])<16:raise ValueError("image [B,3,H,W>=16]")
  if z_t.shape!=(len(image),512) or c_attr.shape!=z_t.shape:raise ValueError("condition [B,512]")
  if z_t.requires_grad or c_attr.requires_grad:raise ValueError("condition frontend must be frozen/detached")
  if not all(torch.isfinite(t).all() for t in (image,z_t,c_attr)) or image.min()<0 or image.max()>1:raise ValueError("invalid pixels/condition")
  a=self.enc0(image*2-1);b=self.enc1(a);h=self.enc2(b)
  gamma,beta=self.condition(torch.cat((self.z_norm(z_t),self.attr_norm(c_attr)),dim=1)).chunk(2,dim=1)
  h=h*(1+.1*gamma.tanh()[:,:,None,None])+beta[:,:,None,None]
  h=self.residual(h)
  h=self.dec1(F.interpolate(h,size=b.shape[-2:],mode="bilinear",align_corners=False))
  h=self.dec0(F.interpolate(h,size=a.shape[-2:],mode="bilinear",align_corners=False))
  return self.epsilon*self.output(h).tanh()
def compose(image,delta):
 if delta.shape!=image.shape:raise ValueError("delta shape")
 return (image+delta).clamp(0,1)
@torch.inference_mode()
def generate_once(generator,image,z_t,c_attr):
 generator.eval()
 delta=generator(image,z_t,c_attr)
 adv=compose(image,delta)
 return {"delta":delta,"adversarial":adv,"effective_delta":adv-image}
def total_variation(x):
 return (x[:,:,1:,:]-x[:,:,:-1,:]).abs().mean()+(x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean()
def training_loss(feature,z_t,v_local,has_local,effective_delta,*,epsilon=EPSILON,global_weight=1.,local_weight=1.,perturbation_weight=.001,tv_weight=.001):
 if feature.shape!=z_t.shape or feature.shape!=v_local.shape or has_local.shape!=(len(feature),):raise ValueError("loss shapes")
 if z_t.requires_grad or v_local.requires_grad:raise ValueError("loss targets frozen")
 # Adam最小化正dot，与攻击损害目标一致。JVP不是属性存在概率。
 global_loss=(feature*z_t).sum(1).mean()
 per_local=(feature*v_local).sum(1)
 local_loss=(per_local*has_local.to(feature.dtype)).mean()
 normalized=effective_delta/epsilon
 perturbation=normalized.square().mean();tv=total_variation(normalized)
 total=global_weight*global_loss+local_weight*local_loss+perturbation_weight*perturbation+tv_weight*tv
 if not torch.isfinite(total):raise ValueError("nonfinite loss")
 return total,{"global":global_loss,"local":local_loss,"perturbation":perturbation,"TV":tv}
