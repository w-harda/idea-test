"""AoA局部/全局思想的TBPS适配；确定性recipe，不访问PID或Test反馈。"""
from dataclasses import dataclass,asdict
import hashlib,random,math
import torch
from torch.nn import functional as F
from tbps.artifacts import digest
from .objectives import require

@dataclass(frozen=True)
class ViewSettings:
    seed:int=42
    views:int=3
    local_rows:int=4
    local_cols:int=2
    global_chain:int=3
    rotation_degrees:float=5.
    translation_fraction:float=.04
    def __post_init__(self):
        require(type(self.seed) is int and self.views==3 and self.local_rows==4 and self.local_cols==2 and self.global_chain==3,'fixed original/local/global view contract')
        require(math.isfinite(self.rotation_degrees) and 0<self.rotation_degrees<=5 and math.isfinite(self.translation_fraction) and 0<self.translation_fraction<=.05,'bounded TBPS geometry')

OPS=('blur','hflip','resize_pad','translate','rotate')
def recipe(settings,row,round_index,height,width):
    require(isinstance(settings,ViewSettings) and isinstance(row,str) and row and type(round_index) is int and round_index>0,'explicit Train row/round recipe')
    require(height%settings.local_rows==0 and width%settings.local_cols==0,'equal fixed body strips')
    seed=int.from_bytes(hashlib.sha256(f'gh-aug-v1|{settings.seed}|{round_index}|{row}'.encode()).digest()[:8],'big')
    rng=random.Random(seed)
    def operation(h,w):
        kind=rng.choice(OPS);op={'kind':kind}
        if kind=='blur':op['sigma']=rng.uniform(.5,1.)
        if kind=='rotate':op['degrees']=rng.uniform(-settings.rotation_degrees,settings.rotation_degrees)
        if kind=='translate':op.update(dx=rng.uniform(-settings.translation_fraction,settings.translation_fraction),dy=rng.uniform(-settings.translation_fraction,settings.translation_fraction))
        if kind=='resize_pad':
            # GLEAM全局resize-pad的长宽比保持版本；也保留SGA先缩放再回原尺寸的思路。
            factor=rng.uniform(.85,1.15);canvas_h,canvas_w=math.ceil(h*1.2),math.ceil(w*1.2)
            ih,iw=max(1,round(h*factor)),max(1,round(w*factor));top=rng.randrange(canvas_h-ih+1);left=rng.randrange(canvas_w-iw+1)
            op.update(size=[ih,iw],padding=[left,canvas_w-iw-left,top,canvas_h-ih-top])
        return op
    local=[]
    for i in range(settings.local_rows):
        perm=list(range(settings.local_cols));rng.shuffle(perm)
        local.append({'row':i,'operations':[operation(height//settings.local_rows,width//settings.local_cols) for _ in perm],'permutation':perm})
    return {'schema':'gh-pushpull-view-recipe-1','row_id':row,'round':round_index,'shape':[height,width],
        'settings':asdict(settings),'local':local,'global':[operation(height,width) for _ in range(settings.global_chain)],
        'view_names':['original','local','global'],'upper_lower_strip_preserved':True}

def transform(x,op):
    require(x.ndim==4 and len(x)==1 and x.shape[1]==3,'single RGB view')
    kind=op['kind'];require(kind in OPS,'known augmentation operation')
    if kind=='hflip':return x.flip(-1)
    if kind=='blur':
        q=torch.arange(-2,3,device=x.device,dtype=x.dtype);k=torch.exp(-q.square()/(2*op['sigma']**2));k=k/k.sum();kernel=(k[:,None]*k[None,:])[None,None].expand(3,1,5,5)
        return F.conv2d(F.pad(x,(2,2,2,2),mode='replicate'),kernel,groups=3)
    if kind=='resize_pad':
        y=F.interpolate(x,size=op['size'],mode='bilinear',align_corners=False)
        y=F.pad(y,tuple(op['padding']),mode='constant',value=0.)
        return F.interpolate(y,size=x.shape[-2:],mode='bilinear',align_corners=False)
    angle=math.radians(op.get('degrees',0.));h,w=x.shape[-2:]
    # 对非正方形行人图保持像素坐标旋转比例。
    theta=x.new_tensor([[[math.cos(angle),-math.sin(angle)*h/w,2*op.get('dx',0.)],
                         [math.sin(angle)*w/h,math.cos(angle),2*op.get('dy',0.)]]])
    grid=F.affine_grid(theta,x.shape,align_corners=False)
    return F.grid_sample(x,grid,mode='bilinear',padding_mode='border',align_corners=False)

def apply_views(image,plan):
    require(image.ndim==4 and image.shape[:2]==(1,3) and list(image.shape[-2:])==plan['shape'],'recipe original RGB shape')
    h,w=plan['shape'];s=ViewSettings(**plan['settings'])
    require(plan==recipe(s,plan['row_id'],plan['round'],h,w),'complete deterministic recipe identity')
    rows=[];ph,pw=h//s.local_rows,w//s.local_cols
    for row in plan['local']:
        parts=[transform(image[:,:,row['row']*ph:(row['row']+1)*ph,j*pw:(j+1)*pw],op) for j,op in enumerate(row['operations'])]
        rows.append(torch.cat([parts[j] for j in row['permutation']],dim=-1))
    local=torch.cat(rows,dim=-2);global_view=image
    for op in plan['global']:global_view=transform(global_view,op)
    result=torch.cat([image,local,global_view],dim=0).clamp(0.,1.)
    require(torch.isfinite(result).all() and torch.equal(result[:1],image),'finite bounded views, original exact')
    return result
