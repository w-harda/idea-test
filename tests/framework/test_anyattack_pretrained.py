"""免训练迁移的实际协议边界；不调用GPU或神经网络。"""
from types import SimpleNamespace
import pytest,torch
from tbps.anyattack.core import mapping,bridge,EPS,no_training,ArtifactError

@pytest.mark.parametrize('sizes',[(2,2,2,2,2),(1,3,2,4),(5,5),(1,1,1)])
def test_reference_bijection(sizes):
 g=[SimpleNamespace(pid=p) for p,n in enumerate(sizes) for _ in range(n)]
 a=mapping(g);assert a==mapping(g);assert sorted(a)==list(range(len(g)))
 assert all(g[i].pid!=g[j].pid for i,j in enumerate(a))

def test_impossible_reference_rejected():
 with pytest.raises(ArtifactError):mapping([SimpleNamespace(pid=i) for i in [0,0,0,1]])

@pytest.mark.parametrize('value',[-2.,0.,2.])
def test_cpu_geometry_and_endpoints(value):
 original=torch.linspace(0,1,384*128).reshape(1,1,384,128).expand(1,3,384,128).clone()
 x=bridge(original,torch.full((1,3,224,224),value))
 assert x['adversarial'].shape==original.shape
 assert x['adversarial'].min()>=0 and x['adversarial'].max()<=1
 assert float((x['adversarial'].double()-original.double()).abs().max())<=EPS
 assert torch.equal(x['clamped_noise'],torch.full((1,3,224,224),max(-EPS,min(EPS,value))))

def test_noise_shape_and_nonfinite_rejected():
 x=torch.zeros(1,3,384,128)
 with pytest.raises(ArtifactError):bridge(x,torch.zeros_like(x))
 with pytest.raises(ArtifactError):bridge(x,torch.full((1,3,224,224),float('nan')))

def test_cpu_guard_rejects_neural_forward():
 model=torch.nn.Identity()
 with pytest.raises(ArtifactError):
  with no_training(cpu=True):model(torch.zeros(1))

def test_training_optimizer_forbidden():
 with pytest.raises(ArtifactError):
  with no_training():torch.optim.SGD([torch.nn.Parameter(torch.ones(1))],lr=.1)

def test_cpu_guard_tensor_load_only(tmp_path):
 p=tmp_path/'x.pt';torch.save(torch.ones(3),p)
 with no_training(cpu=True):assert torch.equal(torch.load(p,map_location='cpu',weights_only=True),torch.ones(3))
 with pytest.raises(ArtifactError):
  with no_training(cpu=True):torch.load(p,map_location='cpu',weights_only=False)

@pytest.mark.parametrize('sign',[-1.,1.])
def test_all_byte_levels_strict_real_budget(sign):
 original=(torch.arange(384*128)%256).float().div(255).reshape(1,1,384,128).expand(1,3,384,128).clone()
 x=bridge(original,torch.full((1,3,224,224),sign*100))
 assert float((x['adversarial'].double()-original.double()).abs().max())<=8/255
 assert EPS==8/255


def test_projection_rounds_inward_instead_of_relaxing_tolerance():
 from tbps.anyattack.core import strict_project
 original=torch.zeros(1,3,384,128)
 candidate=(original+EPS).clamp(0,1)
 assert float((candidate.double()-original.double()).abs().max())>EPS
 fixed=strict_project(original,candidate)
 assert float((fixed.double()-original.double()).abs().max())<=EPS
 assert (fixed<candidate).any()
