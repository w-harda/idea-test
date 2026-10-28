"""作者真实UAP只读接线回归：不加载神经模型、不初始化CUDA。"""
from pathlib import Path
import ast,json,hashlib
import pytest,torch
import torch.nn.functional as F
from tbps.artifacts import ArtifactError
from tbps.engines.provenance import payload_digest
W=Path(__file__).resolve().parents[2]
@pytest.fixture
def c():return json.loads((W/"configs/framework/cpgc_sep_tbps.uap_detach.server.json").read_text())
@pytest.fixture(autouse=True)
def prohibit_gpu_and_updates(monkeypatch):
 def forbidden(*a,**k):raise AssertionError("No CUDA/backward/Optimizer allowed in UAP CPU tests")
 monkeypatch.setattr(torch.cuda,"_lazy_init",forbidden)
 monkeypatch.setattr(torch.Tensor,"backward",forbidden)
 monkeypatch.setattr(torch.autograd,"grad",forbidden)
 for cls in [torch.optim.Optimizer,torch.optim.Adam,torch.optim.SGD]:monkeypatch.setattr(cls,"step",forbidden)
 torch.set_num_threads(1)
def load(c):return torch.load(c["resources"]["cpgc_noise"]["path"],map_location="cpu",weights_only=True)
def test_author_actual_uap_values_file_and_stable_digest(c):
 p=Path(c["resources"]["cpgc_noise"]["path"]);before=p.stat();sha=hashlib.sha256(p.read_bytes()).hexdigest()
 original=load(c);frozen=original.detach()
 assert original.requires_grad and not frozen.requires_grad
 assert torch.equal(original,frozen) and original.shape==frozen.shape==(3,384,384)
 assert original.dtype==frozen.dtype==torch.float32 and frozen.device.type=="cpu"
 assert payload_digest(frozen)==payload_digest(load(c).detach())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==sha==c["resources"]["cpgc_noise"]["sha256"]
 after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 assert not torch.cuda.is_initialized()
def test_original_illegal_author_tensor_still_rejected(c):
 with pytest.raises(ArtifactError,match="unsupported native payload"):payload_digest(load(c))
 with pytest.raises(ArtifactError):payload_digest(torch.empty(1,device="meta"))
@pytest.mark.parametrize("module",["target","audit"])
@pytest.mark.parametrize("victim",["irra","rasa"])
def test_actual_loading_statements_detach_before_interpolate_and_hash(c,module,victim):
 tree=ast.parse((W/("src/tbps/baselines/cpgc_sep/"+module+".py")).read_text())
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==("evaluate" if module=="target" else "audit"))
 block=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.unparse(n.test)=="method == 'cpgc'" and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ("raw_noise","noise") for t in x.targets) for x in n.body))
 if module=="target":
  nodes=[n for n in block.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ("raw_noise","target_size","noise") for t in n.targets)]
 else:nodes=[n for n in block.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="noise" for t in n.targets) or isinstance(n,ast.If)]
 assert ".detach()" in ast.unparse(nodes[0])
 scope={"torch":torch,"F":F,"c":c,"victim":victim}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),"actual_UAP_loading_bridge","exec"),scope)
 raw=load(c);expected=raw.detach().unsqueeze(0)
 if victim=="irra":expected=F.interpolate(expected,size=(384,128),mode="bilinear",align_corners=False)
 actual=scope["noise"]
 assert torch.equal(actual,expected) and not actual.requires_grad and actual.grad_fn is None
 assert payload_digest(actual)==payload_digest(expected)
 assert not torch.cuda.is_initialized()
