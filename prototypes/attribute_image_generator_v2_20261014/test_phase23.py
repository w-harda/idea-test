"""真实接线的CPU单元测试；不加载官方模型或初始化CUDA。"""
import ast,copy,sys,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
from irra_image_adapter import FrozenIRRAImageEncoder,pixels_from_pil,MEAN,STD,SIZE,load_official
from phase23_inputs import check_sample
import phase23_inputs as inputs
import cache_io as io
torch.set_num_threads(2)

class Toy(nn.Module):
    def __init__(self):
        super().__init__(); self.projection=nn.Linear(3,512,bias=False); self.last=None
    def encode_image(self,x):
        self.last=x.detach().clone()
        return self.projection(x.mean(dim=(2,3)))
    def encode_text(self,*args): raise AssertionError("text forbidden")

class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); self.model=Toy(); self.encoder=FrozenIRRAImageEncoder(self.model)
        self.image=torch.rand(1,3,*SIZE)
    def test_shape_and_unit_feature(self):
        f=self.encoder(self.image)
        self.assertEqual(f.shape,(1,512)); self.assertTrue(torch.allclose(f.norm(dim=1),torch.ones(1)))
    def test_preprocessing_matches_official_Normalize(self):
        self.encoder(self.image)
        ref=transforms.Normalize(MEAN,STD)(self.image[0])[None]
        self.assertTrue(torch.equal(self.model.last,ref))
    def test_direct_official_encode_exact(self):
        ref=F.normalize(self.model.encode_image(transforms.Normalize(MEAN,STD)(self.image[0])[None]),dim=-1)
        self.assertTrue(torch.equal(self.encoder(self.image),ref))
    def test_input_gradient_and_frozen_parameters(self):
        image=self.image.requires_grad_()
        self.encoder(image)[:,0].sum().backward()
        self.assertGreater(float(image.grad.norm()),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.encoder.parameters()))
    def test_train_stays_eval(self):
        self.encoder.train(True)
        self.assertFalse(self.encoder.training); self.assertFalse(self.model.training)
    def test_no_buffer_mutation(self):
        before={k:v.clone() for k,v in self.encoder.state_dict().items()}
        self.encoder(self.image)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in self.encoder.state_dict().items()))
    def test_wrong_resolution_rejected(self):
        with self.assertRaises(ValueError): self.encoder(torch.rand(1,3,128,384))
    def test_wrong_dtype_rejected(self):
        with self.assertRaises(ValueError): self.encoder(self.image.double())
    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError): self.encoder(self.image+1)
    def test_invalid_output_shape_rejected(self):
        self.model.encode_image=lambda x:torch.ones(1,511)
        with self.assertRaises(ValueError): self.encoder(self.image)
    def test_nonfinite_output_rejected(self):
        self.model.encode_image=lambda x:torch.full((1,512),float("nan"))
        with self.assertRaises(ValueError): self.encoder(self.image)
    def test_zero_output_rejected(self):
        self.model.encode_image=lambda x:torch.zeros(1,512)
        with self.assertRaises(ValueError): self.encoder(self.image)
    def test_PIL_pixels_exact_and_RGB(self):
        for mode in ("L","RGB"):
            im=Image.new(mode,(21,47),color=80)
            got=pixels_from_pil(im)
            expected=transforms.Compose([transforms.Resize(SIZE),transforms.ToTensor()])(im.convert("RGB"))
            self.assertTrue(torch.equal(got,expected)); self.assertEqual(got.shape,(3,*SIZE))
    def test_checkpoint_mismatch_stops_before_loading(self):
        with patch("irra_image_adapter.io.verify",side_effect=ValueError("SHA")), \
             patch("irra_image_adapter.importlib.import_module",side_effect=AssertionError("must not import")):
            with self.assertRaises(ValueError): load_official()

def fixture():
    identity={"row_id":"cuhk:0:0","image_rel_path":"train_query/x.jpg","caption_index":0,
        "split":"train","role":"train","pid":1,"caption":"caption","K":1}
    protocol={"scope":"test"}
    source={"row_id":"cuhk:0:0","pid":1,"caption":"caption","image":"train_query/x.jpg","K":1,"image_sha256":"hash"}
    record={"row_id":"cuhk:0:0","pid":1,"caption":"caption","paired_image":"train_query/x.jpg",
        "paired_gallery_index":0,"zero_intervention_exact":True,"retrieval_backward_count":1}
    meta={**identity,"source_entry":source,"source_score_record":record,"source_image_sha256":"hash",
          "protocol_hash":io.digest(protocol)}
    raw=[{"split":"train","id":1,"file_path":"train_query/x.jpg","captions":["caption","other"]}]
    split={"records":[{"row_id":"cuhk:0:0"}],"train_indices":[0],"validation_indices":[]}
    gallery={"spec":{"gallery_paths":["train_query/x.jpg"]}}
    return [identity,copy.deepcopy(identity),meta,protocol,raw,split,gallery]

class PairingTests(unittest.TestCase):
    def check(self,args):
        with patch("phase23_inputs.io.verify"): return check_sample(*args)
    def test_complete_identity_passes(self):
        args=fixture(); _,index=self.check(args); self.assertEqual(index,0)
    def test_raw_caption_pair_rejected(self):
        args=fixture(); args[4][0]["captions"][0]="wrong"
        with self.assertRaises(ValueError): self.check(args)
    def test_raw_PID_rejected(self):
        args=fixture(); args[4][0]["id"]=99
        with self.assertRaises(ValueError): self.check(args)
    def test_image_pair_rejected(self):
        args=fixture(); args[4][0]["file_path"]="wrong.jpg"
        with self.assertRaises(ValueError): self.check(args)
    def test_caption_index_rejected(self):
        args=fixture(); args[0]["caption_index"]=1
        with self.assertRaises(ValueError): self.check(args)
    def test_validation_membership_rejected(self):
        args=fixture()
        for i in (0,1,2): args[i]["role"]="validation"
        with self.assertRaises(ValueError): self.check(args)
    def test_protocol_mismatch_rejected(self):
        args=fixture(); args[2]["protocol_hash"]="wrong"
        with self.assertRaises(ValueError): self.check(args)
    def test_score_identity_rejected(self):
        args=fixture(); args[2]["source_score_record"]["caption"]="wrong"
        with self.assertRaises(ValueError): self.check(args)
    def test_source_entry_identity_rejected(self):
        args=fixture(); args[2]["source_entry"]["row_id"]="wrong"
        with self.assertRaises(ValueError): self.check(args)
    def test_image_SHA_provenance_rejected(self):
        args=fixture(); args[2]["source_image_sha256"]="wrong"
        with self.assertRaises(ValueError): self.check(args)
    def test_gallery_index_pair_rejected(self):
        args=fixture(); args[6]["spec"]["gallery_paths"][0]="wrong.jpg"
        with self.assertRaises(ValueError): self.check(args)

class GatesTests(unittest.TestCase):
    def test_protocol_JSON_roundtrip_equivalent(self):
        import json
        from smoke_phase23 import same_protocol
        protocol={"samples":[("row","train",0)],"source_sha256":{"source":"hash"}}
        self.assertTrue(same_protocol(json.loads(json.dumps(protocol)),protocol))
    def test_protocol_source_change_rejected(self):
        from smoke_phase23 import same_protocol
        self.assertFalse(same_protocol({"source_sha256":{"source":"old"}},
                                       {"source_sha256":{"source":"new"}}))
    def test_default_no_GPU(self):
        import smoke_phase23 as smoke
        with patch.object(smoke,"audit_inputs",return_value=([],{},{})), \
             patch.object(smoke,"run",side_effect=AssertionError("GPU forbidden")):
            smoke.main([])
    def test_no_frontend_imports(self):
        forbidden={"integration","common","conditioning","runtime","deps","attribute_tta","selection_audit","generator_model"}
        for name in ("irra_image_adapter.py","phase23_inputs.py","smoke_phase23.py"):
            tree=ast.parse((io.ROOT/name).read_text()); names=set()
            for n in ast.walk(tree):
                if isinstance(n,ast.Import): names.update(a.name.split(".")[0] for a in n.names)
                elif isinstance(n,ast.ImportFrom): names.add(n.module.split(".")[0])
            self.assertFalse(names&forbidden)
    def test_only_five_fixed_samples(self):
        self.assertEqual([e[2] for e in inputs.EXPECTED],[0,1,3,7,4])
        self.assertEqual([e[1] for e in inputs.EXPECTED],["train"]*4+["validation"])

if __name__=="__main__": unittest.main(verbosity=2)
