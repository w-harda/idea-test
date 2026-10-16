from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from PIL import Image
from tbps.artifacts import ArtifactError, digest
from tbps.contracts import ModelSpec, FileRef, PublicTextInput, TokenizerSpec
from tbps.config import IRRA, ATTACKS, validate_access, ConfigError
from tbps.models.base import ModelAdapter, ModelSpace, RepresentationBatch, ScoreBatch, InputBinding
from tbps.models.irra import IRRAAdapter, fp32_runtime
from tbps.models.irra_io import IRRATokenizer, pixels_from_pil, normalize_pixels, SIZE, MEAN, STD
from tbps.models.irra_source import IRRASource, REQUIRED, OFFICIAL_IRRA_SHA, load_official_model
H="1"*64
I="2"*64

class ByteFixture:
    encoder={"<|startoftext|>":49406,"<|endoftext|>":49407}
    def encode(self,text):return list(text.encode("utf-8"))
class NativeBackend(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(3,512);self.register_buffer("frozen_buffer",torch.ones(1))
        self.seen=[]
    def encode_image(self,x):
        self.seen.append((torch.is_grad_enabled(),torch.is_inference_mode_enabled(),self.training))
        return self.linear(x.mean(dim=(-1,-2)))
    def encode_text(self,tokens):
        return self.linear(torch.stack((tokens[:,1].float()/256, tokens[:,2].float()/256,torch.ones(len(tokens),device=tokens.device)),dim=-1))
def adapter():
    tokenizer=IRRATokenizer(ByteFixture(),fingerprint=H)
    space=ModelSpace("irra_fixture",I,H,H,H,H,"fp32/math/no_tf32")
    return IRRAAdapter(NativeBackend(),tokenizer,spec=replace(IRRA,id="irra_fixture"),space=space)

class ModelAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);self.a=adapter()
    def test_inference_frozen_view(self):
        x=torch.rand(2,3,*SIZE,requires_grad=True)
        rep=self.a.inference().encode_images(x)
        self.assertFalse(rep.payload.requires_grad);self.assertFalse(torch.is_inference(rep.payload))
        self.assertEqual(self.a._backend.seen[-1],(False,True,False))
        self.assertFalse(self.a.source_verified)
    def test_surrogate_input_gradient_frozen_parameters(self):
        x=torch.rand(1,3,*SIZE,requires_grad=True)
        rep=self.a.surrogate().encode_images(x);rep.payload[:,0].sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(x.grad.abs().sum().item(),0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.a._backend.parameters()))
        self.assertEqual(self.a._backend.seen[-1],(True,False,False))
    def test_parameters_buffers_unchanged(self):
        before={k:v.detach().clone() for k,v in self.a._backend.state_dict().items()}
        x=torch.rand(1,3,*SIZE,requires_grad=True)
        self.a.surrogate().encode_images(x).payload[:,1].sum().backward()
        self.assertTrue(all(torch.equal(before[k],v) for k,v in self.a._backend.state_dict().items()))
    def test_inference_text_constant_can_enter_surrogate_score(self):
        t,_=self.a.inference().encode_texts((PublicTextInput("red"),))
        x=torch.rand(1,3,*SIZE,requires_grad=True)
        image=self.a.surrogate().encode_images(x)
        score=self.a.surrogate().score(t,image);score.payload.sum().backward()
        self.assertIsNotNone(x.grad);self.assertTrue(torch.isfinite(x.grad).all())
    def test_native_scoring_orientation_and_exact_dot(self):
        view=self.a.inference()
        texts,_=view.encode_texts(("red","blue"))
        images=view.encode_images(torch.rand(3,3,*SIZE))
        score=view.score(texts,images)
        self.assertEqual(score.payload.shape,(2,3));self.assertTrue(torch.equal(score.payload,texts.payload@images.payload.T))
        self.assertEqual(score.scoring_protocol_sha256,self.a.space.score_id)
    def test_wrong_space_rejected(self):
        view=self.a.inference();texts,_=view.encode_texts(("red",));images=view.encode_images(torch.rand(1,3,*SIZE))
        with self.assertRaises(ArtifactError):view.score(replace(texts,space_id=I),images)
        with self.assertRaises(ArtifactError):view.score(images,texts)
    def test_wrong_dimension_or_nonunit_feature_rejected(self):
        view=self.a.inference();texts,_=view.encode_texts(("red",));images=view.encode_images(torch.rand(1,3,*SIZE))
        for payload in (torch.ones(1,37),torch.ones(1,512),torch.full((1,512),float("nan"))):
            with self.subTest(shape=payload.shape),self.assertRaises(ArtifactError):view.score(replace(texts,payload=payload),images)
    def test_model_training_mode_or_parameter_unfreeze_rejected(self):
        self.a._backend.train()
        with self.assertRaises(ArtifactError):self.a.inference().encode_images(torch.rand(1,3,*SIZE))
        self.a._backend.eval();next(self.a._backend.parameters()).requires_grad_(True)
        with self.assertRaises(ArtifactError):self.a.surrogate().encode_images(torch.rand(1,3,*SIZE))
    def test_preprocess_exact_and_no_double_normalize(self):
        image=Image.new("RGB",(29,61),(14,117,239));pixels=pixels_from_pil(image)
        from torchvision import transforms as T
        expected=T.Compose([T.Resize(SIZE),T.ToTensor(),T.Normalize(MEAN,STD)])(image)
        self.assertTrue(torch.equal(normalize_pixels(pixels[None])[0],expected))
        with self.assertRaises(ArtifactError):normalize_pixels(expected[None])
    def test_pixel_validation(self):
        for x in (torch.rand(1,3,10,10),torch.rand(1,3,*SIZE).double(),torch.full((1,3,*SIZE),float("nan")),torch.full((1,3,*SIZE),1.1)):
            with self.subTest(shape=x.shape),self.assertRaises(ArtifactError):normalize_pixels(x)
    def test_token_sot_eot_pad_unicode(self):
        p=self.a._tokenizer.inspect("rеd") # 中间是Cyrillic，按原UTF-8而非ASCII替换。
        self.assertEqual(p.ids[0],49406);self.assertEqual(p.ids[p.untruncated_length-1],49407);self.assertEqual(len(p.ids),77)
        self.assertEqual(p.untruncated_length,len("rеd".encode())+2)
    def test_token_truncation_reports_and_final_eot(self):
        p=self.a._tokenizer.inspect("r"*100);self.assertTrue(p.truncated);self.assertEqual(p.ids[-1],49407)
        with self.assertRaises(ArtifactError):self.a._tokenizer.inspect("r"*100,truncate=False)
    def test_direct_token_validation(self):
        for x in (torch.zeros(1,77),torch.zeros(1,76,dtype=torch.long),torch.zeros(1,77,dtype=torch.long)):
            with self.subTest(shape=x.shape),self.assertRaises(ArtifactError):self.a.inference().encode_tokens(x)
    def test_space_changes_for_weights_preprocess_tokenizer_protocol(self):
        space=self.a.space
        for field in ("checkpoint_sha256","sources_sha256","preprocess_sha256","tokenizer_sha256","scoring_protocol_sha256"):
            other=replace(space,**{field:"3"*64})
            self.assertNotEqual(space.id,other.id);self.assertNotEqual(space.score_id,other.score_id)
    def test_feature_score_artifact_identity(self):
        rep=self.a.inference().encode_images(torch.rand(1,3,*SIZE));content=FileRef("feature.pt",H)
        inputs=InputBinding(self.a.space.id,H,(H,),"image",("g",))
        art=self.a.representation_artifact("rep",rep,content,inputs=inputs).artifact
        self.assertEqual(art.space_id,self.a.space.id);self.assertEqual(art.shape,(1,512))
        score=self.a.score_artifact("score",FileRef("scores.pt",H),("q",),("g",),
            query_inputs=replace(inputs,modality="text",ordered_ids=("q",)),gallery_inputs=inputs).artifact
        self.assertEqual(score.scoring_protocol_sha256,self.a.space.score_id)
        with self.assertRaises(ArtifactError):self.a.representation_artifact("foreign",replace(rep,space_id=I),content,inputs=inputs)
    def test_injected_backend_cannot_claim_real_irra(self):
        with self.assertRaises(ArtifactError):IRRAAdapter(NativeBackend(),self.a._tokenizer,spec=IRRA,space=replace(self.a.space,model_id="irra"))
    def test_runtime_flags_restored_on_exception(self):
        original=(torch.backends.cuda.matmul.allow_tf32,torch.backends.mha.get_fastpath_enabled())
        with self.assertRaises(RuntimeError):
            with fp32_runtime():raise RuntimeError("fixture")
        self.assertEqual(original,(torch.backends.cuda.matmul.allow_tf32,torch.backends.mha.get_fastpath_enabled()))
    def test_no_cuda_initialized(self):self.assertFalse(torch.cuda.is_initialized())
    def test_authorization_gate_before_loading_or_tokenizer(self):
        with patch("tbps.models.irra.load_official_model",side_effect=AssertionError("forbidden")):
            with self.assertRaises(ArtifactError):IRRAAdapter.from_source(None,device="cuda:0")
        with self.assertRaises(ArtifactError):load_official_model(None,device="cuda:0")
    def test_fusion_mock_not_forced_to_dual_tower(self):
        class Fusion:
            spec=ModelSpec("fusion-fixture","v1",(123,87),TokenizerSpec("other",H,128,"word","reject"),None,"fusion","native",frozenset({"fusion","score"}))
            space=ModelSpace("fusion-fixture",H,H,H,H,H,"fixture")
            def score(self,query,gallery):
                return ScoreBatch([[len(t)*len(g) for g in gallery] for t in query],self.space.id,self.space.score_id)
        m=Fusion();self.assertIsInstance(m,ModelAdapter);self.assertFalse(hasattr(m,"encode_images"))
        self.assertEqual(m.score(("red","blue"),("x","xy")).payload,[[3,6],[4,8]])
        self.assertIsNone(m.spec.representation_dim)
    def test_non_512_native_representation_contract(self):
        spec=ModelSpec("dim37","v1",None,None,37,"dot","none",frozenset({"score"}))
        rep=RepresentationBatch(torch.zeros(2,37),H,"native")
        self.assertEqual(rep.payload.shape[-1],spec.representation_dim)
    def test_joint_declares_complete_component_permissions(self):
        a=ATTACKS["joint"];required=ATTACKS["gv2"].train_requirements|ATTACKS["hv1"].train_requirements
        self.assertEqual(a.train_requirements,required)
        validate_access("white_box",a,IRRA,IRRA,H,H,"a","a",required,a.attack_requirements)
    def test_joint_missing_permission_refused(self):
        a=ATTACKS["joint"]
        with self.assertRaises(ConfigError):validate_access("white_box",a,IRRA,IRRA,H,H,"a","a",a.train_requirements-{"train_pids"},a.attack_requirements)
    def test_joint_incomplete_spec_refused_even_with_full_grants(self):
        a=replace(ATTACKS["joint"],train_requirements=frozenset({"train_pairs","source_rewards"}))
        with self.assertRaises(ConfigError):validate_access("white_box",a,IRRA,IRRA,H,H,"a","a",ATTACKS["joint"].train_requirements,a.attack_requirements)
    def test_import_and_load_gate_have_no_model_side_effects(self):
        script="""import builtins,sys
old=builtins.__import__
def guard(name,*a,**k):
 if name.split('.')[0] in {'torch','torchvision','model','prototypes','transformers'}:raise AssertionError(name)
 return old(name,*a,**k)
builtins.__import__=guard
import tbps.preflight, tbps.datasets.cuhk_pedes, tbps.models.irra
from tbps.artifacts import ArtifactError
try:tbps.models.irra.IRRAAdapter.from_source(None,device='cuda:0')
except ArtifactError:pass
else:raise AssertionError('load permission bypass')
assert 'torch' not in sys.modules
print('NO_MODEL_IMPORT_OR_LOAD')
"""
        r=subprocess.run([sys.executable,"-B","-c",script],capture_output=True,text=True,check=True)
        self.assertIn("NO_MODEL_IMPORT_OR_LOAD",r.stdout)

    def test_dataset_input_order_and_edits_isolate_cache(self):
        b=InputBinding(self.a.space.id,H,(H,I),"text",("q1","q2"))
        for other in (replace(b,dataset_manifest_sha256=I),replace(b,ordered_sha256=(I,H)),
                      replace(b,ordered_sha256=(H,H)),replace(b,ordered_ids=("q2","q1")),replace(b,model_space_id=H)):
            self.assertNotEqual(b.cache_id,other.cache_id)
    def test_artifact_input_count_or_modality_mismatch_rejected(self):
        rep=self.a.inference().encode_images(torch.rand(1,3,*SIZE))
        for b in (InputBinding(self.a.space.id,H,(H,I),"image",("g1","g2")),InputBinding(self.a.space.id,H,(H,),"text",("q",))):
            with self.subTest(b=b),self.assertRaises(ValueError):
                self.a.representation_artifact("rep",rep,FileRef("x",H),inputs=b)
    def test_score_cache_binds_query_edits_not_only_query_id(self):
        t=InputBinding(self.a.space.id,H,(H,),"text",("q",));g=InputBinding(self.a.space.id,H,(H,),"image",("g",))
        one=self.a.score_artifact("s",FileRef("score",H),("q",),("g",),query_inputs=t,gallery_inputs=g)
        two=self.a.score_artifact("s",FileRef("score",H),("q",),("g",),query_inputs=replace(t,ordered_sha256=(I,)),gallery_inputs=g)
        self.assertNotEqual(one.cache_id,two.cache_id)

    def test_inference_output_normal_even_inside_outer_inference_mode(self):
        with torch.inference_mode():
            rep=self.a.inference().encode_images(torch.rand(1,3,*SIZE))
        self.assertFalse(torch.is_inference(rep.payload));self.assertFalse(rep.payload.requires_grad)
    def test_surrogate_explicitly_overrides_outer_inference_mode(self):
        x=torch.rand(1,3,*SIZE,requires_grad=True)
        with torch.inference_mode():
            rep=self.a.surrogate().encode_images(x)
        self.assertTrue(rep.payload.requires_grad)
        rep.payload[:,0].sum().backward()
        self.assertGreater(x.grad.abs().sum().item(),0)

    def test_fp32_contract_disables_outer_autocast(self):
        x=torch.rand(1,3,*SIZE)
        expected=self.a.inference().encode_images(x)
        text,_=self.a.inference().encode_texts(("red",))
        reference=self.a.inference().score(text,expected)
        with torch.autocast("cpu",dtype=torch.bfloat16):
            actual=self.a.inference().encode_images(x)
            scores=self.a.inference().score(text,actual)
        self.assertEqual(scores.payload.dtype,torch.float32)
        self.assertTrue(torch.equal(actual.payload,expected.payload))
        self.assertTrue(torch.equal(scores.payload,reference.payload))

if __name__=="__main__":unittest.main()
