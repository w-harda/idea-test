"""真实官方DRA算法的合成CPU接口测试；无GPU、真实模型或攻击数据。"""
import ast
import json
import tempfile
from dataclasses import replace
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from tbps.artifacts import FileRef, ArtifactError, sha_file
from tbps.baselines.dra_official import modules, PassiveTrace, PARAMETERS, projection_check, batch_members, verify_official
from tbps.baselines.sga_official_io import load_file
from tbps.models.dra_irra_source import IRRADRASource
from tbps.models.sga_irra_source import AuditedBert, decode_bert

ROOT=Path("/home/lzf/ldx/projects/VLPTransferAttack-official")
BERT="/home/lzf/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
def official():
    return load_file(FileRef(str(ROOT/"RAttacker.py"),sha_file(ROOT/"RAttacker.py")),"_dra_cpu")
def tokenizer():
    m=load_file(FileRef(str(ROOT/"models/tokenization_bert.py"),sha_file(ROOT/"models/tokenization_bert.py")),"_dra_tok_cpu")
    return AuditedBert(m.BertTokenizer.from_pretrained(BERT,local_files_only=True))

class NativeTokens:
    def __init__(self, tok):self.tok=tok
    def batch(self,texts,truncate=False):
        from types import SimpleNamespace
        values=[self.tok.encode(t,add_special_tokens=False) for t in texts]
        if any(len(v)>77 for v in values):raise ArtifactError("additional IRRA truncation")
        return torch.tensor([v+[0]*(77-len(v)) for v in values]),[SimpleNamespace(truncated=False) for _ in values]
class TinyBackend(torch.nn.Module):
    def __init__(self):
        super().__init__();self.image=torch.nn.Linear(3,512);self.text=torch.nn.Linear(1,512)
    def encode_image(self,x):return self.image(x.mean((2,3))).tanh()
    def encode_text(self,x):return self.text(x.float().mean(1,keepdim=True)/30522).tanh()
class FakeMLM(torch.nn.Module):
    def __init__(self,tok):
        super().__init__();self.register_buffer("anchor",torch.zeros(()));self.words=[tok.convert_tokens_to_ids(x) for x in ["blue","black","red","white","shirt","pants","woman","man","coat","hat"]]
    @property
    def device(self):return self.anchor.device
    def forward(self,ids,attention_mask=None):
        from types import SimpleNamespace
        logits=torch.full((*ids.shape,30522),-20.)
        for n,w in enumerate(self.words):logits[:,:,w]=10.-n*.1
        class Output(SimpleNamespace):
            def __getitem__(self,k):return self.logits
        return Output(logits=logits)
class DRAInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.token=tokenizer()
    def setUp(self):
        torch.manual_seed(42);random.seed(42);np.random.seed(42)
        self.source=IRRADRASource(TinyBackend(),NativeTokens(self.token),self.token)
    def test_fixed_core(self):
        self.assertEqual(sha_file(ROOT/"RAttacker.py"),"75518cb9465d9f8e994ea2f59c86b89fd4c39e3fe74f66b7a1d0a090eec2d313")
    def test_image_input_gradient(self):
        x=torch.rand(1,3,16,8,requires_grad=True)
        z=self.source.inference_image(x)["image_feat"]
        g,=torch.autograd.grad(z[0,0],x)
        self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0)
    def test_text_bridge_77_and_specials(self):
        for text in ["a man wearing a blue shirt","a [UNK] wearing red pants","a man's black coat"]:
            b=self.token([text],padding="max_length",truncation=True,max_length=77,return_tensors="pt")
            r=decode_bert(b,self.token)[0]
            self.assertTrue(r["wordpiece_roundtrip_exact"])
            self.assertEqual(tuple(self.source.inference_text(b)["text_feat"].shape),(1,512))
    def test_extra_context_rejected(self):
        class TooLong:
            def batch(self,*a,**k):raise ArtifactError("additional IRRA truncation")
        self.source.irra_tokenizer=TooLong()
        b=self.token(["blue shirt"],padding="max_length",max_length=77,truncation=True,return_tensors="pt")
        with self.assertRaisesRegex(ArtifactError,"truncation"):self.source.inference_text(b)
    def test_native_77_boundary(self):
        self.assertEqual(NativeTokens(self.token).batch(["man "*77])[0].shape,(1,77))
        with self.assertRaisesRegex(ArtifactError,"truncation"):NativeTokens(self.token).batch(["man "*78])
    def test_real_official_image_loop_frozen_flags_incompatible(self):
        image=official().ImageAttacker(lambda x:x,eps=8/255,steps=10,step_size=2/255,sample_numbers=5)
        self.source.backend.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError,"does not require grad"):
            image.txt_guided_attack(self.source,torch.rand(2,3,16,8),[0,1],"cpu",scales=PARAMETERS["scales"],txt_embeds=torch.rand(2,512))
    def test_real_official_complete_attack_and_unchanged_weights(self):
        o=official();mlm=FakeMLM(self.token)
        a=o.Attacker(self.source,o.ImageAttacker(lambda x:x,eps=8/255,steps=10,step_size=2/255,sample_numbers=5),
            o.TextAttacker(mlm,self.token,cls=False,max_length=77,number_perturbation=1,topk=10,threshold_pred_score=.3,text_ratios=[.6,.2,.2]))
        self.assertIs(a.attack.__func__,o.Attacker.attack)
        before={k:v.clone() for k,v in self.source.backend.state_dict().items()}
        x=torch.rand(2,3,16,8)
        trace=PassiveTrace(ROOT/"RAttacker.py",self.source)
        with trace.active():
            adv,texts,_=a.attack(x,["a man in a blue shirt","a woman wearing red pants"],[0,1],"cpu",max_length=77,scales=PARAMETERS["scales"])
        r=trace.report()
        self.assertEqual(r["events"],["text1","image","text2"])
        self.assertEqual(r["counts"]["call:rand3Num"],45)
        self.assertEqual(r["counts"]["backward"],100)
        self.assertEqual(len(r["selected_ratios"]),9)
        self.assertTrue(r["all_image_gradients_finite"])
        self.assertTrue(projection_check(x,adv)["official_FP32_bounds"])
        self.assertEqual(len(texts),2)
        for name,v in self.source.backend.state_dict().items():self.assertTrue(torch.equal(v,before[name]))
        self.assertIsNone(sys.gettrace());self.assertFalse(torch.cuda.is_initialized())
    def test_observer_does_not_replace_or_change_output_or_rng(self):
        def run(observe):
            torch.manual_seed(5);random.seed(5);np.random.seed(5)
            source=IRRADRASource(TinyBackend(),NativeTokens(self.token),self.token)
            image=official().ImageAttacker(lambda x:x,eps=8/255,steps=10,step_size=2/255,sample_numbers=5)
            method=image.txt_guided_attack.__func__
            x=torch.rand(2,3,16,8);text=torch.rand(2,512)
            if observe:
                with PassiveTrace(ROOT/"RAttacker.py").active():value=image.txt_guided_attack(source,x,[0,1],"cpu",txt_embeds=text)
            else:value=image.txt_guided_attack(source,x,[0,1],"cpu",txt_embeds=text)
            self.assertIs(method,image.txt_guided_attack.__func__)
            return value,torch.get_rng_state(),random.getstate(),np.random.get_state()
        a,b=run(False),run(True)
        for x,y in zip(a[0],b[0]):self.assertTrue(torch.equal(x,y))
        self.assertTrue(torch.equal(a[1],b[1]));self.assertEqual(a[2],b[2]);self.assertTrue(np.array_equal(a[3][1],b[3][1]))
    def test_tracer_restored_on_failure(self):
        with self.assertRaisesRegex(RuntimeError,"fixture"):
            with PassiveTrace(ROOT/"RAttacker.py").active():raise RuntimeError("fixture")
        self.assertIsNone(sys.gettrace())
    def test_final_projection_rejects_violation(self):
        x=torch.full((1,3,16,8),.5)
        with self.assertRaises(ArtifactError):projection_check(x,x+.1)
    def test_wrapper_contains_no_attacker_operation(self):
        tree=ast.parse((Path(__file__).parents[2]/"src/tbps/models/dra_irra_source.py").read_text())
        self.assertFalse(any(isinstance(n,ast.For) or isinstance(n,ast.While) for n in ast.walk(tree)))
        self.assertEqual([n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)],["__init__"])
    def test_real_byte_bpe_77_roundtrip_candidates(self):
        from tbps.models.irra_io import official_tokenizer
        from tbps.models.irra_source import IRRASource
        c=json.loads((Path(__file__).parents[2]/"configs/framework/dra_irra_audit.server.json").read_text())
        bpe=official_tokenizer(IRRASource.from_dict(c["irra"]))
        for text in ["a man wearing a blue shirt","a [UNK] wearing red pants","a [CLS] woman in a coat",
                     "white shirt [SEP]","black jacket .","[MASK] dress"]:
            ids=self.token([text],padding="max_length",truncation=True,max_length=77,return_tensors="pt")
            row=decode_bert(ids,self.token)[0]
            value,reports=bpe.batch([row["decoded_text"]],truncate=False)
            self.assertEqual(value.shape,(1,77));self.assertFalse(reports[0].truncated)
    def test_true_full_mapping_and_ten_caption_member(self):
        from tbps.baselines.dra_official import ordered_dataset
        c=json.loads((Path(__file__).parents[2]/"configs/framework/dra_irra_audit.server.json").read_text())
        _,raw,matched=ordered_dataset(c,"full")
        self.assertEqual((len(raw.gallery),len(raw.queries)),(3074,6156))
        counts=[len(batch_members(raw,i,1)[1]) for i in range(3074)]
        self.assertEqual(counts.count(10),1);self.assertEqual(counts.count(2),3073)
        index=counts.index(10);g,q,m=batch_members(raw,index,1)
        self.assertEqual(m,[0]*10)
        self.assertTrue(all(x.paired_image_id==g[0].image_id for x in q))
        self.assertEqual([q.query_id for q in raw.queries],[q.query_id for q in matched.queries])
    def test_source_sha_tamper_rejected(self):
        from tbps.baselines.dra_irra import configuration
        c=json.loads((Path(__file__).parents[2]/"configs/framework/dra_irra_audit.server.json").read_text())
        c["official"]["attacker"]["sha256"]="0"*64
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"bad.json";path.write_text(json.dumps(c))
            with self.assertRaises(ArtifactError):configuration(path)
    def test_parameter_replacement_rejected(self):
        from tbps.baselines.dra_irra import configuration
        c=json.loads((Path(__file__).parents[2]/"configs/framework/dra_irra_audit.server.json").read_text())
        c["parameters"]["steps"]=9
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"bad.json";path.write_text(json.dumps(c))
            with self.assertRaisesRegex(ArtifactError,"fixed official"):configuration(path)
    def test_compact_pixels_immutable(self):
        from tbps.baselines.dra_irra import save_pixels
        from tbps.artifacts import ArtifactStore
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/"tbps-framework-dra-fixture";store=ArtifactStore(base/"run",new_base=base)
            x=torch.arange(1000.).view(100,10)[:1]
            save_pixels(store,"batch",{"x":x},{},())
            got=torch.load(store.root/"batch/pixels.pt",map_location="cpu",weights_only=True)["x"]
            self.assertTrue(torch.equal(got,x));self.assertEqual(got.untyped_storage().nbytes(),got.numel()*4)
            with self.assertRaises(ArtifactError):save_pixels(store,"batch",{"x":x},{},())
    def test_default_and_full_gpu_not_available(self):
        from tbps.baselines.dra_irra import main
        config=Path(__file__).parents[2]/"configs/framework/dra_irra_audit.server.json"
        with self.assertRaisesRegex(ArtifactError,"explicit"):main(["--config",str(config),"--action","source"])
        with self.assertRaises(SystemExit):main(["--config",str(config),"--scope","full"])
        self.assertFalse(torch.cuda.is_initialized())
    def test_new_runner_never_monkey_patches_official(self):
        p=Path(__file__).parents[2]
        for file in [p/"src/tbps/baselines/dra_irra.py",p/"src/tbps/baselines/dra_official.py"]:
            tree=ast.parse(file.read_text())
            for n in ast.walk(tree):
                if isinstance(n,(ast.Assign,ast.AnnAssign)):
                    targets=n.targets if isinstance(n,ast.Assign) else [n.target]
                    self.assertFalse(any(isinstance(t,ast.Attribute) and t.attr in
                        ("attack","txt_guided_attack","img_guided_attack","loss_func","rand3Num") for t in targets))
    def test_real_irra_import_closure_without_model_load_or_cuda(self):
        from tbps.models.irra_source import IRRASource,official_model_modules
        c=json.loads((Path(__file__).parents[2]/"configs/framework/dra_irra_audit.server.json").read_text())
        with patch("torch.load",side_effect=AssertionError("CPU preflight cannot load weights")), patch("torch.cuda._lazy_init",side_effect=AssertionError("no CUDA")):
            with official_model_modules(IRRASource.from_dict(c["irra"])) as (build,clip):
                self.assertTrue(callable(build.build_model))
        self.assertFalse(torch.cuda.is_initialized())
    def test_original_multiword_candidate_is_one_official_slot(self):
        o=official()
        source=IRRADRASource(TinyBackend(),NativeTokens(self.token),self.token)
        attacker=o.Attacker(source,o.ImageAttacker(lambda x:x,eps=8/255,steps=10,step_size=2/255,sample_numbers=5),
            o.TextAttacker(FakeMLM(self.token),self.token,cls=False,max_length=77,number_perturbation=1,topk=10,threshold_pred_score=.3,text_ratios=[.6,.2,.2]))
        trace=PassiveTrace(ROOT/"RAttacker.py",source)
        with trace.active():
            adv,words,_=attacker.attack(torch.rand(1,3,16,8),["leggings"],[0],"cpu",max_length=77,scales=PARAMETERS["scales"])
        action=[r for r in trace.report()["text_actions"] if r["stage"]==2][0]
        self.assertEqual(action["official_change_count"],1)
        self.assertEqual(len(action["final_word_slots"]),1)
        self.assertGreater(len(words[0].split(" ")),1)
        self.assertEqual(words[0]," ".join(action["final_word_slots"]))
if __name__=="__main__":unittest.main()
