"""H-v1合成CPU契约、掩码与三个Head测试。"""
import dataclasses, inspect, json, unittest
from unittest.mock import patch
import torch
from contracts import AttributeSpan, TextExample, from_text_spans
from tokenizer_probe import TokenizerProbe
from constraints import CONFUSABLES, REPLACEMENTS, make_edit, replay, candidates
from policy import CharacterPolicy, choose, natural_k

torch.set_num_threads(2)

class ByteTokenizer:
    encoder = {"<|startoftext|>": 1001, "<|endoftext|>": 1002}
    def encode(self, text):
        return [b+1 for b in text.encode("utf-8")]

def example(caption="coat", start=0, end=None, region="upper"):
    end = len(caption) if end is None else end
    return TextExample(caption, (AttributeSpan("garment_type", "coat", region,
                                               caption[start:end], start, end),))
def model():
    return CharacterPolicy(TokenizerProbe(ByteTokenizer()), width=16, layers=1, heads=4)

class ContractTests(unittest.TestCase):
    def test_exact_source_and_canonical_alias_preserved(self):
        x = example("a tee", 2, 5)
        self.assertEqual(x.attributes[0].canonical_value, "coat")
        self.assertEqual(x.caption[2:5], x.attributes[0].raw_text)
    def test_no_string_relocation_for_repeated_word(self):
        x = TextExample("red red", (AttributeSpan("color", "red", "upper", "red", 4, 7),))
        self.assertEqual(x.attributes[0].char_start, 4)
        with self.assertRaises(ValueError):
            TextExample("red blue", (AttributeSpan("color", "red", "upper", "red", 4, 7),))
    def test_bad_span_types_and_bounds(self):
        for a,b in ((-1,2),(0,10),(True,4),(2,2)):
            with self.subTest(a=a,b=b),self.assertRaises(ValueError):
                TextExample("coat",(AttributeSpan("color","blue","upper","coat",a,b),))
    def test_duplicate_source_rejected(self):
        a=example().attributes[0]
        with self.assertRaises(ValueError):TextExample("coat",(a,a))
    def test_unknown_axis_or_region_rejected(self):
        for axis,region in (("PID","upper"),("color","test_image")):
            with self.assertRaises(ValueError):
                TextExample("coat",(AttributeSpan(axis,"coat",region,"coat",0,4),))
    def test_only_explicit_text_fields(self):
        a=dataclasses.asdict(example().attributes[0])
        self.assertEqual(from_text_spans("coat",[a]),example())
        for key in ("pid","row_id","image","S_raw","paired_index","label","JVP"):
            with self.subTest(key=key),self.assertRaises(ValueError):
                from_text_spans("coat",[{**a,key:123}])
    def test_text_example_cannot_accept_pairing_fields(self):
        for key in ("pid","image","paired_index","scores"):
            with self.subTest(key=key),self.assertRaises(TypeError):
                TextExample("coat",example().attributes,**{key:1})
    def test_edit_codepoints_and_replay_roundtrip(self):
        x=example();e=make_edit(x,0,1,"о")
        self.assertEqual(e.original_codepoint,"U+006F")
        self.assertEqual(e.replacement_codepoint,"U+043E")
        self.assertEqual(replay(x,(e,),1),"cоat")
        self.assertEqual(len(replay(x,(e,),1)),len(x.caption))
        json.dumps(dataclasses.asdict(e),ensure_ascii=False)
    def test_original_unicode_byte_offsets(self):
        x=example("é coat",2,6);e=make_edit(x,0,3,"ο")
        self.assertEqual(e.position,3)
        self.assertEqual(e.original_utf8_offset,4)
        self.assertEqual(e.position_in_attribute,1)
    def test_illegal_replacement_and_outside_span(self):
        x=example("a coat",2,6)
        for pos,replacement in ((0,"а"),(3,"z"),(-1,"о"),(99,"о"),(True,"о")):
            with self.subTest(pos=pos),self.assertRaises(ValueError):
                make_edit(x,0,pos,replacement)
    def test_trace_tampering_rejected(self):
        x=example();e=make_edit(x,0,1,"о")
        with self.assertRaises(ValueError):
            replay(x,(dataclasses.replace(e,original_codepoint="U+9999"),),1)
    def test_edit_budget_and_attribute_limits(self):
        x=example();a=make_edit(x,0,1,"о");b=make_edit(x,0,2,"а")
        with self.assertRaises(ValueError):replay(x,(a,),0)
        with self.assertRaises(ValueError):replay(x,(a,b),2)
    def test_shared_word_across_attributes_one_edit(self):
        attrs=(AttributeSpan("color","black","upper","black",0,5),
               AttributeSpan("color","black","lower","black",0,5))
        x=TextExample("black",attrs)
        with self.assertRaises(ValueError):
            replay(x,(make_edit(x,0,2,"а"),make_edit(x,1,3,"с")),2)
    def test_whitelist_has_only_single_visible_non_ascii_characters(self):
        import unicodedata
        for source,rs in CONFUSABLES.items():
            self.assertTrue(source.isascii())
            for r in rs:
                self.assertEqual(len(r),1)
                self.assertFalse(r.isascii())
                self.assertTrue(unicodedata.category(r).startswith("L"))
                self.assertFalse(r.isspace())

class SelectionTests(unittest.TestCase):
    def test_empty_and_all_zero_K(self):
        self.assertEqual(natural_k([]),0)
        self.assertEqual(natural_k([0,0]),0)
    def test_uniform_dynamic_K_no_cap4(self):
        for k in (1,3,7,9):
            self.assertEqual(natural_k([1.]*k),k)
    def test_gamma3_entropy_definition(self):
        import math
        s=[4.,2.,1.];p=[v**3/sum(x**3 for x in s) for v in s]
        self.assertEqual(natural_k(s),math.ceil(math.exp(-sum(x*math.log(x) for x in p))))
    def test_bad_score_rejected(self):
        for v in (-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError):natural_k([v])
    def test_mask_never_selects_illegal_maximum(self):
        logits=torch.tensor([100.,1.,2.],requires_grad=True)
        mask=torch.tensor([False,True,True])
        for seed in range(10):
            i,_=choose(logits,mask,"sample",torch.Generator().manual_seed(seed))
            self.assertIn(i,(1,2))
        self.assertEqual(choose(logits,mask,"greedy")[0],2)
    def test_masked_logits_receive_zero_gradient(self):
        logits=torch.tensor([100.,1.,2.],requires_grad=True)
        _,lp=choose(logits,torch.tensor([False,True,True]),"greedy")
        lp.backward()
        self.assertEqual(float(logits.grad[0]),0.)
        self.assertNotEqual(float(logits.grad[1]),0.)
    def test_single_forced_action_zero_log_prob(self):
        logits=torch.tensor([1.,2.],requires_grad=True)
        i,lp=choose(logits,torch.tensor([False,True]),"sample")
        self.assertEqual(i,1);self.assertEqual(float(lp),0.)
    def test_empty_mask_or_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):choose(torch.ones(2),torch.zeros(2,dtype=torch.bool),"sample")
        with self.assertRaises(ValueError):choose(torch.ones(2),torch.ones(2,dtype=torch.bool),"targeted")

class NetworkTests(unittest.TestCase):
    def test_default_two_layer_configuration(self):
        p=CharacterPolicy(TokenizerProbe(ByteTokenizer()))
        pooled,chars,attrs=p(example())
        self.assertEqual(len(p.encoder.layers),2)
        self.assertEqual((pooled.shape,chars.shape,attrs.shape),
                         (torch.Size([64]),torch.Size([4,64]),torch.Size([1,64])))
    def test_invalid_architecture_rejected(self):
        for kw in ({"heads":0},{"width":3},{"width":18,"heads":4},{"layers":False}):
            with self.assertRaises(ValueError):CharacterPolicy(TokenizerProbe(ByteTokenizer()),**kw)
    def test_byte_transformer_shapes(self):
        p=model();pool,chars,attrs=p(example("é coat",2,6))
        self.assertEqual(pool.shape,(16,))
        self.assertEqual(chars.shape,(6,16))
        self.assertEqual(attrs.shape,(1,16))
        self.assertTrue(torch.isfinite(chars).all())
    def test_caption_byte_limit_rejects_without_relocation(self):
        p=CharacterPolicy(TokenizerProbe(ByteTokenizer()),width=16,layers=1,heads=4,max_bytes=3)
        with self.assertRaises(ValueError):p(example())
    def test_no_attributes_K0_identity_and_backward(self):
        p=model();out=p.rollout(TextExample("coat",()),"sample")
        self.assertEqual(out.budget_k,0)
        self.assertEqual(out.adversarial,"coat")
        self.assertEqual(out.edits,())
        out.log_prob.backward()
    def test_noop_record_retains_full_canonical_source_mapping(self):
        x=example("bt")
        out=model().attack(x)
        self.assertEqual(out.record()["attributes"],tuple(dataclasses.asdict(a) for a in x.attributes))
        self.assertEqual(out.edits,())
    def test_no_legal_glyph_identity(self):
        p=model();out=p.rollout(example("bt"),"sample")
        self.assertEqual(out.budget_k,0)
        self.assertEqual(out.adversarial,"bt")
    def test_sampling_reproducible_with_explicit_rng(self):
        torch.manual_seed(2);p=model()
        a=p.rollout(example(),"sample",torch.Generator().manual_seed(77))
        b=p.rollout(example(),"sample",torch.Generator().manual_seed(77))
        self.assertEqual(a.record(),b.record())
        self.assertTrue(torch.equal(a.log_prob,b.log_prob))
    def test_greedy_deterministic_and_inference_has_no_graph(self):
        p=model();a=p.attack(example());b=p.attack(example())
        self.assertEqual(a.record(),b.record())
        self.assertFalse(a.log_prob.requires_grad)
    def test_every_decision_within_mask_and_budget(self):
        p=model()
        for seed in range(25):
            out=p.rollout(example(),"sample",torch.Generator().manual_seed(seed))
            self.assertLessEqual(len(out.edits),out.budget_k)
            self.assertLessEqual(len(out.selected_attribute_ids),out.budget_k)
            self.assertEqual(len(out.selected_attribute_ids),len(set(out.selected_attribute_ids)))
            self.assertTrue(all(d["index"] in d["legal_indices"] for d in out.decisions))
            self.assertEqual(replay(example(),out.edits,out.budget_k),out.adversarial)
    def test_all_three_heads_receive_gradient(self):
        torch.manual_seed(3);p=model()
        out=None
        for seed in range(64):
            candidate=p.rollout(example(),"sample",torch.Generator().manual_seed(seed))
            if any(d["head"]=="replacement" and len(d["legal_indices"])>1 for d in candidate.decisions):
                out=candidate;break
        self.assertIsNotNone(out)
        p.zero_grad();(-out.log_prob).backward()
        for name in ("attribute_head","position_head","replacement_head"):
            self.assertGreater(float(getattr(p,name).weight.grad.norm()),0.,name)
        self.assertGreater(float(p.byte_embedding.weight.grad.norm()),0.)
    def test_greedy_attributes_sorted_by_predicted_importance(self):
        x=TextExample("coat top cap",tuple(AttributeSpan("garment_type",raw,"upper",raw,a,b)
                          for raw,a,b in (("coat",0,4),("top",5,8),("cap",9,12))))
        p=model()
        with torch.no_grad():
            p.attribute_head.weight.zero_();p.attribute_head.weight[0,0]=1;p.attribute_head.bias.zero_()
            p.position_head.weight.zero_();p.position_head.weight[0,0]=1;p.position_head.bias.zero_()
        pool=torch.zeros(16);chars=torch.ones(len(x.caption),16);attrs=torch.zeros(3,16);attrs[:,0]=torch.tensor([3.,1.,2.])
        with patch.object(p,"forward",return_value=(pool,chars,attrs)):
            out=p.rollout(x,"greedy")
        expected=tuple(x.attributes[i].attribute_id for i in (0,2,1))[:out.budget_k]
        self.assertEqual(out.selected_attribute_ids,expected)
    def test_shared_spans_never_duplicate_word_edits(self):
        x=TextExample("orange",(AttributeSpan("color","orange","upper","orange",0,6),
                                AttributeSpan("color","orange","lower","orange",0,6)))
        p=model()
        for seed in range(20):
            out=p.rollout(x,"sample",torch.Generator().manual_seed(seed))
            self.assertLessEqual(len(out.edits),1)
    def test_public_api_rejects_image_PID_and_Sraw(self):
        p=model()
        for kw in ("image","pid","paired_index","S_raw","JVP","training_labels"):
            with self.subTest(kw=kw),self.assertRaises(TypeError):
                p.rollout(example(),**{kw:123})
        with self.assertRaises(TypeError):p({"caption":"coat","pid":123})
    def test_policy_does_not_import_feedback_or_G(self):
        from pathlib import Path
        import ast
        source=ast.parse((Path(__file__).parent/"policy.py").read_text())
        modules=[n.module or "" for n in ast.walk(source) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any("reward" in n or "integration" in n or "generator" in n for n in modules))
    def test_cpu_only(self):
        self.assertFalse(torch.cuda.is_initialized())

if __name__=="__main__":unittest.main()
