"""G/H属性与K只读分析的合成CPU测试；不运行策略、模型或Tokenizer。"""
import ast,copy,hashlib,json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tbps.ablations import gh_attribute_audit as m
class TestAttributeAudit(unittest.TestCase):
    def setUp(self):
        self.caption="a cat"
        self.attrs=[dict(semantic_axis="gender",region="person",canonical_value="male",char_start=0,char_end=1,raw_text="a"),
            dict(semantic_axis="garment_type",region="upper",canonical_value="cat",char_start=2,char_end=5,raw_text="cat")]
        self.q=dict(caption=self.caption,attributes=self.attrs,legal_candidate_counts=[1,1],eligible=True)
    def action(self):
        a=self.attrs[0]
        e=dict(attribute_id=m.attr_id(a),source_span=[0,1],source_raw="a",position=0,original_char="a",replacement_char="а",
            original_codepoint="U+0061",replacement_codepoint="U+0430")
        return dict(attributes=self.attrs,predicted_importance=[1.,1.],budget_k=2,selected_attribute_ids=[m.attr_id(a)],
            decisions=[dict(head="attribute",index=0,legal_indices=[0,1,2]),dict(head="position",index=0,legal_indices=[0,1]),
                dict(head="replacement",index=0,legal_indices=[0]),dict(head="attribute",index=2,legal_indices=[1,2])],
            edits=[e],adversarial="а cat")
    def test_natural_zero(self):self.assertEqual(m.natural([0,0])["k"],0)
    def test_natural_no_cap_four(self):self.assertEqual(m.natural([1]*9)["k"],9)
    def test_natural_concentrated(self):self.assertEqual(m.natural([1000,1,1])["k"],2)
    def test_natural_negative_rejected(self):
        with self.assertRaises(ValueError):m.natural([-1])
    def test_repeated_word_exact_span(self):
        attrs=[dict(self.attrs[0],char_start=i,char_end=i+1) for i in (0,2)]
        rows=m.strict_alignment(attrs,attrs[::-1],"a a")
        self.assertEqual([x[1] for x in rows],["EXACT","EXACT"])
    def test_different_semantics_not_fuzzy(self):
        rows=m.strict_alignment(self.attrs,[dict(self.attrs[0],region="head")],self.caption)
        self.assertFalse(any(x[1]=="EXACT" for x in rows))
    def test_ambiguous_duplicate(self):
        rows=m.strict_alignment([self.attrs[0]]*2,[self.attrs[0]],self.caption)
        self.assertEqual(rows[0][1],"AMBIGUOUS")
    def test_invalid_span(self):
        with self.assertRaises(ValueError):m.key(dict(self.attrs[0],raw_text="x"),self.caption)
    def test_stop_loses_remaining_slot(self):
        x=m.breakdown(self.action(),self.q)
        self.assertEqual((x["edit_count"],x["unused_K"],x["STOP_unused_slots"],x["position_NOOP"]),(1,1,1,0))
    def test_position_noop_consumes_slot(self):
        a=self.action();a["selected_attribute_ids"].append(m.attr_id(self.attrs[1]))
        a["decisions"][-1]=dict(head="attribute",index=1,legal_indices=[1,2])
        a["decisions"].append(dict(head="position",index=3,legal_indices=[0,1,3]))
        x=m.breakdown(a,self.q)
        self.assertEqual((x["edit_count"],x["unused_K"],x["position_NOOP"],x["STOP_unused_slots"]),(1,1,1,0))
    def test_unicode_chars_same_bytes_different(self):
        a=self.action();x=m.breakdown(a,self.q)
        self.assertEqual((x["char_length_delta"],x["utf8_length_delta"]),(0,1))
    def test_illegal_unicode_rejected(self):
        a=self.action();a["edits"][0]["replacement_char"]="z";a["edits"][0]["replacement_codepoint"]="U+007A";a["adversarial"]="z cat"
        with patch.object(m,"WHITELIST",{"a":("а",)}):
            with self.assertRaises(ValueError):m.breakdown(a,self.q)
    def test_selected_id_conflict_rejected(self):
        a=self.action();a["selected_attribute_ids"]=["wrong"]
        with self.assertRaises(ValueError):m.breakdown(a,self.q)
    def test_tie_rank(self):self.assertEqual(m.ranks([3,1,1]).tolist(),[3,1.5,1.5])
    def test_constant_rank_not_fake_zero(self):self.assertIsNone(m.corr([1,1,1],[1,2,3])["spearman"])
    def test_empty_jaccard_undefined(self):self.assertIsNone(m.overlap(set(),set())["jaccard"])
    def test_source_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.json";p.write_text("{}");h=m.sha(p);p.write_text("[]")
            with self.assertRaises(ValueError):m.Inputs().add(p,h)
    def test_no_model_execution_import(self):
        tree=ast.parse(Path(m.__file__).read_text())
        names=[n.module or "" for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        names += [x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names]
        self.assertFalse(any(x.startswith(("torch","transformers","policy","rewards")) for x in names))


class TestOfficialPureFunctions(unittest.TestCase):
    def getfn(self,path,name,namespace):
        tree=ast.parse(Path(path).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),"exec"),namespace);return namespace[name]
    def test_official_H_natural_formula(self):
        f=self.getfn(m.H/"policy.py","natural_k",{"math":math})
        for x in ([],[0,0],[1]*9,[1000,1,1],[.1,3,2,1]):
            self.assertEqual(f(x),m.natural(x)["k"])
    def test_official_G_uncapped_formula(self):
        ns={"math":math,"copy":copy,"GAMMA":3.,"BETA":1.,"CAP":4,"span_gate":lambda c,a:True}
        f=self.getfn(m.LEGACY/"prototypes/attribute_tta_step74_dynamic_k_20261004/selection_audit.py","selection",ns)
        attrs=[dict(status="scored",S_raw=1.,instance=dict(instance_id=str(i),semantic_representation=dict(basis_rank=1),unresolved=False)) for i in range(9)]
        x=f(dict(attributes=attrs,caption="fixture",row_id="fixture"))
        self.assertEqual((x["natural_k"],len(x["selected_attributes"]),x["capped_k"]),(9,9,4))
    def test_same_softmax_different_K_possible(self):
        import numpy as np
        logits=np.array([-2.,0.,2.,0.])
        sm=lambda x:np.exp(x-x.max())/np.exp(x-x.max()).sum()
        self.assertTrue(np.allclose(sm(logits),sm(logits+10),atol=1e-14,rtol=0))
        score=lambda x:np.logaddexp(0,x[:3]).tolist()
        self.assertEqual((m.natural(score(logits))["k"],m.natural(score(logits+10))["k"]),(2,3))
    def test_budget_telescope(self):
        for kg,gc,hc,kh in [(3,3,2,2),(3,3,2,3),(4,3,2,2),(0,0,0,0)]:
            self.assertEqual((kg-gc)+(gc-hc)+(hc-kh),kg-kh)
    def test_two_candidate_ranking_and_ties(self):
        self.assertEqual(m.ranks([1,1]).tolist(),[1.5,1.5])
        import numpy as np
        self.assertAlmostEqual(float(np.corrcoef(m.ranks([1,2]),m.ranks([2,1]))[0,1]),-1.)
    def test_G_ID_and_H_ID_not_alignment_key(self):
        a=dict(semantic_axis="color",region="upper",canonical_value="blue",char_start=0,char_end=4,raw_text="blue")
        self.assertEqual(m.strict_alignment([dict(a,instance_id="g")],[dict(a,attribute_id="h")],"blue")[0][1],"EXACT")



class TestCategorySampleCoverage(unittest.TestCase):
    def test_empty_attribute_sample_kept(self):
        from tbps.ablations.gh_attribute_category_overlap import group_attributes
        x=dict(scope="Validation",H_epoch="20",row_id="empty")
        y=dict(x,row_id="has_attributes")
        grouped=group_attributes([x,y],[y])
        self.assertEqual(len(grouped),2)
        self.assertEqual(grouped[("Validation",20,"empty")],[])
        self.assertEqual(len(grouped[("Validation",20,"has_attributes")]),1)
    def test_unknown_category_member_rejected(self):
        from tbps.ablations.gh_attribute_category_overlap import group_attributes
        with self.assertRaises(ValueError):
            group_attributes([], [dict(scope="Validation",H_epoch="20",row_id="unknown")])
if __name__=="__main__":unittest.main()
