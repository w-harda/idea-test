from dataclasses import FrozenInstanceError, fields
import unittest
from tbps.contracts import *
H="1"*64

class ContractsTests(unittest.TestCase):
    def model(self,dim=1024,size=(256,256),length=128):
        return ModelSpec("fixture","v1",size,TokenizerSpec("fixture",H,length,"token","reject"),dim,
                         "dot","none",frozenset({"score","tokenize"}))
    def test_arbitrary_model_dimensions(self):
        for dim,size,n in ((37,(320,112),33),(1024,(256,256),128),(768,(224,224),None)):
            with self.subTest(dim=dim):
                spec=self.model(dim,size,n);self.assertEqual(spec.representation_dim,dim);self.assertEqual(spec.image_size_hw,size)
    def test_native_fusion_without_global_embedding(self):
        m=ModelSpec("fusion","v1",None,None,None,"fusion","model_native",frozenset({"fusion","score","local_features"}))
        self.assertIsNone(m.representation_dim)
    def test_fusion_missing_capability(self):
        with self.assertRaises(ValueError):ModelSpec("x","v1",None,None,None,"fusion","none",frozenset({"score"}))
    def test_tokenizer_required(self):
        with self.assertRaises(ValueError):ModelSpec("x","v1",None,None,17,"dot","none",frozenset({"tokenize"}))
    def test_immutable_spec(self):
        m=self.model()
        with self.assertRaises(FrozenInstanceError):m.id="changed"
    def test_mutable_nested_rejected(self):
        with self.assertRaises(TypeError):ModelSpec("x","v1",[1,2],None,17,"dot","none",frozenset())
        with self.assertRaises(ValueError):ModelSpec("x","v1",(1,2),None,17,"dot","none",set())
    def manifest(self):
        pid=PersonID("fixture","1");ref=FileRef("annotation.json",H)
        g=(GalleryItem("g2","2.jpg",H,pid,"test"),GalleryItem("g1","1.jpg",H,pid,"test"))
        q=(QueryItem("q0","one",0,"g1",pid,"test"),QueryItem("q1","two",1,"g1",pid,"test"))
        return DatasetManifest("fixture","test",ref,g,q)
    def test_multi_caption_multi_positive_order(self):
        m=self.manifest();e=m.evaluation_identity()
        self.assertEqual(e.gallery_ids,("g2","g1"));self.assertEqual(e.positives(0),(0,1));self.assertEqual(len(m.queries),2)
    def test_pid_namespaces(self):
        self.assertNotEqual(PersonID("a","1"),PersonID("b","1"))
        with self.assertRaises(ValueError):EvaluationIdentity(("q",),("g",),(PersonID("a","1"),),(PersonID("b","1"),))
    def test_wrong_caption_pair_rejected(self):
        m=self.manifest()
        with self.assertRaises(ValueError):DatasetManifest("fixture","test",m.annotation,m.gallery,(QueryItem("q","bad",0,"missing",m.queries[0].pid,"test"),))
    def test_split_mismatch_rejected(self):
        m=self.manifest()
        with self.assertRaises(ValueError):DatasetManifest("fixture","train",m.annotation,m.gallery,m.queries)
    def test_duplicate_order_rejected(self):
        m=self.manifest()
        with self.assertRaises(ValueError):DatasetManifest("fixture","test",m.annotation,m.gallery,m.queries*2)
    def test_unsafe_image_paths(self):
        for p in ("../x","/x","C:/x","x\\y"):
            with self.subTest(path=p),self.assertRaises(ValueError):GalleryItem("a",p,H,PersonID("x","1"),"test")
    def test_text_span_exact_unicode_and_repeated_word(self):
        t="red red 衣";a=AttributeSpan("color","red","red",4,7)
        x=PublicTextInput(t,(a,));self.assertEqual(x.caption[a.start:a.end],"red")
        with self.assertRaises(ValueError):PublicTextInput(t,(AttributeSpan("color","red","red",3,6),))
    def test_public_input_no_identity_or_reward(self):
        self.assertEqual({f.name for f in fields(PublicTextInput)},{"caption","attributes"})
        for k in ("pid","paired_index","gallery","reward","S_raw"):
            with self.subTest(key=k),self.assertRaises(TypeError):PublicTextInput("red",**{k:1})
    def test_training_context_positive_and_negative(self):
        p=PersonID("f","1");n=PersonID("f","2")
        c=TrainRewardContext(p,(p,n),0,"train","attack_train",("attack_train","attack_train"));self.assertEqual(c.paired_index,0)
    def test_training_context_refuses_validation_test(self):
        p=PersonID("f","1");n=PersonID("f","2")
        for split,role,roles in (("test","attack_train",("attack_train",)*2),("train","attack_validation",("attack_train",)*2),("train","attack_train",("attack_train","attack_validation"))):
            with self.subTest(split=split,role=role),self.assertRaises(ValueError):TrainRewardContext(p,(p,n),0,split,role,roles)
    def test_training_context_requires_negative_and_paired_positive(self):
        p=PersonID("f","1");n=PersonID("f","2")
        for gallery,index in (((p,p),0),((p,n),1),((p,PersonID("other","2")),0)):
            with self.subTest(gallery=gallery),self.assertRaises(ValueError):TrainRewardContext(p,gallery,index,"train","attack_train",("attack_train",)*2)
    def test_explicit_budget_spaces(self):
        self.assertEqual(ImageBudget("l2",1,"native_pixel_domain").pixel_space,"native_pixel_domain")
        self.assertEqual(TextBudget("byte",4,None,"fixture").max_edits,4)
        self.assertEqual(TextBudget("unicode_character",None,"learned_dynamic","fixture",1,1).per_attribute,1)
    def test_budget_invalid(self):
        for x in (float("nan"),float("inf"),-1,True):
            with self.subTest(x=x),self.assertRaises(ValueError):ImageBudget("linf",x,"rgb_0_1")
        for fixed,dynamic in ((None,None),(3,"dynamic")):
            with self.subTest(fixed=fixed),self.assertRaises(ValueError):TextBudget("word",fixed,dynamic,"x")
    def test_representation_variable_shape(self):
        r=RepresentationArtifact("a",H,"model-native","local_tokens",(2,35,768),"float32","none",H,None,FileRef("x",H))
        self.assertEqual(r.shape,(2,35,768))
    def test_scores_preserve_native_direction_order(self):
        s=ScoreArtifact("s",H,H,("q2","q1"),("g",),"lower_better",FileRef("scores",H));self.assertEqual(s.query_order,("q2","q1"))
    def test_invalid_sha_rejected(self):
        for sha in ("0"*63,"A"*64,"z"*64):
            with self.subTest(sha=sha),self.assertRaises(ValueError):FileRef("x",sha)
    def test_attack_types_not_all_generator(self):
        for kind,training in (("uap","offline"),("iterative",None),("policy","scst"),("composite","multi_stage")):
            with self.subTest(kind=kind):
                a=AttackSpec("future","v1",frozenset({"text"}),kind,training,kind=="iterative",frozenset(),frozenset({"captions"}),None)
                self.assertEqual(a.artifact_kind,kind)

if __name__=="__main__":unittest.main()
