"""Phase 2.5仅CPU只读审计测试，不执行G/optimizer/Epoch或JVP计算。"""
import copy,unittest,weakref,gc
from unittest.mock import patch
from phase25_prepare import CallCounter
import torch
from torch.nn import functional as F
import cache_io as io
from test_phase22 import condition
from training_data import identity
from phase25_audit import condition_audit,distribution

def fixture(k=3):
    c=condition(k);entry=identity(c)
    src={k:c[k] for k in ("row_id","pid","caption","K")}
    src.update(image=c["image_rel_path"],image_sha256="image_sha")
    selected=[{"instance_id":i,"S_raw":s} for i,s in zip(c["selected_instance_ids"],c["S_raw"].tolist())]
    md={"role":c["role"],"payload":{"selected_attributes":selected,"ranked_attributes":selected,"natural_k":k},
        "selected_instance_ids":c["selected_instance_ids"],"score_normalized_weights":c["score_weights"].tolist(),
        "score_record":{"row_id":c["row_id"],"frozen":True}}
    meta={"source_selection":copy.deepcopy(md["payload"]),"source_score_record":copy.deepcopy(md["score_record"]),
        "source_fingerprint":io.digest([src,md]),"source_image_sha256":src["image_sha256"],
        "JVP_checks":[{"instance_id":v,"direction_norm":float(c["attribute_jvps"][i].double().norm()),
                      "zero_exact":True,"central_fd_relative_error":.0002} for i,v in enumerate(c["selected_instance_ids"])]}
    return entry,c,meta,src,md,{"z_t":c["z_t"].clone(),"v_local":c["v_local"].clone()}

class AuditTests(unittest.TestCase):
    def test_K0_empty_protocol(self):
        r=condition_audit(*fixture(0))
        self.assertEqual((r["K"],r["FD_errors"],r["equal_JVP_extra_instances"]),(0,[],0))
        self.assertEqual(r["aggregate_error"]["max_abs_error"],0)
    def test_K7_dimensions_order_no_cap(self):
        r=condition_audit(*fixture(7));self.assertEqual(len(r["FD_errors"]),7)
        self.assertTrue(r["z_t_exact"]);self.assertEqual(r["aggregate_error"]["max_abs_error"],0)
    def test_S_raw_must_equal_original(self):
        args=fixture();args[1]["S_raw"][0]+=1e-15
        with self.assertRaisesRegex(ValueError,"S_raw"):condition_audit(*args)
    def test_weights_must_equal_original_even_within_schema_tolerance(self):
        args=fixture();args[1]["score_weights"][0]+=1e-16
        with self.assertRaisesRegex(ValueError,"weights"):condition_audit(*args)
    def test_ID_permutation_rejected(self):
        args=fixture();args[1]["selected_instance_ids"]=args[1]["selected_instance_ids"][::-1]
        with self.assertRaisesRegex(ValueError,"ID/order"):condition_audit(*args)
    def test_caption_or_source_image_SHA_mismatch_rejected(self):
        for field,value in (("caption","wrong"),("image_rel_path","wrong.png"),("pid",999)):
            args=fixture();args[1][field]=value;args[0][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"source identity"):condition_audit(*args)
        args=fixture();args[2]["source_image_sha256"]="wrong"
        with self.assertRaisesRegex(ValueError,"image SHA"):condition_audit(*args)
    def test_z_t_unit_but_different_rejected(self):
        args=fixture();args[1]["z_t"]=F.normalize(args[1]["z_t"]+.01,dim=0)
        with self.assertRaisesRegex(ValueError,"z_t"):condition_audit(*args)
    def test_fingerprint_change_rejected(self):
        args=fixture();args[2]["source_fingerprint"]="wrong"
        with self.assertRaisesRegex(ValueError,"fingerprint"):condition_audit(*args)
    def test_score_record_change_rejected(self):
        args=fixture();args[2]["source_score_record"]["frozen"]=False
        with self.assertRaisesRegex(ValueError,"score/selection"):condition_audit(*args)
    def test_zero_JVP_never_used_as_replacement(self):
        args=fixture();args[1]["attribute_jvps"][0].zero_()
        with self.assertRaisesRegex(ValueError,"zero attribute JVP"):condition_audit(*args)
    def test_FD_nan_negative_and_over_threshold_rejected(self):
        for value in (float("nan"),float("inf"),-.001,.02001):
            args=fixture();args[2]["JVP_checks"][0]["central_fd_relative_error"]=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"FD/zero"):condition_audit(*args)
    def test_zero_exact_false_rejected(self):
        args=fixture();args[2]["JVP_checks"][0]["zero_exact"]=False
        with self.assertRaisesRegex(ValueError,"FD/zero"):condition_audit(*args)
    def test_missing_JVP_check_and_wrong_order_rejected(self):
        args=fixture();args[2]["JVP_checks"].pop()
        with self.assertRaisesRegex(ValueError,"coverage/order"):condition_audit(*args)
        args=fixture();args[2]["JVP_checks"].reverse()
        with self.assertRaisesRegex(ValueError,"coverage/order"):condition_audit(*args)
    def test_JVP_norm_metadata_mismatch_rejected(self):
        args=fixture();args[2]["JVP_checks"][0]["direction_norm"]+=.001
        with self.assertRaisesRegex(ValueError,"norm audit"):condition_audit(*args)
    def test_different_instance_equal_JVPs_reported_without_merging(self):
        args=fixture();c=args[1];c["attribute_jvps"][1]=c["attribute_jvps"][0]
        c["v_local"]=(c["score_weights"].float()[:,None]*c["attribute_jvps"]).sum(0)
        args[5]["v_local"]=c["v_local"].clone()
        args[2]["JVP_checks"][1]["direction_norm"]=args[2]["JVP_checks"][0]["direction_norm"]
        r=condition_audit(*args);self.assertEqual(r["equal_JVP_extra_instances"],1);self.assertEqual(r["K"],3)
    def test_distribution_train_validation_and_empty_bins(self):
        rows=[{"role":"train","K":k} for k in (0,1,3,7)]+[{"role":"validation","K":4}]
        d=distribution(rows)
        self.assertEqual(d["all"]["JVP_total"],15);self.assertEqual(d["train"]["JVP_total"],11)
        self.assertEqual(d["all"]["K_counts"],{"0":1,"1":1,"2":0,"3":1,"4":1,"5":0,"6":0,"7":1})
        self.assertEqual(d["all"]["mean_K"],3)
    def test_out_of_range_K_rejected(self):
        with self.assertRaisesRegex(ValueError,"0..7"):distribution([{"role":"train","K":8}])
    def test_counter_does_not_retain_tensor_arguments(self):
        counter=CallCounter(lambda x:x.detach().clone())
        x=torch.ones(10,requires_grad=True);ref=weakref.ref(x)
        result=counter(x);del x;gc.collect()
        self.assertIsNone(ref());self.assertEqual(counter.call_count,1)
        self.assertFalse(result.requires_grad)
    def test_counter_keeps_original_autograd_JVP_and_count(self):
        counter=CallCounter(torch.autograd.grad)
        with patch.object(torch.autograd,"grad",new=counter):
            value,v=torch.autograd.functional.jvp(lambda x:x**3,torch.tensor(2.),torch.tensor(1.),strict=True)
        self.assertEqual((float(value),float(v),counter.call_count),(8.,12.,2))
    def test_counter_records_raised_call_without_retaining_arguments(self):
        def fail(x):raise RuntimeError("observed failure")
        counter=CallCounter(fail);x=torch.ones(3);ref=weakref.ref(x)
        with self.assertRaises(RuntimeError):counter(x)
        del x;gc.collect();self.assertIsNone(ref());self.assertEqual(counter.call_count,1)
    def test_no_CUDA_initialized(self):self.assertFalse(torch.cuda.is_initialized())

if __name__=="__main__":unittest.main()
