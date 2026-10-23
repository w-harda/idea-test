"""CPU复核测试：真实控制失败不得改写为PASS，不从SHA虚构距离。"""
import importlib.util
from pathlib import Path
import pytest
import torch
spec=importlib.util.spec_from_file_location("review",Path(__file__).parents[2]/"operations/gh_smoke_readonly_review.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def g(before="p",after="q",grad="g"):
 return {"G_before":before,"G_after":after,"gradient_observations":[{"gradient_sha256":grad,"record":{"G_gradient_l2":1.}}]}
def h(delta,adv=0.,gradient=0.,changed=True):
 return {"advantage":adv,"gradient_norm":gradient,"H_state_sha256_before":"h0",
 "H_state_sha256_after":"h1" if changed else "h0","optimizer_steps_this_step":1,"reward_delta_same_text":delta}
def test_exact_noop():
 assert all(m.noop_checks(g(),g()).values())
def test_equal_loss_does_not_hide_changed_update():
 c=m.noop_checks(g(),g(after="other",grad="other"))
 assert c["same_G_before"] and not c["same_G_after"] and not c["same_gradient_observations"]
def test_raw_feedback_not_strict_confirmation():
 a=[g(),g("q","r","x")];b=[g(after="Q",grad="G"),g("Q","R","X")]
 r=m.feedback(a,b,[h([1.,2.])])
 assert r["derived_original_expression"]["bidirectional_update_evidence_verified"] is True
 assert r["bidirectional_update_evidence_verified"] is False
 assert not r["first_nonNOOP_same_G_before"]
def test_parent_difference_detected_even_with_gradient_difference():
 a=[g(),g("q","r","x")];b=[g(),g("Q","R","X")]
 assert not m.feedback(a,b,[h([1.,2.])])["bidirectional_update_evidence_verified"]
def test_reward_and_advantage_are_distinct():
 s=m.h_summary([h([1.,1.]),h([0.,2.],adv=1.,gradient=2.),{"fixed_NOOP":True}])
 assert s["same_text_Reward_changed"]==2 and s["same_text_Advantage_changed"]==1
 assert s["nonzero_gradient"]==1 and s["zero_gradient_but_state_changed"]==1 and s["Adam_updates"]==2
def test_tensor_distance_from_real_values():
 r=m.tensor_difference({"w":torch.tensor([1.,2.])},{"w":torch.tensor([4.,6.])})
 assert r=={"max_abs":4.,"L2":5.,"changed_elements":2,"elements":2}
def test_tensor_shape_mismatch_rejected():
 with pytest.raises(AssertionError):m.tensor_difference({"w":torch.ones(1)},{"w":torch.ones(2)})
def test_sha_tamper_rejected(tmp_path):
 from tbps.artifacts import ArtifactStore,ArtifactError
 root=tmp_path/"tbps-framework-fixture"
 s=ArtifactStore(root/"unit",new_base=root)
 s.publish("x",{"payload":b"original"},{"fixture":True})
 (root/"unit/x/payload").write_bytes(b"tampered")
 with pytest.raises(ArtifactError):s.verify("x")
