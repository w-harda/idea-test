"""H薄插件：原TextAttacker及validate_action完整复用，标签不进入后端。"""
from ..config import ATTACKS
from ..contracts import AttackBudget, TextBudget
from .base import PluginInfo, TextRequest, need, frozen_state, output

BUDGET = AttackBudget(text=TextBudget("unicode_character",None,"H-natural-k-gamma3",
                                      "hv1-visual-unicode-tokenizer-mask-v1",1,1))
INFO = PluginInfo(ATTACKS["hv1"],frozenset({"train_pairs","train_pids","source_rewards"}),
                  frozenset({"captions","caption_attributes"}))

class HV1Plugin:
    info = INFO
    def __init__(self, *, backend, provenance, budget=BUDGET):
        need(budget==BUDGET, "frozen H Dynamic-K/Unicode constraints required")
        need(provenance.attack_id=="hv1", "wrong selected H provenance")
        need(backend.execution_kind==provenance.execution_kind, "fixture/legacy provenance mismatch")
        self.backend,self.provenance,self.budget=backend,provenance,budget
    def generate(self, request):
        import torch
        need(type(request) is TextRequest, "H requires Caption-only public request")
        before=frozen_state(self.backend.model)
        # 身份binding、Dataset、PID、Gallery或目标Tokenizer不传给旧策略。
        calls=[]
        def observe(module,args):
            need(len(args)==1 and set(vars(args[0]))=={"caption","attributes"} and
                 args[0].caption==request.public.caption, "H forward must receive Caption-only TextExample")
            calls.append(1)
            need(len(calls)<=1, "at most one H forward per Caption")
        handle=self.backend.model.register_forward_pre_hook(observe)
        try:
            with torch.inference_mode():
                record=self.backend.attack(request.public.caption)
                self.backend.validate(request.public.caption,record)
        finally: handle.remove()
        need(record["original"]==request.public.caption and type(record["adversarial"]) is str,
             "wrong original/adversarial Caption")
        k=record["budget_k"]
        need(type(k) is int and k>=0 and len(record["edits"])<=k, "Dynamic-K edit budget")
        need(record["H_forward"] in (0,1) and len(calls)==record["H_forward"], "actual H forward count mismatch")
        need(frozen_state(self.backend.model)==before, "H state changed")
        trace={"stage":"online_generation","action":record,"state_sha256_before":before,
               "state_sha256_after":before,"optimizer_steps":0,"surrogate_forwards":0,
               "target_tokenizer_used":False,"target_feedback_used":False}
        return output("hv1",request,self.budget,record["adversarial"],None,trace,self.provenance.as_dict())
