"""SEP的分类→检索I/O桥；DPP、概率margin与候选算法定义仍直接读取作者代码。"""
from collections import OrderedDict
from types import SimpleNamespace
import math,re
from .official import sep_ensemble_class,definitions
from .dataset import require
def prepare_caption(c,text):
 from pathlib import Path
 src=c["official"]["sep"];p=Path(src["root"])/"dataloader.py"
 fn=definitions(p,["clean_str"],src["files"][str(p)],{"re":re})["clean_str"]
 return " ".join(fn(text.strip()).lower().split()[:256])
def pid_logits(scores,gallery_pids):
 """z(T,c)=max_{i:PID(i)=c} native_score(T,I_i)，然后由原Ensemble执行softmax和margin。"""
 import torch
 order=tuple(dict.fromkeys(gallery_pids))
 require(scores.ndim==2 and scores.shape[1]==len(gallery_pids),"complete native Gallery scores")
 values=torch.stack([scores[:,[i for i,p in enumerate(gallery_pids) if p==pid]].max(dim=1).values for pid in order],dim=1)
 return values,order
class Texts(list):
 """只运输原字符串；实际分词在原生Victim Adapter内部执行，不用另一Tokenizer替代。"""
 def to(self,device):return self
class TransportTokenizer:
 def __call__(self,texts,**kwargs):return {"texts":Texts(texts)}
def make_ensemble(c,models,device):
 """models必须为IRRA、RaSa的原生完整检索→PID分类包装；不加载官方分类任务权重。"""
 original=sep_ensemble_class(c)
 class NativeEnsemble(original):
  def _parse_models(self):
   require(tuple(self.models_name)==("irra","rasa") and set(models)=={"irra","rasa"},"fixed two-model ensemble")
   self.models=OrderedDict((k,models[k]) for k in self.models_name)
   self.tokenizers=OrderedDict((k,TransportTokenizer()) for k in self.models_name)
 return NativeEnsemble(OrderedDict((k,k) for k in ("irra","rasa")),device,c["sep_protocol"]["k_margin"])
def legal_semantic_flags(original,adversarial):
 """只做派生风险标记，不修改同义词候选、不据此挑选攻击文本。"""
 a=original.split();b=adversarial.split()
 require(len(a)==len(b),"SEP word-position mapping")
 colors={"black","white","blue","red","green","yellow","brown","gray","grey","pink","purple","orange","beige"}
 male={"man","men","male","boy","boys","gentleman","gentlemen","guy","guys"}
 female={"woman","women","female","girl","girls","lady","ladies"}
 edits=[];flags=[]
 for i,(x,y) in enumerate(zip(a,b)):
  if x==y:continue
  edits.append({"position":i,"original":x,"replacement":y})
  if x in colors and y in colors and {x,y}!={"gray","grey"}:flags.append({"position":i,"type":"COLOR_ATTRIBUTE_CHANGED"})
  if (x in male and y in female) or (x in female and y in male):flags.append({"position":i,"type":"GENDER_ATTRIBUTE_CHANGED"})
 return {"edits":edits,"word_change_count":len(edits),"ratio":len(edits)/len(a) if a else 0,
   "identity_semantic_flags":flags,"full_semantic_equivalence":"NOT_VERIFIED"}
