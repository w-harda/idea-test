"""源模型Tokenizer适配：保留原Unicode/BPE合法性，阻止新增原生截断。"""
from .common import *
class NativeGuard:
    def __init__(self,base,tokenizer,normalize,limit):
        self.base,self.tokenizer_native,self.normalize,self.limit=base,tokenizer,normalize,limit
    def __getattr__(self,key):return getattr(self.base,key)
    def legal_change(self,current,modified):
        if not self.base.legal_change(current,modified):return False
        def size(s):return len(self.tokenizer_native(self.normalize(s),add_special_tokens=True,truncation=False)['input_ids'])
        before,after=size(current),size(modified)
        return after<=self.limit or after<=before
def bind(c,bundle,name):
    from importlib import import_module
    mod=import_module('tbps.models.'+name)
    source=(mod.RasaSource if name=='rasa' else mod.APTMSource)(c['targets'][name]['path'])
    with mod.native_modules(source) as (_,tok,utils):
        tokenizer=mod.official_tokenizer(tok,source)
        if name=='rasa':normalize=lambda s:utils.pre_caption(s,50)
        else:normalize=lambda s:utils.pre_caption(s,56,icfg_rstp=False,is_eda=False)
    guard=NativeGuard(bundle.probe,tokenizer,normalize,50 if name=='rasa' else 56)
    bundle.probe=guard;bundle.h.probe=guard
    return bundle
