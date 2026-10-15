"""仅离线prepare：复用冻结原JVP/uncapped plan，无G-v1模块或评分。"""
import importlib,sys
from unittest.mock import patch
from pathlib import Path
import torch
from torch.nn import functional as F
from condition_schema import build,validate,error
import cache_io as io

EXPERIMENT=Path("/home/lzf/ldx/projects/idea-TBPS-test1")
BASIS=EXPERIMENT/"prototypes/attribute_unit_v55_interface_20261003/outputs/full/semantic_bases.json"

class TextVictim:
    def __init__(self,encoder,tokenizer): self.model=encoder.model; self.tokenizer=tokenizer
    def encode_texts(self,texts):
        t=self.tokenizer; rows=[]
        for text in texts:
            ids=[t.encoder["<|startoftext|>"]]+t.encode(text)+[t.encoder["<|endoftext|>"]]
            if len(ids)>77: ids=ids[:77];ids[-1]=t.encoder["<|endoftext|>"]
            rows.append(ids+[0]*(77-len(ids)))
        tokens=torch.tensor(rows,device=next(self.model.parameters()).device,dtype=torch.long)
        return F.normalize(self.model.encode_text(tokens).float(),dim=-1)

class Frontend:
    def __init__(self,encoder):
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(EXPERIMENT/"prototypes/attribute_tta_selection_value_20261005"))
        importlib.import_module("deps") # 冻结原JVP依赖路径，不含G-v1。
        from attribute_tta import TextTransport
        from paired_prefix import prepare_uncapped
        from utils.simple_tokenizer import SimpleTokenizer
        self.transport=TextTransport; self.plan=prepare_uncapped
        self.victim=TextVictim(encoder,SimpleTokenizer());self.bases=io.load(BASIS)
    def __call__(self,entry,md):
        plan=self.plan(md["payload"],md["score_record"],self.bases,"cuda")
        transport=self.transport(self.victim,entry["caption"])
        old=torch.load(entry["tensor_path"],map_location="cpu",weights_only=True)
        if not torch.equal(transport.clean.cpu().reshape(-1),old["z_t"]): raise ValueError("caption/JVP base changed")
        vectors=[];checks=[];grad_calls=0
        for a in plan:
            with patch.object(torch.Tensor,"backward",side_effect=AssertionError("scoring/training backward forbidden during prepare")), \
                 patch.object(torch.autograd,"grad",wraps=torch.autograd.grad) as calls:
                v,check=transport.read(a)
            grad_calls+=calls.call_count
            if not check["zero_exact"] or check["central_fd_relative_error"]>.02: raise ValueError("original JVP FD/zero gate")
            vectors.append(v.detach().cpu());checks.append(check)
        identity={"row_id":entry["row_id"],"image_rel_path":entry["image"],"caption_index":int(entry["row_id"].split(":")[-1]),
            "split":"train","role":md["role"],"pid":entry["pid"],"caption":entry["caption"]}
        value=build(identity,md["selected_instance_ids"],[a.metadata["S_raw"] for a in plan],
                    md["score_normalized_weights"],torch.cat(vectors) if vectors else torch.empty(0,512),old["z_t"])
        if not torch.allclose(value["v_local"],old["v_local"],atol=1e-9,rtol=1e-5): raise ValueError("original aggregate changed")
        return value,{"source_selection":md["payload"],"source_score_record":md["score_record"],
            "source_image_sha256":entry["image_sha256"],"JVP_checks":checks,
            "new_scoring_backwards":0,"new_JVPs":len(plan),"new_JVP_autograd_grad_calls":grad_calls,"aggregate_reconstruction":validate(value),
            "legacy_aggregate_error":error(value["v_local"],old["v_local"])}
