"""原Tokenizer/Span函数的精确字节接口；不改变token ID、文本或reader数学。"""
from copy import deepcopy
from .contracts import require, text_sha

def token_audit(probe, caption):
    from dataclasses import asdict
    report=probe.inspect(caption)
    require(not report.truncated, "new context truncation forbidden")
    ids=tuple(report.model_tokens)
    require(len(ids)==77 and ids.count(probe.eot)==1
            and ids[report.total_length-1]==probe.eot
            and ids[0]==probe.sot, "native EOT identity/position")
    return {**asdict(report),"eot_index":report.total_length-1,
            "caption_sha256":text_sha(caption)}

class ByteSpanBridge:
    """保留原mapper输出供审计；仅恢复byte-piece拆开单个Unicode字符的已证明坐标。
    其他cleaning/截断/边界/重放错误保持拒绝；旧mapper文件和函数不修改。
    """
    def __init__(self, module, tokenizer):
        self.module,self.tokenizer=module,tokenizer
        self.last=None

    def align(self,caption,tokenizer):
        require(tokenizer is self.tokenizer, "tokenizer identity")
        original=self.module.align_irra_tokens(caption,tokenizer)
        require(original["tokenizer_encode_matches_replay"] and not original["truncated"],
                "native BPE replay/truncation")
        out=deepcopy(original)
        require(not set(original["reasons"])-{"utf8_character_split"},
                "unsupported native alignment: "+repr(original["reasons"]))
        normalized,offsets,expanded=self.module._normalized_source(caption)
        require(not expanded and normalized==original["normalized_text"], "unprovable normalized offsets")
        words=list(tokenizer.pat.finditer(normalized))
        repaired=[]
        for row in out["all_content_tokens"]:
            if not row["reasons"]:
                continue
            require(set(row["reasons"])=={"utf8_character_split"}, "non-byte mapping error")
            match=words[row["word_index"]]
            byte=0;overlap=[]
            for i,ch in enumerate(match.group()):
                end=byte+len(ch.encode("utf-8"))
                if byte<row["byte_end"] and row["byte_start"]<end:
                    overlap.append((i,byte,end))
                byte=end
            require(overlap and 0<=row["byte_start"]<row["byte_end"]<=byte, "invalid byte interval")
            ns=match.start()+overlap[0][0];ne=match.start()+overlap[-1][0]+1
            start,end=offsets[ns][0],offsets[ne-1][1]
            require(start<end and all(len(caption[offsets[j][0]:offsets[j][1]])==1
                    for j in range(ns,ne)), "ambiguous codepoint mapping")
            row.update(start=start,end=end,raw=caption[start:end],span_status="valid",reasons=[])
            row["byte_span_proof"]={"normalized_word_byte_span":[row["byte_start"],row["byte_end"]],
                "source_character_span":[start,end],
                "source_utf8_span":[len(caption[:start].encode()),len(caption[:end].encode())],
                "whole_codepoint_token":False}
            repaired.append(row["token_index"])
        out["content_tokens"]=out["all_content_tokens"][:77-2]
        require([r["token_id"] for r in out["all_content_tokens"]]==list(tokenizer.encode(caption)),
                "byte bridge cannot change native tokens")
        out.update(status="valid",reasons=[],
            interface_policy="exact-byte-span-v1",
            native_alignment_status=original["status"],native_alignment_reasons=original["reasons"],
            repaired_byte_piece_indices=repaired)
        self.last=out
        return out

    def span_tokens(self,caption,span,tokens):
        # 原边界/覆盖/可见性规则原样复用。跨属性边界的BPE仍拒绝。
        return self.module._span_token_indices(caption,span,tokens)
