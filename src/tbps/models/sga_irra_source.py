"""固定SGA的IRRA源接口；只转换BERT输入，不含攻击Loss或迭代。"""
import torch
import torch.nn.functional as F
from tbps.artifacts import ArtifactError

def require(ok, message):
    if not ok:
        raise ArtifactError(message)

class AuditedBert:
    """仍委托官方BertTokenizer；元数据不进入MLM Tensor字典。"""
    def __init__(self, tokenizer):
        self.native = tokenizer
    def __getattr__(self, name):
        return getattr(self.native, name)
    def __call__(self, texts, *args, **kwargs):
        result = self.native(texts, *args, **kwargs)
        raw = (texts,) if isinstance(texts, str) else tuple(texts)
        object.__setattr__(result, "_sga_original_texts", raw)
        return result

def decode_bert(batch, tokenizer):
    """移除真正的CLS/attention padding（官方单句不添加SEP），保留内部[UNK]及MLM候选。"""
    output = []
    raw = getattr(batch, "_sga_original_texts", (None,) * len(batch.input_ids))
    for i, (ids, mask) in enumerate(zip(batch.input_ids, batch.attention_mask)):
        visible = ids[mask.bool()].detach().cpu().tolist()
        require(visible[0] == tokenizer.cls_token_id, "official BERT CLS boundary")
        content = visible[1:]
        text = tokenizer.decode(content, skip_special_tokens=False,
                                clean_up_tokenization_spaces=True).strip()
        require(tokenizer.encode(text, add_special_tokens=False) == content,
                "BERT decode/re-encode loses candidate token IDs")
        full = tokenizer.encode(raw[i], add_special_tokens=True) if raw[i] is not None else visible
        output.append({"input_text": raw[i], "bert_visible_ids": visible, "decoded_text": text,
                       "bert_untruncated_tokens": len(full), "bert_truncated": len(full) > len(visible),
                       "wordpiece_roundtrip_exact": True,
                       "UNK_count": content.count(tokenizer.unk_token_id)})
    return output

class IRRASGASource(torch.nn.Module):
    def __init__(self, backend, irra_tokenizer, bert_tokenizer, *, write_trace=None):
        super().__init__()
        self.backend = backend.float().eval().requires_grad_(False)
        self.irra_tokenizer = irra_tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.write_trace = write_trace or (lambda row: None)
        self.counts = {"image_samples": 0, "text_samples": 0, "bridge_rows": 0,
                       "BERT_truncated_rows": 0, "UNK_rows": 0}
        self.last_text = None
        self.eval()
    def inference_image(self, normalized_pixels):
        require(normalized_pixels.dtype == torch.float32 and torch.isfinite(normalized_pixels).all(),
                "finite normalized FP32 IRRA input")
        self.counts["image_samples"] += len(normalized_pixels)
        features = F.normalize(self.backend.encode_image(normalized_pixels).float(), dim=-1)
        require(features.shape == (len(normalized_pixels), 512), "native 512-D IRRA image features")
        return {"image_feat": features}
    def inference_text(self, inputs):
        trace = decode_bert(inputs, self.bert_tokenizer)
        texts = [r["decoded_text"] for r in trace]
        # 原生Byte-BPE，不把BERT ID直接送入IRRA；不静默截断额外候选。
        ids, reports = self.irra_tokenizer.batch(texts, truncate=False)
        device = next(self.backend.parameters()).device
        features = F.normalize(self.backend.encode_text(ids.to(device)).float(), dim=-1)
        require(features.shape == (len(texts), 512), "native IRRA text feature dimensions")
        for row, token_ids, report in zip(trace, ids.tolist(), reports):
            row.update(irra_ids=token_ids, irra_truncated=report.truncated)
            self.write_trace(row)
            self.counts["bridge_rows"] += 1
            self.counts["BERT_truncated_rows"] += int(row["bert_truncated"])
            self.counts["UNK_rows"] += int(row["UNK_count"] > 0)
        self.counts["text_samples"] += len(texts)
        self.last_text_requests = [row["input_text"] for row in trace]
        self.last_text = (texts, features.detach().cpu().clone())
        return {"text_feat": features}
