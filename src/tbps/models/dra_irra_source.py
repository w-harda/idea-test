"""DRA的IRRA源I/O：复用已验证Token桥，仅管理官方反向所需的参数开关。"""
from .sga_irra_source import IRRASGASource, AuditedBert, decode_bert, require

class IRRADRASource(IRRASGASource):
    """权重固定与requires_grad=False不同；官方候选评分保留参数反向图，无Optimizer。"""
    def __init__(self, backend, irra_tokenizer, bert_tokenizer, *, write_trace=None):
        super().__init__(backend, irra_tokenizer, bert_tokenizer, write_trace=write_trace)
        self.backend.requires_grad_(True)
        self.eval()
