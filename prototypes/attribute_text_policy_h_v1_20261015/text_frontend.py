"""Caption-only Canonical前端；复用冻结规则及原Span，不读取评分或数据集。"""
from dataclasses import asdict
import importlib.util
import sys
from contracts import TextExample, AttributeSpan
from phase2_sources import REPO, ROOT, sha, checked, read_json

class CaptionFrontend:
    def __init__(self):
        config = read_json(ROOT / "config_phase2.json")
        for relative, expected in config["frontend_sources"].items():
            checked(REPO / relative, expected)
        sys.path.insert(0, str(REPO))
        from src.attributes.text_extractor import extract_with_provenance
        path = REPO / "prototypes/attribute_unit_v55_interface_20261003/attribute_interface.py"
        spec = importlib.util.spec_from_file_location("h_v1_caption_only_interface", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        # 动态依赖也须确实来自锁定文件，禁止同名sys.modules劫持。
        for m in (module.audit, module.audit.old):
            relative = str(__import__("pathlib").Path(m.__file__).resolve().relative_to(REPO))
            checked(m.__file__, config["frontend_sources"][relative])
        self.extract = extract_with_provenance
        self.adapter = module
        self.sources = config["frontend_sources"]

    def convert(self, caption):
        if type(caption) is not str or not caption.strip():
            raise TypeError("caption string only; records/PID/images forbidden")
        result = self.extract(caption)
        instances = self.adapter.convert_record({"row_id": "caption-only",
                                                 "caption": caption, **result})
        attrs, seen = [], set()
        for a in self.adapter.text_candidates(instances, caption, require_representation=False):
            attr = AttributeSpan(a.semantic_axis, a.canonical_value, a.region,
                                 a.raw_text, a.char_start, a.char_end)
            attr.validate(caption)
            if attr.attribute_id not in seen:
                seen.add(attr.attribute_id)
                attrs.append(attr)
        example = TextExample(caption, tuple(attrs))
        audit = {"input_contract": "caption-only", "source_sha256": self.sources,
                 "candidate_instances": len(instances), "resolved_attributes": len(attrs),
                 "unresolved": [{"axis": a.semantic_axis, "raw": a.raw_text,
                                 "reasons": list(a.unresolved_reasons)}
                                for a in instances if a.unresolved],
                 "attributes": [asdict(a) for a in attrs],
                 "exact_span_count": sum(caption[a.char_start:a.char_end] == a.raw_text for a in attrs)}
        return example, audit

    def __call__(self, caption):
        return self.convert(caption)[0]
