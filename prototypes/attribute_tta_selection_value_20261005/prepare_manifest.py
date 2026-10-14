"""纯CPU selection-only准备，不评分、不提特征、不启动攻击。"""
import deps
from selection import build,cache_sources
from selection_audit import load,save,new_output,digest,verify
def main():
    deps.assert_formal_clean();hashes=deps.frozen_hashes()
    reference,protocol=cache_sources()
    split=load(deps.V4/"outputs/split.json")
    validation={split["records"][i]["row_id"]:split["records"][i] for i in split["validation_indices"]}
    manifest=build(deps.rows(reference["score_cache_path"]),validation,reference)
    manifest.update({k:reference[k] for k in ["score_cache_path","score_cache_sha256","score_protocol_path","score_protocol_sha256","split_sha256"]})
    manifest["source_reference_manifest_sha256"]=digest(deps.P74/"outputs/clean_scores/selection_manifest.json")
    manifest["full_preparation"]=True
    out=new_output(deps.ROOT/"outputs/manifest")
    save(out/"selection_manifest.json",manifest)
    save(out/"completed.json",{k:v for k,v in manifest.items() if k!="rows"})
    verify(hashes);deps.assert_formal_clean()
    print("MANIFEST_COMPLETE",{k:v for k,v in manifest.items() if k not in ("rows","candidate_axis_counts")},flush=True)
if __name__=="__main__":main()
