"""显式小规模CPU官方接口对照；合成图像/分数，少量Tokenizer，不加载真实模型。"""
from pathlib import Path
import ast
import json
import argparse
from ..artifacts import ArtifactError, checked, read_json
from .irra_source import IRRASource
from .irra_io import official_tokenizer, pixels_from_pil, normalize_pixels, PREPROCESS, METRICS

def official_function(source,relative,name,globals_):
    ref=source.ref(relative);checked(ref)
    tree=ast.parse(Path(ref.path).read_text(encoding="utf-8"))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(nodes)!=1:raise ArtifactError("expected single official function")
    # 只执行该纯函数；不import官方datasets/model包或其训练入口。
    ns=dict(globals_);exec(compile(ast.Module(body=nodes,type_ignores=[]),ref.path,"exec"),ns)
    checked(ref);return ns[name]

def check_io(source):
    import torch
    import torchvision.transforms as T
    from PIL import Image
    if torch.cuda.is_initialized():raise ArtifactError("CPU comparison requires uninitialized CUDA")
    tokenizer=official_tokenizer(source)
    reference_tokenize=official_function(source,"datasets/bases.py","tokenize",{"torch":torch})
    captions=("red shirt","  red  red shirt  ","rеd shirt","blue ѕhirt","a café &amp; coat","帽子 blue coat",
              "A man's black T-shirt.","<|mask|> red","red "*100,"")
    token_reports=[]
    for caption in captions:
        report=tokenizer.inspect(caption)
        expected=reference_tokenize(caption,tokenizer._tokenizer,text_length=77,truncate=True)
        actual=torch.tensor(report.ids,dtype=torch.long)
        if not torch.equal(actual,expected):raise ArtifactError("official tokenization mismatch")
        token_reports.append({"utf8_bytes":len(caption.encode()),"tokens_before_truncation":report.untruncated_length,"truncated":report.truncated,"exact":True})
    reference_transform=official_function(source,"datasets/build.py","build_transforms",{"T":T})((384,128),is_train=False)
    synthetic=[]
    for mode,size,color in (("RGB",(29,61),(14,117,239)),("L",(63,17),129),("RGBA",(7,13),(80,190,5,70))):
        image=Image.new(mode,size,color).convert("RGB")
        actual=normalize_pixels(pixels_from_pil(image)[None])[0]
        expected=reference_transform(image)
        if not torch.equal(actual,expected):raise ArtifactError("official PIL preprocessing mismatch")
        synthetic.append({"mode":mode,"source_size":list(size),"max_abs_error":float((actual-expected).abs().max())})
    # 仅合成4x12分数，验证官方rank定义与文档；不读取任何Full特征。
    rank=official_function(source,"utils/metrics.py","rank",{"torch":torch})
    generator=torch.Generator().manual_seed(42)
    text=torch.nn.functional.normalize(torch.rand(4,37,generator=generator),dim=-1)
    image=torch.nn.functional.normalize(torch.rand(12,37,generator=generator),dim=-1)
    scores=text@image.T;query_pids=torch.tensor([0,1,2,0]);gallery_pids=torch.tensor([0,1,2]*4)
    cmc,map_,minp,indices=rank(scores,query_pids,gallery_pids,max_rank=10,get_mAP=True)
    manual_ap=[];manual_inp=[];manual_cmc=[]
    for q,order in zip(query_pids,indices):
        matches=gallery_pids[order].eq(q);positions=matches.nonzero().flatten()+1
        manual_ap.append((torch.arange(1,len(positions)+1)/positions).mean()*100)
        manual_inp.append(len(positions)/positions[-1]*100)
        manual_cmc.append((positions[0]<=torch.arange(1,11)).float()*100)
    if not torch.allclose(map_,torch.stack(manual_ap).mean(),atol=1e-5,rtol=0) or not torch.allclose(minp,torch.stack(manual_inp).mean(),atol=1e-5,rtol=0) or not torch.equal(cmc,torch.stack(manual_cmc).mean(0)):
        raise ArtifactError("official small synthetic rank definition mismatch")
    if torch.cuda.is_initialized():raise ArtifactError("unexpected CUDA initialization")
    return {"schema":"tbps-stage2-cpu-io-1","tokenizer_cases":token_reports,"preprocess_cases":synthetic,
            "official_rank_fixture_shape":[4,12],"rank_definition_passed":True,
            "preprocess_contract":PREPROCESS,"ranking_contract":METRICS,"real_model_loaded":False,
            "real_image_text_forwards":0,"formal_ranking_recomputed":False,"cuda_initialized":False}

def main(argv=None):
    parser=argparse.ArgumentParser(description="Small explicit CPU-only official I/O equivalence")
    parser.add_argument("--sources",required=True);args=parser.parse_args(argv)
    try:result=check_io(IRRASource.from_dict(read_json(args.sources)["irra"]))
    except (ValueError,TypeError,OSError) as exc:parser.exit(1,"CPU_IO_REJECTED "+str(exc)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2));print("STAGE2_CPU_IO_EXACT_PASS")

if __name__=="__main__":main()
