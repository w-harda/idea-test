"""SEP固定1个原始空格词位置；官方AST只适配必要预算及探针边界。"""
from collections import OrderedDict
from pathlib import Path
import ast
import math
import re
import sys
import time

from tbps.artifacts import sha_file
from tbps.baselines.cpgc_sep.dataset import require
from tbps.baselines.cpgc_sep.official import sep_functions
from tbps.baselines.cpgc_sep.sep_bridge import legal_semantic_flags
from operations.sep_source_only.attack import resources,s1_ensemble

PROTOCOL="SEP-TBPS-TextOnly-WordBudget1"


def budget_check(original,candidate):
    a=original.split() if isinstance(original,str) else list(original)
    b=candidate.split() if isinstance(candidate,str) else list(candidate)
    require(len(a)==len(b),f"WordBudget1 insertion/deletion: original={a!r}; candidate={b!r}")
    changed=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    require(len(changed)<=1,f"WordBudget1 violation positions={changed}: original={a!r}; candidate={b!r}")
    return changed


def restore_original(raw,candidate):
    positions=budget_check(raw,candidate)
    if not positions:return raw
    spans=list(re.finditer(r"\S+",raw))
    index=positions[0];span=spans[index];word=candidate.split()[index]
    return raw[:span.start()]+word+raw[span.end():]


def prepare_caption(_c,raw):
    return " ".join(raw.split())


def aliases(raw,vocabulary):
    mapping,inverse,neighbors=vocabulary
    lookup=dict(mapping)
    for word in raw.split():
        # Alias only vocabulary lookup. Other original positions keep their exact bytes.
        match=re.fullmatch(r"[^A-Za-z]*([A-Za-z]+(?:['-][A-Za-z]+)*)[^A-Za-z]*",word)
        if match and match.group(1).lower() in mapping:
            lookup[word]=mapping[match.group(1).lower()]
    return lookup,inverse,neighbors


def eligible_positions(raw,vocabulary,pos):
    mapping,inverse,neighbors=aliases(raw,vocabulary)
    result=[]
    for i,(word,tag) in enumerate(zip(raw.split(),pos)):
        if tag not in ("ADJ","ADV","VERB","NOUN") or len(word)<=2 or word not in mapping:continue
        values=[inverse[j] for _,j in neighbors[mapping[word]]]
        require(all(v and len(v.split())==1 for v in values),"official synonym must be one whitespace word")
        if len(values)>1 and any(v!=word for v in values):
            result.append(i)
    return result


def verify_edits(clean,adv,vocabulary,pos):
    budget_check(clean,adv)
    mapping,inverse,neighbors=aliases(clean,vocabulary)
    info=legal_semantic_flags(clean,adv)
    for edit in info["edits"]:
        word=edit["original"]
        require(word in mapping and pos[edit["position"]] in ("ADJ","ADV","VERB","NOUN") and len(word)>2,
                "illegal SEP edit position")
        require(edit["replacement"] in {inverse[i] for _,i in neighbors[mapping[word]]},
                "replacement outside original top50")
    return {**info,"word_budget":1}


def budget_functions(c,original):
    """按SHA加载原定义，限定替换两项预算、整词截断和单位置重要性探针。"""
    functions=sep_functions(c)
    path=Path(c["official"]["sep"]["root"])/"SEP_Attack.py"
    require(sha_file(path)==c["official"]["sep"]["files"][str(path)],"official SEP source SHA")
    names=("get_change_ls","get_important_order","find_max_synonym","optimize","early_stop","attack")
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    changed={"max_change_num":0,"att_succ_num":0,"text_ls":0,"probe_base":0,"probe_mask":0,"singleton":0}
    for function in nodes:
        if function.name=="attack":
            for node in ast.walk(function):
                if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):continue
                name=node.targets[0].id
                if name in ("max_change_num","att_succ_num") and isinstance(node.value,ast.Call) and getattr(node.value.func,"id",None) in ("min","max"):
                    # The later min(...,len(words_perturb)) assignments stay original.
                    if any(isinstance(n,ast.Attribute) and n.attr=="floor" for n in ast.walk(node.value)):
                        node.value=ast.copy_location(ast.Call(func=ast.Name(id="min",ctx=ast.Load()),
                            args=[ast.Constant(value=1),ast.Name(id="len_text",ctx=ast.Load())],keywords=[]),node.value)
                        changed[name]+=1
                if name=="text_ls" and isinstance(node.value,ast.Subscript) and isinstance(node.value.slice,ast.Slice):
                    if isinstance(node.value.slice.upper,ast.Constant) and node.value.slice.upper.value==256:
                        node.value.slice.upper=None;changed["text_ls"]+=1
        elif function.name=="get_important_order":
            for node in ast.walk(function):
                if isinstance(node,ast.Assign) and len(node.targets)==1:
                    target=node.targets[0]
                    if isinstance(target,ast.Name) and target.id=="new_text":
                        node.value=ast.copy_location(ast.Call(func=ast.Name(id="list",ctx=ast.Load()),
                            args=[ast.Name(id="budget_original_words",ctx=ast.Load())],keywords=[]),node.value)
                        changed["probe_base"]+=1
                    if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=="new_text":
                        require(isinstance(node.value,ast.Constant) and node.value.value=="","original importance deletion")
                        node.value=ast.copy_location(ast.Constant(value="[UNK]"),node.value)
                        changed["probe_mask"]+=1
        elif function.name=="optimize":
            for node in ast.walk(function):
                if isinstance(node,ast.While):
                    # A singleton has probability 1. Its zero span cannot be scaled to 1.
                    condition=ast.Compare(left=ast.Call(func=ast.Name(id="len",ctx=ast.Load()),
                        args=[ast.Name(id="wi_score",ctx=ast.Load())],keywords=[]),
                        ops=[ast.Gt()],comparators=[ast.Constant(value=1)])
                    node.test=ast.copy_location(ast.BoolOp(op=ast.And(),values=[condition,node.test]),node.test)
                    changed["singleton"]+=1
    require(changed==dict.fromkeys(changed,1),"fixed WordBudget1 AST delta mismatch")
    ns={**functions,"budget_original_words":tuple(original)}
    module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(module,str(path),"exec"),ns)
    return ns


def attack_caption(c,raw,classifier,device,vocabulary,report):
    import torch
    start=time.monotonic()
    words=raw.split()
    clean=" ".join(words)
    calls,hits=classifier.calls,classifier.hits
    requests,oracle_calls,candidates,timings=[],[],[],[]
    phase="SOURCE_PSEUDO_PID"
    mutation_checks=0

    def observed(_module,args,kwargs):
        texts=tuple(args[0] if args else kwargs["texts"])
        requests.append({"phase":phase,"texts":texts})
        for text in texts:
            try:budget_check(words,text)
            except Exception:
                report("WORD_BUDGET_VIOLATION",phase=phase,candidate=text,original=raw)
                raise
        if len(requests)%16==0:
            report("CANDIDATE_PROGRESS",phase=phase,requests=len(requests),calls=classifier.calls-calls,
                   hits=classifier.hits-hits,elapsed=time.monotonic()-start)

    def stage(name,**fields):
        nonlocal phase
        phase=name
        record={"stage":name,"elapsed":time.monotonic()-start,**fields}
        timings.append(record);report("QUERY_STAGE",**record)

    hook=classifier.register_forward_pre_hook(observed,with_kwargs=True)
    old_trace=sys.gettrace()
    legal=[]
    result=None
    dpp={"N":1,"E":1,"draws":0,"weights":[]}
    try:
        logits=classifier([clean]).logits.detach().cpu()
        require(logits.shape==(1,len(classifier.pid_plan.order)) and torch.isfinite(logits).all(),"finite Source PID scores")
        label=int(logits[0].argmax())
        functions=budget_functions(c,words)
        pos=functions["criteria"].get_pos(words) if words else []
        legal=eligible_positions(clean,vocabulary,pos)
        if not legal:
            chosen=clean
            reason="NO_LEGAL_SYNONYM"
        else:
            attack=functions["attack"]
            optimizer=functions["optimize"]
            # Restrict official vocabulary aliases to the original eligible single-word positions.
            lookup,inverse,neighbors=aliases(raw,vocabulary)
            allowed={words[i] for i in legal}
            lookup={w:lookup[w] for w in allowed}
            # The original attack still sees POS eligible OOV words. Its importance
            # probes remain legal and no OOV synonym can be selected.
            tree=ast.parse(Path(c["official"]["sep"]["root"],"SEP_Attack.py").read_text())
            tau=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Subscript)
                     and isinstance(n.iter.value,ast.Name) and n.iter.value.id=="log_stack_sub")
            scaling=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.While)
                         and isinstance(n.test,ast.Compare))
            optimize_line=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                               and isinstance(n.value.func,ast.Name) and n.value.func.id=="optimize")

            def oracle(texts,orig_label,_cache):
                require(orig_label==label,"fixed Source pseudo PID")
                value=classifier(texts=texts).logits.argmax(-1)!=label
                oracle_calls.append({"texts":list(texts),"source_top1_changed":value.cpu().tolist()})
                return value.squeeze()

            ensemble=s1_ensemble(c,classifier,device)
            codes={functions[name].__code__ for name in ("attack","optimize","get_important_order","find_max_synonym")}
            def trace(frame,event,_arg):
                nonlocal mutation_checks
                if frame.f_code not in codes:return None
                if event=="line":
                    for name in ("x_t","x_t_tmp","asn_text","new_text","tmp_text"):
                        value=frame.f_locals.get(name)
                        if isinstance(value,list) and len(value)==len(words) and all(isinstance(x,str) for x in value):
                            budget_check(words,value);mutation_checks+=1
                    if frame.f_code is optimizer.__code__ and frame.f_lineno==scaling:
                        importance=frame.f_locals["wi_score"]
                        if not torch.isfinite(importance).all() or (len(importance)>1 and (importance.max()-importance.min()).item()==0):
                            raise RuntimeError("Official SEP importance scaling cannot terminate for constant/nonfinite scores")
                    if frame.f_code is attack.__code__ and frame.f_lineno==optimize_line:
                        require(frame.f_locals["max_change_num"]==frame.f_locals["att_succ_num"]==1,"search budget must be 1")
                        stage("OFFICIAL_OPTIMIZE",iteration=frame.f_locals["t"],max_change_num=1,att_succ_num=1)
                    if frame.f_code is attack.__code__ and frame.f_lineno==tau and frame.f_locals["ms_idx"]==1:
                        candidates[:]=[{"text":" ".join(t),"tau":float(score)} for t,score in frame.f_locals["log_stack_sub"]]
                        for candidate in candidates:budget_check(words,candidate["text"])
                return trace

            sys.settrace(trace)
            result=attack(1000000,words[:],label,oracle,lookup,inverse,neighbors,ensemble,device,.9,0,
                          "source-self-consistency",".",max_change_rate=.2)
            require(result[0] is not None and oracle_calls[0]["source_top1_changed"]==[False],"Source pseudo PID initial gate")
            require(ensemble.weights_zoo.cpu().tolist()==[[1.]],"S1 DPP degenerate weight")
            dpp={"N":1,"E":1,"draws":100,"weights":[[1.]]}
            require(all(math.isfinite(x["tau"]) for x in candidates),"finite tau")
            chosen=candidates[0]["text"] if candidates else clean
            reason="NO_DISTINCT_OFFICIAL_TAU_CANDIDATE" if chosen==clean else None
    finally:
        sys.settrace(old_trace);hook.remove()
    edits=verify_edits(clean,chosen,vocabulary,pos)
    adversarial=restore_original(raw,chosen)
    positions=budget_check(raw,adversarial)
    require(len(positions)==edits["word_change_count"],"raw original one-word mapping")
    forward_calls=classifier.calls-calls;cache_hits=classifier.hits-hits
    return {"protocol":PROTOCOL,"original":raw,"matched_clean":clean,"selected":chosen,"adversarial":adversarial,
            "noop":not positions,"noop_reason":reason,"word_budget":1,"word_positions":positions,
            "pseudo_pid_index":label,"pseudo_pid":classifier.pid_plan.order[label].value,"candidates":candidates,
            "requests":requests,"source_oracle":oracle_calls,"DPP":dpp,"original_source_statistics":result,
            "word_edits":edits,"pos":pos,"legal_positions":legal,"mutation_checks":mutation_checks,
            "budget_violations":0,"calls":forward_calls,"cache_hits":cache_hits,"text_forwards":forward_calls-cache_hits,
            "cache_hit_rate":cache_hits/forward_calls,"timings":timings,"elapsed_seconds":time.monotonic()-start},logits
