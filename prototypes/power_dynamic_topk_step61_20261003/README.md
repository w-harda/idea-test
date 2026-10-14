# Step 6.1 Power-Sharpened Dynamic Top-K

仅读取Step5.1固定25句diagnostic中118个positive S_raw，产生100个selection payload。

不导入torch/numpy/IRRA/reader/scorer；不调用forward/backward、intervention、提feature或攻击；无训练。冻结S_raw、AttributeInstance及所有ontology/basis/provenance保持原样。rank0/unresolved沿旧状态排除，S=0显式排除（当前0例）。

固定gamma=1/1.5/2/3，beta=1，kmax=4；p=softmax(gamma*log S_raw)，H=-sum p log p，N_eff=exp(H)，k=min(ceil(N_eff),m,4)。排序永远原S_raw descending、输入顺序tie-break。没有kmax3。空caption约定H/N_eff/k全0。

## 执行与产物

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/power_dynamic_topk_step61_20261003
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_selection
PYTHONDONTWRITEBYTECODE=1 python3 -u run.py
```

本轮tests及固定25句已由Codex完成，不需要重新运行。已有outputs/diagnostic会拒绝覆盖。

- outputs/tests.log：19项tests。
- protocol.lock.json：5份原Step5.1结果SHA及预先确定的5个例子；gamma/上限固定，不选样。
- outputs/diagnostic/selection_payloads.json：100个严格指定key的payload；probabilities由instance_id索引，ranked/selected实例包含原source/e/region/value/S_raw/provenance。
- summary.json：gamma汇总，含未cap的natural k和cap作用、denominators、逐caption变化ID。
- examples.json：预先固定35:1/0:0/61:0/99:1/27:1的原S/概率/N/K/selected。
- summary.md：人可读完整表。
- audit.json / protocol.json / completed.json；原输入SHA前后不变，排序/Top1不变、prefix嵌套、纯stdlib审计。

## 当前推荐

推荐gamma=3作为下一步selection设置候选，不修改正式Top-K或自动视为用户已冻结gamma。

gamma2仍14/25句k=4，median N=3.2198导致ceil=4；gamma3有14/25句自然k=2–3，9句k=4，仅1句依赖cap。其23个m>=2句没有全选，且都仍选至少2个，满足更集中、保留动态性及多个高排名实例的当前目标。没有使用attack success或retrieval degradation选择gamma。

限制：25句小样本，16validation只3PID；实例可包含同axis/value多个source span（61:0两个t_shirt保留）。gamma3将99:1的hair_length第三名移出，只剩color/type；尖锐35:1的top1概率升至.9984，但ceil仍选2，第二项概率很低。保留多个实例不等于多语义覆盖或攻击有效性；本轮不改去重、ontology、score、ceil或kmax。

相比旧Step5.1无cap，gamma1仅因kmax4就改变10句K；不得把该效果归因power sharpening。本批meanN/meanK包含1个空caption；k/m与全选率以24个非空为分母，另外列23个m>=2结果。

源代码、tests、JSON/logs全部服务器；Windows仅Markdown摘要；正式tracked/staged diff空，无commit/push。
