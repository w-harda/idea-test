# Step 5.1：Attribute Importance Score Prototype

独立服务器prototype，仅比较S_raw/S_norm，不改冻结v5.5接口/reader/basis和正式scorer/Top-K/G。

- 原始caption、frozen IRRA block8、fp32；原34054 clean train gallery，paired soft rank τ=.02（prototype参数）；ε=.01。
- 每caption一次torch.autograd.grad(F,H)，所有AttributeInstance复用G。
- g=wᵀG；p=Qᵀg；S_raw=||p||；S_norm=S_raw/sqrt(rank)。不加其他score。
- 复用v5.5定位：direct/hair用原slot alpha；upper/lower服装用原三个slot alpha均值；full_body仅原linked_object/type-source span上uniform；均按L2归一，baseline detach固定。
- validation的e直接来自冻结CachedReader，并核对fresh pooling和缓存回归；原9句在PID/cache外，仅按同一冻结规则做单caption forward/表示附加，不重建cache。
- Q从v5.5 full/semantic_bases.json原样加载，不运行centroid/SVD，不训练。
- rank0 score_unsupported，unresolved和不可用表示也排除；不填零、不fallback。
- DynamicTopK复用v5.5 float/opaque-instance adapter，beta=1，无正式Top-K改动；random永不进入Top-K。
- 同rank random：每axis独立CPU float64 QR，seed42，一次固定。不选seed/rank。

## 执行

默认只3句smoke；正式固定诊断原9句+数字row_id排序前16条validation（25句、3个validation PID），没有full模式。全validation禁止自动执行。

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_importance_step51_20261003
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest -v test_importance
# 以下只由用户手动运行。set -o pipefail保证tee不吞掉失败exit code。
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python -u run.py --mode diagnostic --allow-diagnostic 2>&1 | tee outputs/diagnostic.log
```

输出：本目录outputs/diagnostic/，已有同模式目录会拒绝覆盖。不要删除/覆盖结果以反复择优。

文件：protocol.json（70份冻结输入SHA及新源码SHA）、selection.json、results.json（每instance来源/e/w/g/p/d、score、真实F/cosine、Top-K）、rank_bias.json、audit.json、summary.md、completed.json。日志保留在outputs/diagnostic.log。

## Smoke事实与边界

22项tests通过；固定99:1、26:0、26:1共3caption/14scored/1backpack rank0。

实际预算最大相对误差2.0193e-6；一阶相对误差median=.2651%，max=19.0193%。最大项99:1 gender预测8.3908e-6、真实6.7949e-6，绝对差1.5959e-6；保留，不能仅凭小绝对差宣称所有项都通过严格线性近似，也未用新实验区分非线性与fp32 forward数值因素。

3句的k*为4/5/4，两score选中集合在这3句相同，2句排序改变。rank-score pooled Spearman raw=.5799/norm=.2186；3句不足以判定纯维数偏置或不合理排序，不冻结正式S。

S_raw对应固定w、正交Q的局部一阶最大等预算敏感度；S_norm是维数归一化排序诊断，不对应同ε的原图最大响应。方法选择等待固定25句结果，不运行攻击。

70份冻结文件前后SHA、zero exact、IRRA/reader无梯度/状态冻结、一次backward、cached e回归通过。参数训练无新增，正式tracked/staged diff为空；未commit/push。Windows仅Markdown摘要。
