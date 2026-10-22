"""仅探针：复用原始投影和Loss计算梯度，不执行任何Adam更新。"""
from .contracts import require
from .parents import no_optimizer_steps
from .state import tree_sha

def original_gradient(g, encoder, image, condition, native):
    import torch
    # 只比较原K+1训练器的第一步目标；不模拟/替代完整串行训练。
    g.train()  # 与原train_single_sample第一步模式一致；状态SHA仍须完全不变。
    original=image.detach().clone()
    delta=g(original)
    adv=native.get("budget_projection").add_projected_delta(original, original, delta)
    feature=encoder(adv)
    losses=native.get("losses")
    z=condition["z_t"].to(image)[None]
    serial=native.get("serial_training")
    if condition["K"]:
        local=condition["attribute_jvps"][0:1].to(image)
        terms=losses.attribute_losses(feature,z,local)
        total=serial._weighted(terms,serial.LossWeights(),False)
    else:
        terms=losses.joint_losses(feature,z,condition["v_local"].to(image)[None],
                                  has_local=torch.tensor([condition["has_local"]],dtype=torch.bool,device=image.device))
        total=serial._weighted(terms,serial.LossWeights(),True)
    gradients=torch.autograd.grad(total,tuple(g.parameters()),allow_unused=False)
    require(all(torch.isfinite(x).all() for x in gradients), "nonfinite G probe gradient")
    return tuple(x.detach().cpu().clone() for x in gradients), float(total.detach())

def compare_gradients(g,encoder,rows,native,make_condition):
    """rows固定NOOP/非NOOP；真实与Fixture均运行这段完整控制流。"""
    import torch
    require(len(rows)==2 and rows[0]["noop"] and not rows[1]["noop"], "fixed negative/positive controls")
    before=tree_sha(g.state_dict())
    result={}
    with no_optimizer_steps():
        for row in rows:
            original=row["original"]
            changed=make_condition(row)
            if row["noop"]:
                require(tree_sha(original.payload)==tree_sha(changed.payload), "NOOP conditions differ")
            a,la=original_gradient(g,encoder,row["image"],original.payload,native)
            b,lb=original_gradient(g,encoder,row["image"],changed.payload,native)
            difference=max(float((x-y).abs().max()) for x,y in zip(a,b))
            result["noop" if row["noop"] else "nonnoop"]={
                "max_abs":difference,"original_loss":la,"context_loss":lb,
                "original_gradient_sha256":tree_sha(a),"context_gradient_sha256":tree_sha(b),
                "condition_evidence":changed.evidence}
    require(tree_sha(g.state_dict())==before,"probe modified G parameters/buffers")
    return result
