# G-v2 Phase 1：逐属性训练监督缓存

本目录独立于 G-v1，仅实现条件缓存。G-v2 的输入契约为 G(I)，只输入图像。
z_t 仅作为 Global Loss 监督缓存，不作为 G-v2 输入。属性与文本只用于后续训练监督。

## 来源与数据

只读复用 G-v1 outputs/conditions 的2048 train、128内部validation样本及评分/选择结果。
split 保存官方划分 train；role 表示内部 train/validation。图片文件夹名称不能代替官方划分。
每条保存 row_id、image_rel_path、caption_index、split、role、pid、caption。
不重新评分，不修改 S_raw、gamma=3/beta=1/Natural Dynamic-K、JVP、reader或basis。

condition.pt 使用独立 schema gv2-attribute-supervision-1：

- K、selected_instance_ids、S_raw（float64 [K]）、score_weights（float64 [K]）。
- attribute_jvps（float32 [K,512]），保留原尺度。
- z_t（float32 [512]），原完整caption单位化retrieval feature。
- v_local（float32 [512]），原S归一权重加权JVP，不再次单位化。
- has_local，以及完整样本标识。

K=0协议固定为空ID、空分数、空权重、[0,512] JVP、零v_local、has_local=false。
每次构建、写入和读取均从逐属性JVP重建聚合并校验；metadata和summary记录误差。
CPU持久化聚合采用原fp32乘法和sum表达式；与原设备/G-v1聚合比较 atol=1e-9、rtol=1e-5。
该容差只用于缓存聚合数值检查，不改变JVP、score、选择或攻击方法。
原JVP继续使用zero exact与FD relative error <=0.02门禁。

## 安全与恢复

写入仅限本目录解析后的outputs，拒绝符号链接逃逸。
逐样本写入pending目录，fsync后原子发布；未发布pending不参与恢复。
不覆盖已发布样本，恢复须通过协议、来源、SHA、identity、K、ID顺序和聚合检查。
完成缓存再次执行仅CPU验证，不重新加载victim或计算JVP。
所有来源及本阶段源码/config SHA记录在protocol；来源改变需新输出。
当前仅提供固定5条smoke入口，不提供完整缓存/训练/评测入口。

## 手动命令

服务器Python：/home/lzf/ldx/envs/tbps-image/bin/python。
先进入本目录；为防止冻结依赖产生字节码，始终设置 PYTHONDONTWRITEBYTECODE=1。

~~~bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_conditions -v
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python prepare.py
PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python smoke_conditions.py --allow-smoke
~~~

smoke固定选择G-v1 manifest顺序中的首条 train K=0/1/3/7和validation K=4，共5条。
不会重新抽样或按新结果选样。新增计算仅原text-forward JVP及zero/FD检查、CPU聚合与SHA验证。
new_scoring_backwards_this_run=0；JVP内部autograd.grad调用单独统计，不能说本阶段没有梯度计算。
没有G、optimizer、训练backward、图像编码、TTA、完整缓存准备或Test使用。
Retain Loss和训练调度留待后续阶段确认。
