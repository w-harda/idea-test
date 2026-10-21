# G60引导H10→H60适配续训

这是已完成H1–H10（G10奖励条件）之后，使用冻结G60图库进行Epoch11–60的适配续训，
不是全部60轮从零使用G60训练。只更新H，不训练G/IRRA；不执行官方Test或RaSa/APTM。

## 来源与复用

- 父H10：4ae964d51ea94840de71d9597d90a2234a36104e5d649cfee96577f59f94bacd。
- G60：9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049。
- IRRA：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。
- 原训练catalog/2048成员/资格/绝对Epoch顺序/Adam/SCST/Unicode规则均来自冻结旧源码。
- 真实CPU父恢复PASS：H、完整Adam、Python/NumPy/Torch/H sampler RNG exact。
  CUDA RNG保存1设备，在GPU Smoke加载冻结IRRA后再恢复并比较；CPU阶段不初始化CUDA。
- 原H10全局更新20370，参数Adam step实际为20124/20248/20370三种，按原值恢复，不改账本。
- 复用原Clean Train特征，新增2048 G60 Adv图像/IRRA特征；每图一次G。
  128 Validation直接复用上一阶段审计通过的G60 Gallery，H训练奖励不含这些成员。
- 新Extension保留完整原epochs=10协议和H10 SHA，目标60、起点11、奖励Gallery来源单独绑定。
  旧CLI/Checkpoint/current/Selection/协议均不调用写接口，不修改。

## 原函数与新接线

hv1_g60_state只管理来源、Extension、父恢复和新链审计。
hv1_g60_adaptation仅调度原CharacterPolicy、phase4_training.train_step、RewardBank/FreshReward、
formal_training.checkpoint.begin/append/publish/verify/restore。
没有复制SCST、奖励、Dynamic-K或Unicode算法；IRRAAdapter仅提供原生冻结图像/文本编码。
训练每合资格样本重新生成Sampled/Greedy，真实原文/二者3条文本GPU编码。
G生成仅在独立图库会话；后续训练/Validation图像前向硬拒绝。
原Adam lr1e-4、betas(.9,.999)、eps1e-8、batch1、seed42；无新Loss、Scheduler或梯度技巧。

## 授权入口与命令

工作树：/home/lzf/ldx/projects/idea-test-tbps-hv1-g60-adaptation-20261021。
新输出：/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60。
CLI默认CPU。以下阶段一旦已完成，不能重复启动或覆盖；任何失败保留现场，不自动重试。

~~~bash
cd /home/lzf/ldx/projects/idea-test-tbps-hv1-g60-adaptation-20261021
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src"
PY=/home/lzf/ldx/envs/tbps-image/bin/python
CFG="$PWD/configs/framework/hv1_g60_adaptation.server.json"
CUDA_VISIBLE_DEVICES="" "$PY" -B -m tbps.ablations.hv1_g60_adaptation --config "$CFG" --action preflight
CUDA_VISIBLE_DEVICES=0 "$PY" -B -m tbps.ablations.hv1_g60_adaptation --config "$CFG" --action gallery --allow-gpu
CUDA_VISIBLE_DEVICES="" "$PY" -B -m tbps.ablations.hv1_g60_adaptation --config "$CFG" --action audit-gallery
CUDA_VISIBLE_DEVICES=0 "$PY" -B -m tbps.ablations.hv1_g60_adaptation --config "$CFG" --action smoke --allow-gpu --allow-training
CUDA_VISIBLE_DEVICES="" "$PY" -B -m tbps.ablations.hv1_g60_adaptation --config "$CFG" --action audit-smoke
~~~

图库和4 visits Smoke（独立目录，不是正式Epoch11）审计均PASS之后，正式后台链只执行训练、预设Validation、新独立CPU审计和派生报告：

~~~bash
mkdir -p /home/lzf/ldx/tmp/hv1_g60_adaptation_epoch60_v1
nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-hv1-g60-adaptation-20261021/src/tbps/ablations/run_hv1_g60_adaptation.sh >> /home/lzf/ldx/tmp/hv1_g60_adaptation_epoch60_v1/launcher.log 2>&1 < /dev/null &
~~~

本任务由用户授权Codex在前置门禁后实际执行；是否已启动以实际启动报告和进程为准。
资源门槛：6144MiB空闲显存、32GiB可用RAM、35GiB磁盘，共享利用率只观察；不干预其他进程。

## 账本与安全停止

原H10：20480 visits/20370 updates。
新增50轮：102400 visits/101850 updates；H60累计122880 visits/122220 updates。
原11不合资格样本每轮记录NOOP/跳过，不删除。绝对Epoch11使用原order(seed42+10)。
每256更新/每完整Epoch保存不可变H/Adam/RNG及父子SHA；正式Epoch节点20/30/40/50/60内部Validation。
保存全部50轮；固定六节点选优含历史H10，不把旧H7/H9加入新六点候选后冒充预设选择。

正常终止监督器会向自己的训练子进程转发TERM，等待当前Epoch完整提交后写PARTIAL_USER_STOP。
未提交pending可能含已经执行的更新，必须保留并拒绝自动回放。
只有无pending、来源一致的显式--resume才可读取最后已提交Checkpoint，不跳过或重复计入Step：

~~~bash
CUDA_VISIBLE_DEVICES=0 "$PY" -B -m tbps.ablations.hv1_g60_adaptation --config "$CFG" --action train --allow-gpu --allow-training --resume
~~~

恢复需先检查中断证据并由用户决定，不由监督器自动调用。
PARTIAL写独立attempt记录，不能覆盖最终completed.json。
CPU恢复Fixture证明提交点恢复后H/Adam/RNG与不中断CPU轨迹一致；不宣称与历史未中断CUDA轨迹bitwise等价。

## 查看状态

~~~bash
cat /home/lzf/ldx/tmp/hv1_g60_adaptation_epoch60_v1/phase
tail -n 10 /home/lzf/ldx/tmp/hv1_g60_adaptation_epoch60_v1/status.log
tail -n 15 /home/lzf/ldx/tmp/hv1_g60_adaptation_epoch60_v1/training.log
~~~

主训练日志含HV1_G60_TRAIN_PROGRESS、HV1_G60_EPOCH_COMPLETE、HV1_G60_VALIDATION_COMPLETE。
最终成功仅在新独立CPU审计和Git报告发布后：
SUCCESS | HV1_G60_ADAPTATION_EPOCH60_VALIDATION_AND_CPU_AUDIT。

## 限制

重复使用原128内部Validation并披露选择偏差；不覆盖历史H7 Selection。
延长训练与G60条件变化未用G10控制分支分离，不作单因素因果断言。
未授权Test、跨模型Full、H61、新交互Loss或自动G10对照训练。
