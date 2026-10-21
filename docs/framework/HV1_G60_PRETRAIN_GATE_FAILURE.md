# H10→H60 G60适配：前置验证与CPU审计门禁失败报告

日期：2026-10-21（Asia/Shanghai）。
执行代码Commit：9765a536a11a9a0e9111f2f834230c53d41bf2d8。
独立分支codex/hv1-g60-adaptation-epoch60。

**当前状态：STOPPED_SMOKE_CPU_AUDIT_CODE_ERROR。正式H11–H60训练未启动。**
没有自动重试GPU、重复生成图库、启动H正式训练或官方Test。
训练、Validation六节点及最终审计均不能标记完成。

## 已完成的前置工作

| 阶段 | 实际结果 |
| --- | --- |
| CPU父H10完整恢复 | PASS；20480 visits/20370 updates，H/Adam/Python/NumPy/Torch/H sampler状态一致 |
| CUDA RNG | 原H10保存1设备；实际GPU恢复及Checkpoint重载exact |
| CPU回归 | 78个不同用例通过；初轮75、追加受影响/恢复24，公共用例不重复计数 |
| 扩展协议设计 | 原epochs=10完整保留，新目标60、起点11、原绝对Epoch顺序、G60奖励图库及独立输出绑定 |
| G60 Train图库 | 2048/2048一次生成；实际G/IRRA图像前向2048/2048；加载各1次 |
| 独立图库CPU审计 | G60_TRAIN_GALLERY_CPU_AUDIT_PASS |
| H GPU Smoke | HV1_G60_SMOKE_PASS；4 visits/4 H更新、12真实文本编码，G/图像前向0 |
| 独立Smoke CPU审计 | **FAIL：新增审计器Python变量遮蔽，未发布PASS回执** |
| 正式H训练 | **未启动；training目录不存在，正式新增更新0** |

H10的参数Adam step实际有20124、20248、20370三种，完整恢复原值，没有统一改账本。
GPU恢复前H状态SHA为1b1e3888ee8dc0e4dff39af3438ec48987cd472746cbfde71ad4d10f562ad24d，
Adam状态SHA为dfe930621066f140b5dd37d77d5a4fc4226cdef3f74ff05bc181de86423b6a4d，
全部RNG状态SHA为3a0a00a5a10c9cc029f9da14a4367026d837f2230103256fef07494dab7fb804，与CPU检查一致。

## 错误定位

文件：src/tbps/ablations/hv1_g60_audit.py。

- 第31行将逐Epoch统计字典赋给局部变量stat。
- 第62行期望调用导入的文件状态函数stat(path)，实际得到该局部dict，触发原始错误：
  TypeError: 'dict' object is not callable。
- 错误发生在历史保护文件检查循环，独立审计进程exit1。
- 这是本轮新增CPU审计器的命名错误；不能将GPU Smoke PASS替代该审计PASS。

CPU测试覆盖了原Checkpoint恢复、提交点CPU轨迹、来源SHA、调度、原SCST、角色隔离等，
但没有用组合Fixture走完新独立审计器的历史保护循环，这是本轮测试覆盖缺口。

完整失败日志：
/home/lzf/ldx/tmp/hv1_g60_adaptation_development_20261021/smoke_cpu_audit.log
SHA：a0a37e3784a7a0bc369b181094db5e978aac6b9787e261ae8c29db6fbefb0ce1。
退出码记录同目录smoke_cpu_audit_exit.json。原日志、代码和全部产物均保留。

## 最小修复建议（本轮未实施、未重跑）

将审计器第31–33行的局部统计字典stat重命名为epoch_stat，保留第62行文件状态检查，
并增加执行完整审计控制流的CPU Fixture，覆盖历史文件保护、正常通过与异常拒绝。
不涉及H/G/IRRA、奖励、训练循环或任何GPU算法变化。

已发布图库和Smoke协议绑定全部113项producer及原配置SHA。
**不能直接在冻结工作树覆盖审计文件并更新原配置，否则会破坏已完成产物的来源核验。**
后续应在独立CPU修复版本中建立明确受限的审计器来源兼容/只读引用：
保留9765a53树、原配置与全部SHA，只允许此审计器命名修复，并绑定修复差异和新来源。
只读复用本轮2048图库与4-step Smoke，无需再次G生成或GPU Smoke。
修复版CPU审计PASS后才具备启动正式训练的门禁条件；正式训练仍必须从原H10恢复，不继承Smoke更新。
本轮没有自动创建修复Attempt或继续训练。

## 真实Smoke与成本

4个预先固定的绝对Epoch11前缀样本：
cuhk:9074:0、cuhk:16997:0、cuhk:11702:0、cuhk:22830:0。
全部4次H参数发生变化，第3步具有非零梯度范数0.07009412446572012；
其他3步Sampled/Greedy奖励相同、当前梯度为0，但原Adam动量仍使参数变化，这符合原实现，不改算法。

采样R_sim、R_rank、总奖励和synergy均真实记录于EVIDENCE.json；含负值，不把总奖励解释为攻击成功。
Smoke结果仅验证恢复/梯度/真实奖励/Checkpoint，不计入正式Epoch11进度。
临时Checkpoint：
/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60/smoke_v1/checkpoints/visit_00020484/checkpoint.pt
SHA：e6bd68eeb2a0d09f328d910c3b8cf8e78a765f33b398a27cd6829ab8fe480922。
恢复/重载H/Adam/RNG exact；IRRA参数/buffers前后SHA exact，G无加载或前向。

- 图库执行：436.745秒（约7.28分钟）。
- 图库GPU peak allocated：797.614 MiB，reserved：866.000 MiB。
- Smoke GPU peak allocated：761.311 MiB，reserved：824.000 MiB。
- Smoke进程Peak RSS：2300.898 MiB。
- 首4步均值：0.140214秒/更新；后3步均值：0.058046秒/更新。
- 只按这4步外推，单Epoch约2–5分钟，50 Epoch纯Step约1.6–4小时；
  样本长度、共享GPU干扰、Checkpoint/Validation/最终CPU审计会增加耗时，不能当作可靠完成时限。
- 每Checkpoint实测8826249字节；
  400个周期/完整Epoch Checkpoint约3.288 GiB。
- 逐步日志按4步估计约0.887 GiB；
  不同K/文本长度会改变记录大小。
- 已完成图库实际占用3.404 GiB。
- CPU线程本轮固定2；历史协议未记录线程数，不据此承诺CUDA/CPU未来轨迹bitwise等价。

## 数据与历史保护

原2048 Train/128内部Validation，PID隔离通过；资格2037更新/11跳过不变。
图库最大L∞=0.0313725471496582≤8/255，FP32原图/Adv/effective_delta逐图Claim、SHA、读回通过。
G60 Train特征内容SHA：0b4bc2e6d6d3b5f7c9d34861e2371c984371d32324cbf2844c8e5be79978ad2b。
原Clean特征及上一阶段128 G60 Validation缓存只读引用；后续不得每Epoch重新生成G。

失败后另做标准库只读stat核验：151378个历史文件size/mtime/inode不变，
113项冻结producer SHA不变；H10、G60及旧H7 Selection保持。
这项故障现场保护检查**不替代失败的独立Smoke审计**，未将其改写成PASS。

## 未完成项目与停止边界

- 正式H11–H60：未启动。
- 新H20/30/40/50/60：不存在。
- 六节点新的Text-only/Joint指标、选优、曲线：尚未生成。
- 正式训练最终CPU审计：未执行。
- 未执行Test、RaSa/APTM、G10控制分支、新Loss或H61。

修复/重新审计应保留本次失败证据并遵守新的明确授权，不删除旧数据或放宽来源检查。
