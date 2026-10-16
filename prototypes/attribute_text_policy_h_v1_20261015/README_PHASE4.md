# H-v1 Phase 4：真实SCST首轮32 Step与Checkpoint

本阶段使用Phase 3已核验的10张Train Gallery/10条Train Query，实际更新H参数32次。奖励来自GPU官方IRRA，图库特征只读复用；G与IRRA图像前向为0。不是2048 Train正式训练，不据此选择正式H模型。

## 固定协议与训练链路

- 基准Commit：08d8775749cc3cd3e86cc887c26af660781bb717；所有新增文件在H独立目录。
- H仍为原Byte Transformer/三个Head，width64、2层、4head、max_bytes1024，CPU正常autograd。
- Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0；batch1、seed42，总更新上限32。
- 原SCST、Dynamic-K、STOP/NOOP、每属性/共享词最多一次编辑不变。R=R_sim+R_rank，权重1/1；synergy_increment只诊断。
- 10个Query按独立seed42的randperm循环打乱，完整32项顺序写入协议及Checkpoint；采样动作另用CPU Generator seed42。
- H只收到Caption-only TextExample；PID及正确配对仅在train奖励绑定层，Validation/Test不得参与反馈。
- 每Step由当前H重新生成sampled/greedy，原文与两变体每次都由GPU IRRA重新编码（3向量、1次调用），不读取Phase3未训练H的文本结果。
- Phase3 Gallery原图/样本身份、metadata/receipt/特征SHA、模型及来源SHA严格核验后只读加载；没有实例化缓存写入路径，没有生成新图像。
- G仅在CPU加载以验证冻结参数/buffers；IRRA在CUDA冻结/eval/inference。真实调用点限制G/图像前向为0，文本向量上限96。
- 不加入entropy、reward shaping、梯度裁剪或其他算法。只对H的序列log-prob反传。

## 新增模块

phase4_inputs.py负责只读来源核验与独立协议；
phase4_real_reward.py复用Phase3 InferenceRuntime和原BoundReward，提供每Step新文本奖励；
phase4_training.py复用原scst_loss，实现H-only optimizer与标量日志；
phase4_checkpoint.py保存不可变Step版本、SHA、RNG并支持已发布边界恢复；
train_phase4.py提供默认CPU入口、显式训练授权、锁和停止边界；
phase4_audit.py独立CPU核验完成根日志SHA、H/Adam/CPU RNG、缓存文本及奖励；
test_phase4.py、config_phase4.json提供测试与固定配置。

## Checkpoint契约

每个已发布step目录包含intent.json、checkpoint.pt、sample.json、checkpoint.json；Step1–32另有紧凑text_features.pt（当步真实3×512向量）。

checkpoint.pt保存H、完整Adam、当前/下一Step、配置/协议、32项样本顺序与cursor、Python/NumPy/PyTorch CPU/CUDA RNG及动作采样Generator状态。sidecar锁定Checkpoint、关联日志、文本特征和intent SHA；完成summary另锁定全部33条step日志，completed锁定summary。独立audit核验该完成根的日志清单。

先在独占pending目录记录意图并fsync，再执行更新；完整保存/核验后原子rename发布。版本不覆盖，无效最高版本不静默回退。SIGINT/SIGTERM在发布边界停止；clean已发布边界可恢复。

存在未发布pending意图时拒绝自动恢复，因为不能证明optimizer是否已执行；不得删除pending/改SHA后继续。Step32已发布但收尾缺失也不能再更新，需另行只读审查收尾。恢复承诺限于已发布完整Step边界，不是任意指令中断无损恢复。

本次Step16独立新H/Adam/RNG恢复后继续17–32；Step32再次精确恢复，没有额外forward/step。CPU测试另验证连续6步与3+恢复+3步的H/Adam/RNG/动作日志逐值一致，并模拟子进程退出。没有运行第二条真实32-step对照，不声称跨CUDA环境未来轨迹bitwise一致。

## 本次实测摘要

- GPU前201项CPU测试通过；新增只读审计3项后最终204项通过（H Phase1/2/3=166，Phase4=38）。
- 实际32次H optimizer.step；10条Train Query均使用，2条出现4次、其余3次。全部K=2。
- G forward0、IRRA image forward0；IRRA text32次/96向量，均为当前策略输出。
- sampled编辑44次，完整NOOP 3/32；greedy编辑48次，完整NOOP 0/32。
- H梯度范数0–3.5686009595429957。Step2/7/10/19/21/29的Advantage恰为0，因此当步梯度0；Adam历史动量仍使参数变化，32步无参数完全不变。
- 三Head和Byte Embedding/Encoder均收到过非零梯度，G/IRRA无梯度且参数/buffers完整SHA前后不变。
- allocated峰值761.3720703125MiB，reserved824MiB；受控运行12.931669186800718秒（不含CPU Preflight，包含加载/CPU H/IO/恢复验证）。
- Step16及32恢复精确；完成CPU审计通过，缓存重建32条训练奖励与日志最大误差0，无新前向或更新。
- 无NaN/OOM/预算/来源/Checkpoint异常、无pending、无重试；228份历史文件SHA/大小/mtime保持。GPU已退出。
- sampled平均R_sim=-0.001434975303709507、R_rank=0.025704778963699937、Total=0.02426980365999043、synergy=-0.01066343393176794；greedy对应-0.0040666162967681885、0.01728112157434225、0.013214505277574062、-0.009901504963636398。

这些是小图库训练诊断。负synergy不构成修改奖励的理由，正Total也不构成方法成功证据。最终目标仍是独立Validation选择Checkpoint后，在正式Full上验证mAP_Joint < mAP_ImageOnly，并比较R@1/5/10。

## CPU命令

临时文件限定于/home/lzf/ldx/tmp，首次需要mkdir -p /home/lzf/ldx/tmp。

~~~bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_phase4 test_phase3 test_phase2 test_policy test_scst test_tokenizer_real -v
~~~

~~~bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lzf/ldx/envs/tbps-image/bin/python train_phase4.py --action preflight
~~~

只读训练资产的完整CPU审计（默认打印，不创建/覆盖报告）：

~~~bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lzf/ldx/envs/tbps-image/bin/python phase4_audit.py --input outputs/phase4/scst32
~~~

## 已执行GPU命令与恢复

以下是本次已执行命令，仅留存记录，不再次运行或换目录重复32步：

~~~bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch \
/home/lzf/ldx/envs/tbps-image/bin/python -u train_phase4.py --action run --allow-scst32 \
--stop-after-updates 32 --output outputs/phase4/scst32
~~~

若后续明确授权恢复同一未完成run，应在同一环境、同一输出上增加 --resume；持久化总上限仍32，--stop-after-updates仅限制本次新增更新次数。协议/配置/来源不符或pending时拒绝恢复，不能擅自放宽。已完成的--resume分支只CPU核验，不再加载模型或更新；本阶段没有再次运行GPU恢复命令。

进度：tail -n 40 outputs/phase4/gpu_scst32.log。
停止：Ctrl-C或SIGTERM，等待当前Step发布；不要在未提交状态删回执。
确认退出：nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader。

## 输出及后续边界

最终受控Checkpoint：
outputs/phase4/scst32/checkpoints/step_0032/checkpoint.pt
SHA-256：6476603a2ab12f0152fb71cab46bb7f0f124e88c4864a755ce39831f705217ef。

主要证据：outputs/phase4/gpu_scst32.log、cpu_h_verified.log、cpu_h_audit_final.log、cpu_final_restore.{json,log}、audit_verified.{json,log}；scst32下protocol/summary/completed、33个Step版本及attempts/attempt_0001的execution和两次恢复记录。

正式2048 Train前还需：全量Caption资格检查（属性覆盖/合法编辑/字节长度/截断/空候选协议）、独立2048 Real缓存与恢复入口、正式训练配置和初始化策略、长程调度及未提交Step处理契约、独立Validation评测/Checkpoint选择。正式总Epoch、是否从本轮32步承接及全量准备均待用户确认。

本阶段未准备全量图库、未做正式多Epoch训练、未使用Validation/Test奖励、未做100+100/Full检索评测。完成后停止，等待独立审查。
