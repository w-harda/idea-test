# H-v1 Phase 3：真实G-v2 / IRRA奖励链路Smoke

本阶段已完成一次RTX 4090真实10 Gallery + 10 Train Query Smoke。只验证真实图文协同奖励接线，未训练H，不提供mAP/ASR结果。H网络、Dynamic-K、Canonical/Span、动作约束、SCST及奖励公式均未修改。

## 新增接口与来源

- phase3_inputs.py：默认CPU来源预检；核验原2048 Train/128 Validation协议及文件SHA，从train确定性选5 PID×2图，10个Query均有2个正例和8个负例。仅使用Train PID绑定奖励，PID/图像从不进入H。
- real_feedback.py：复用G-v2 inference_once.load_generator/generate_once及irra_image_adapter.load_official。所有真实G/IRRA前向严格inference_mode/eval/requires_grad=False；计数观察实际G forward和IRRA encode_image/encode_text调用。检查每次参数/buffer version，结束时复核完整状态SHA。
- real_feature_cache.py：独立Real协议，绑定训练身份、图像SHA、权重SHA、相关源码、预处理、Tokenizer/BPE和特征SHA。逐图/逐query文本bundle保存紧凑CPU Tensor、metadata、意图和发布回执。不得读取Phase2 Mock或官方Test缓存。
- smoke_phase3.py：默认preflight；真实smoke必须显式 --allow-gpu-smoke。固定10+10且拒绝已有输出，失败不自动重试；SIGINT/SIGTERM在发布边界停止并留证。verify仅核验完成文件及缓存SHA，无新前向或指标重算。
- test_phase3.py：42项CPU fixture测试；config_phase3.json：冻结54项既有源码/配置SHA，运行协议另锁定本阶段4项执行源码、配置、Tokenizer/BPE与Checkpoint sidecar。

真实G-v2 Epoch10：
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt

SHA-256：f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。

官方IRRA权重：
/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth

SHA-256：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。

## 图文与奖励路径

原图经原G-v2 PIL RGB Resize(384,128)+ToTensor得到[0,1]像素；一次generate_once产生8/255预算内对抗图。Clean/Adv均由同一个官方IRRA实例，经原mean/std预处理、encode_image及L2归一化输出512维检索特征。不能把预算投影放在归一化空间。

Caption-only前端→TextExample→未训练H的sampled/greedy动作。H仍在CPU，这是第一阶段的设备契约；仅真实图文编码后端在GPU。本阶段H也被冻结，禁止optimizer/backward。

原文、sampled和greedy各一个文本，共30个文本向量，分10次IRRA encode_text调用。包含一个采样NOOP，仍在该query三文本bundle中编码；它不是额外训练Query。Tokenizer为官方Byte-BPE，原Unicode替换完整进入Tokenizer，不做额外转写/还原；截断或非法动作拒绝。

奖励复用未修改的BoundReward/RewardEvaluator：
R_sim = sim(I_adv,T)−sim(I_adv,T_adv)；
R_rank = Margin(T)−Margin(T_adv)，Margin=max同PID正例−max负例；
Reward=R_sim+R_rank，两个λ仍为1。
synergy_increment=adv_sim_drop−clean_sim_drop，只记录，不进入目标。
SCST只记录R_sample−R_greedy用于接口诊断，没有构造Loss反传或优化器。

## 本次验证结果

- CPU测试166项通过：Phase3新增42 + Phase2 56 + Phase1 68；未重复G-v2 Full审计。
- 真实10 Gallery、10 Query，5 Train PID；G10次、IRRA image20次、IRRA text10次/30文本，H20次CPU前向。
- 最大L∞=0.0313725471496582≤8/255，图像像素范围、特征形状/范数/有限性通过。
- 全部Query预测K=2。sampled共11次编辑/1个NOOP；greedy共17次编辑/0个NOOP。所有真实编辑改变Tokenizer输入且无截断；不需要人工单字符fallback。
- G/IRRA参数与buffers SHA精确不变；H参数SHA不变；无梯度/optimizer/JVP/训练。
- GPU peak allocated805.740234375MiB，reserved886MiB；Smoke阶段11.01696678251028秒，含7.078916180878878秒模型加载及CPU H/IO，不含8.748803734779358秒CPU preflight。
- 独立CPU证据verify通过；10图像单位+10文本bundle，81个缓存文件SHA受summary锁定；无pending、缺回执、NaN、失败或中断。
- 129项历史保护文件SHA保持。GPU进程已退出，运行后nvidia-smi无计算PID，显存回到40MiB。

奖励正负都有，synergy均值为负。H尚未训练且图库只有10张，这些数值不能证明方法有效、跨模态协同收益或检索攻击性能。详细逐Query结果见中文阶段报告与服务器summary。

## CPU命令

测试临时目录必须位于项目允许写入的范围，首次执行前运行 mkdir -p /home/lzf/ldx/tmp。

~~~bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_phase3 test_phase2 test_policy test_scst test_tokenizer_real -v
~~~

~~~bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lzf/ldx/envs/tbps-image/bin/python smoke_phase3.py --action preflight
~~~

本阶段已执行的GPU命令仅供记录，不自动重复执行。原输出存在时会拒绝运行，不能换新目录绕过本阶段预算：

~~~bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch \
/home/lzf/ldx/envs/tbps-image/bin/python -u smoke_phase3.py --action smoke --allow-gpu-smoke \
--output outputs/phase3/smoke10
~~~

完成证据的CPU只读核验：

~~~bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lzf/ldx/envs/tbps-image/bin/python smoke_phase3.py --action verify --output outputs/phase3/smoke10
~~~

## 证据与停止/恢复边界

服务器证据：本目录outputs/phase3/gpu_smoke.log、cpu_h_final.log、preflight.log、cpu_evidence_verify.log，以及smoke10/{protocol,summary,completed}.json和attempts/attempt_0001/execution.json。

查看进度：tail -n 40 outputs/phase3/gpu_smoke.log；确认GPU退出：nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader。

如果运行中需停止，可Ctrl-C或向自己拥有的进程发送SIGTERM，程序在下一个原子发布边界停止。失败/pending/缺回执一律留证并拒绝自动重复G；本阶段没有GPU自动恢复入口，需审查原因和获得后续授权。不得删除回执或覆盖旧协议来继续。完成后只用verify，不重跑。

## 下一阶段最小待办（未执行）

1. 在独立正式输出中准备/审计2048 Train的Real Clean/Adv Gallery缓存，保持每个唯一图像一次G；严格排除128 Validation/Test。
2. 添加真实SCST调度，sample/greedy奖励来自冻结后端，只有H的sequence log-prob保留计算图；解冻H参数，optimizer严格只包含H。可先保留H CPU、奖励GPU，避免改网络或设备契约。
3. 添加确定性样本顺序、动作/奖励/advantage/梯度日志、H与Adam/RNG事务Checkpoint及恢复、来源与日志SHA检查。
4. 建议首轮候选配置：Adam lr=1e-4、betas=(0.9,0.999)、单query batch=1、seed42，λ_sim=λ_rank=1、Dynamic-K与字符预算不变。先授权≤32次受控SCST更新核验梯度/恢复，再决定正式Epoch数量；这些建议尚未冻结或执行，不增加奖励正则/额外算法。
5. 后续正式训练、全量Real缓存、H设备调整或评测都需要新授权。本阶段完成后停止，等待独立审查。
