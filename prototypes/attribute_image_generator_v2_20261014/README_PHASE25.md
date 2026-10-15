# G-v2 Phase 2.5：完整条件缓存准备与审计

原2048 train +128内部validation的2176条缓存已完成并通过独立审计；K=0..7、JVP总数6385。复用原Phase 1五条/15个独立JVP，本阶段新增6370个，未重新评分或计算Top-K。

7个已锁producer文件、G/Loss/Retain/串行调度与训练协议均未修改。G构造/forward、optimizer.step、正式Epoch训练与Gallery评测均0。停止等待独立审查，不进入Phase2.6。

## 新增接口

- phase25_prepare.py：调用原prepare_full.main(["--allow-full-preparation"])，记录5秒资源采样、耗时、异常与grad整数计数；设置禁止G/optimizer/Epoch训练的断言。不改变producer或JVP数学。
- phase25_audit.py：CPU preflight及独立逐条监督/身份/SHA审计，输出K/FD/聚合统计；不调用模型/JVP计算。
- test_phase25.py：21项新增CPU审计和观测器回归测试。

详细K分布、SHA、异常/恢复与训练前待办见[中文阶段报告](../../docs/stages/09_2_5_gv2_full_condition_cache_audit.md)。

## 已执行命令

使用 /home/lzf/ldx/envs/tbps-image/bin/python，工作目录为本目录。GPU环境变量沿既有 /home/lzf/ldx/cache/：

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python phase25_audit.py --preflight --output outputs/phase25/preflight
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python phase25_prepare.py --allow-full-preparation
    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python phase25_audit.py --output outputs/phase25/audit

完整Cache固定在 outputs/conditions_full/。准备入口支持原子样本恢复，完成态只核验；审计输出不覆盖，重审使用audit_new等新目录。

## 结果与异常

- Train/validation PID交集0，与官方Test PID交集0；完整身份/源图SHA、z_t/S_raw/权重/顺序exact。
- 6385个zero_exact检查通过，最大FD相对误差0.3131863%，原门槛2%。
- 聚合重建误差0，原G-v1聚合对照最大差9.313225746e-10，原容差内。
- 6个K0协议正确，无失败JVP/零向量替代/遗漏，pending=0。
- 158条样本有170个额外重复JVP方向；对应冻结w和单位projected_context完全相同。保持原实例和监督，未去重或改方法。

首轮观测器MagicMock保留张量参数，79条后OOM，安全退出。只修新增外壳为不保留参数的CallCounter，66项CPU tests通过；79条SHA/inode/mtime核验后原样恢复。首轮45.5748秒，恢复406.1039秒；恢复峰值allocated1225.1001MiB/reserved1312MiB，无新异常、IRRA参数/buffers exact，其他进程未动。

两轮累计新增6370个JVP/12740次内部grad，原五条复用15个。Cache summary中的6130/12260/3为恢复这轮的this_run计数，累计值由两轮记录与独立审计核对。

完成态再次禁止CUDA/模型/前端/backward/JVP重入成功，4356个缓存/协议文件SHA/inode/mtime均保持，GPU/JVP为0。

## CPU测试与证据

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_phase25 test_conditions -v

21新增+45历史缓存=66项CPU tests通过，不执行G/optimizer/Epoch。

Manifest SHA：506caf30b8f82b8a29fc35e06df7bb04a5b2bcddc58a1f159b524bb107e7511e。
Protocol SHA：9ad4903a6de6dc221df1a15af48d9e63e6332d7fcd3a9a62ecd3f11a2ed91c9d。

outputs/phase25/保存全部原始证据、失败wrapper快照、两次资源采样、逐条审计、恢复快照与只读重入报告。数据/cache/权重/outputs/日志不提交Git。
