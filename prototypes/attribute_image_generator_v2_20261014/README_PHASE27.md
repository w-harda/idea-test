# G-v2 Phase 2.7：从Epoch 1恢复至Epoch 3

本阶段仅授权Epoch 2–3，持久化总目标仍为10。复用原outputs/train_v2、config_phase26.json、全部2176条条件缓存以及原方案A；不修改任何既有训练源码或配置。

## 新增文件

- phase27_resume.py：默认CPU预检；唯一正式执行路径为run_epochs(resume=True, stop_after_epochs=2)。仅接受当前恰有epoch_0001的输出，拒绝重复运行、新建训练或已有更晚Epoch。
- phase27_analysis.py：只读核对Epoch 1–3样本顺序、缓存属性ID、K+1轮次、每轮日志/预算/有限性、Adam累计step以及固定128 Validation；按阶段、K和同K内属性轮次比较。
- test_phase27.py：12项新增CPU测试，包括连续3个合成Epoch与1+2恢复的G/Adam/RNG/日志逐值一致，Epoch1不覆盖，Epoch4不启动。
- 本说明及docs/stages/09_2_7_gv2_formal_epochs2_3_stability.md中文阶段报告。

既有epoch_training.py、epoch_checkpoint.py、serial_training.py、G/Loss/投影/IRRA适配器及缓存协议均不改。新入口只加入进程内观测外壳，不重写更新算法，不导入G-v1或integration，不重新计算评分/Top-K/JVP。

## 受控命令

工作目录为本目录，使用既有Python环境与cache目录：

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_conditions test_phase21 test_phase22 test_phase23 test_phase24 test_phase25 test_phase26 test_phase27 -v

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python phase27_resume.py

本阶段受控训练命令（完成后不可重复运行）：

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python -u phase27_resume.py --allow-epochs2-3

完成后的独立CPU只读审计：

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python phase27_analysis.py --audit

不要直接运行无受控上限的epoch_training.py训练CLI；本阶段不授权Epoch4–10。

## 恢复和观测

启动前验证Epoch1 Checkpoint固定SHA、关联样本/补充日志、原配置/来源/运行环境和完整缓存身份。实际加载后再次比较G、完整Adam和Python/NumPy/CPU/CUDA RNG；起点next_epoch=2、Adam8068。

每个Epoch严格2048 Train、6020属性+2048原图联合=8068 G/IRRA/backward/zero/step；128 Validation no_grad，G/Adam/RNG均保持。逐轮检查上一轮detach输入、立即更新后的G参数版本、上一轮JVP与stop-gradient参考、联合原图重启、预算与有限性。

新增证据仅位于outputs/phase27/，正式Checkpoint仍在原outputs/train_v2/checkpoints/epoch_0002和epoch_0003。逐Epoch独立封存round_audit、validation、resources日志并将SHA纳入Checkpoint，避免Epoch3继续写入Epoch2已绑定的日志。每次保存后用独立新G/Adam验证恢复，恢复保存的RNG后才继续；不增加训练forward/step。原Epoch1资产完整SHA保护。

仅支持Epoch边界恢复，不承诺Step级无损恢复或CUDA后续轨迹bitwise一致。预算投影饱和仍使用原真实clamp梯度，没有STE或超参数调整。训练样本均值仅作诊断，固定Validation也不能代替官方TBPS Gallery指标。

详细训练结果、Checkpoint SHA及风险见中文阶段报告；数据、权重、Checkpoint、缓存、outputs和日志不进入Git。
