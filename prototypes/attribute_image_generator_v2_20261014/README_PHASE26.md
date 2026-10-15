# G-v2 Phase 2.6：正式Epoch 1

2048 Train、6020属性更新+2048原图联合更新=8068次step完成；128 Validation无更新。240历史+19新增=259项CPU tests通过，正式Checkpoint的G/Adam/RNG恢复exact。总目标10，本次仅Epoch1，不自动继续Epoch2。

## 文件与受控入口

- phase26_train.py：固定复用run_epochs(stop_after_epochs=1)，默认仅CPU preflight；--allow-epoch1才启动新正式运行，已有输出拒绝。
- phase26_support.py：状态摘要和逐轮/K/窗口统计，无训练算法。
- config_phase26.json：固定width32/4-block、官方冻结IRRA/fp32、Adam2e-4/(.5,.999)、三个lambda1、seed42、8/255、总Epoch10；缓存与入口源码SHA绑定。
- epoch_checkpoint.py：本次唯一旧源码修改，恢复前核验样本日志SHA、内容/顺序/K+1/进度及补充日志；缺失/损坏拒绝，不先加载状态。
- test_phase26.py：19项新增CPU测试。

G-v1、IRRA、G架构、属性/选择/JVP、Loss与方案A保持。只读取Phase2.5完整缓存，没有新评分/JVP。

## 命令

使用/home/lzf/ldx/envs/tbps-image/bin/python与既有cache环境变量，工作目录本目录：

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_conditions test_phase21 test_phase22 test_phase23 test_phase24 test_phase25 test_phase26 -v
    python phase26_train.py
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python phase26_train.py --allow-epoch1

正式Epoch1已经完成，重复--allow-epoch1会拒绝现有outputs/train_v2，不能借此继续Epoch2。

## 关键结果

- Train G/IRRA/backward/zero/step各8068；属性6020、联合2048。
- 额外训练前128 Validation进行clean/初始G无梯度诊断，Epoch后128 Validation单步诊断，均不更新。
- Loss均值Global .40836280 / Local .0000709728 / Retain .0000660307 / Total .40849980。
- G梯度L2 .03504761–7.88458928，零梯度/无参数变化轮次均0；IRRA全参数/buffers exact。
- 累计L∞最大8/255；预算饱和率均值72.8335%/max99.1143%，严格裁剪均值47.3079%/max99.3510%。
- Validation Global：clean .49905692，初始G .49691020，Epoch1 .39919289；Local：.000510984/.000518634/.000141642。
- 联合前/后256样本Global .38573436/.38880811，没有单调下降；不同样本窗口不作同图收敛结论。
- 受控段699.414秒（含缓存/诊断/保存恢复），峰值allocated1033.164MiB/reserved1196MiB，无OOM或异常。

## Checkpoint

    outputs/train_v2/checkpoints/epoch_0001/checkpoint.pt

SHA：876547ffeb827cbb4691bbf5b5d473da495590d15df2adba12c9e1415e52a27a。
样本日志SHA：c37611250443677a1d6fb1bd0238059ca72ddac4fb167a774524b88a348fa2b8。
配置SHA：7e87bd272abfee440afc13109d439507a34b140506a19a132a8f740afe093bc4。

GPU新建G/Adam恢复后参数、Adam/RNG exact，所有Adam step8068，next_epoch2但未执行。CPU独立复核8068条记录与缓存顺序/ID/角色/K+1一致；正式日志副本篡改和缺失被拒绝。

恢复必须保留sample_log_path及补充日志，不能只复制权重文件。仅Epoch边界，非Step级恢复，不保证CUDA后续轨迹bitwise重复。未来继续须另获授权并保持原配置/来源/环境，可使用原epoch_training.py --resume --config config_phase26.json；当前未执行。

详细统计、各K结果和边界见[中文报告](../../docs/stages/09_2_6_gv2_formal_epoch1_validation.md)。证据在outputs/train_v2/和outputs/phase26/，不提交Git。GPU已结束，停止等待独立审查，不进行Gallery或下一阶段。
