# CUDA RNG验证与A1/B两轮Smoke执行协议

本轮用户明确授权：独立验证CUDA RNG，通过全部门禁后立即后台启动唯一smoke_v2，并在生成结束后自动执行原独立CPU审计。不得启动正式训练、Validation/Test/Full。

## 冻结执行来源

- 生产工作树：/home/lzf/ldx/projects/idea-test-tbps-gh-probe-shared-20261023。
- 当前文档HEAD：aafc7715ddafa78f9a30410b6a6ed8852e81780a；数值执行源码a7597f9c964dda0c8c48973e4678f0b9d3fa095a。
- 既有probe_v2 Manifest：d6e0b960e20a1ba5128f9e0b4edf497e7ea4b098fbf5cd3125799d2ad1136764。
- 配置：gh_bidirectional_backend.probe_v2.shared.server.json，SHA77ae41af56189ece991ee7cf21fd0ea0f3be5645202bd1cb7cf7e0b2ae735649。
- source_key：06b5c6a15061e635e49a9e13deb14022956fdbc272ad0ed378676e57ca5f826b。
- 本分支仅增加operations脚本、CPU测试及文档，不修改冻结src/configs或旧probe回执。运行始终PYTHONPATH指向上述生产工作树。

## 独立RNG验证

operations/gh_cuda_rng_validation.py复用prepare_cpu_gate恢复G60/H10完整CPU父状态，重新验证来源、probe、历史保护与原Smoke资源门槛后，在独立cuda_rng_v1发布Claim，再初始化CUDA。
只执行少量Python/NumPy/CPU/CUDA/sampler随机数调用；实际使用原restore_rng、capture_rng和NativeBackend.enter_phase_rng/leave_phase_rng。
对两个父状态执行exact恢复及随机序列重放，G→H→G和H→G→H阶段切换，状态持久化/读取/再次恢复；比较G/H/双Adam原内容不变。
全程禁止神经前向、autograd和Optimizer.step，不加载IRRA/reader。源脚本SHA纳入独立RNG产物来源，不改生产producer身份。
任一真实CUDA异常保留独立FAIL，不重试，不启动Smoke。

RNG输出：/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/cuda_rng_v1/。
控制与日志：/home/lzf/ldx/tmp/gh_rng_smoke_v2_20261023/。

## A1/B后台单次执行

operations/run_gh_rng_smoke.sh使用nohup+setsid启动，flock、O_EXCL启动标记及原native_cli唯一Attempt门禁共同保护。
重新CPU Preflight，核验CUDA_RNG_PASS、同来源probe回执和smoke_v2未消费；即时检查空闲GPU≥10240MiB、RAM≥32GiB、磁盘≥25GiB、inode≥80000。
利用率仅记录，无轮询、不抢占其他进程。

严格调用原native_cli --action smoke --allow-gpu-smoke，原A1-r1/A1-r2/B-r1/B-r2及固定8条Train样本顺序不变。
原G K+1更新108次；H访问32次、更新≤28次；四次完整2048图库刷新，共8192图；B新JVP≤38。
G(I)/H(T)、原K/Loss/Reward、FP32、8/255保持。
本工具不实现或替换任何训练、采样、JVP或Loss。

生成成功后单独CPU进程调用原--action audit-smoke，并检查历史保护与反馈证据。
控制流成功、CPU审计成功和双向数值反馈是否成立分别记录，不把Smoke成功写成检索性能提升。
native_cli原生日志GH_BANK_PROGRESS显示臂、轮次和图库逐图进度；逐样本G/H更新没有额外实时打印，通过已提交Round ledger查看，不为日志修改冻结训练器。
任一异常立即停止，不重试/恢复/删除Claim或pending。不再创建其他Smoke。
