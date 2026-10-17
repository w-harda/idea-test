# Stage4C-2报告：真实Test物化通过，IRRA启动顺序FAIL

## 结论与开发范围

**最终FAIL，端到端IRRA等价未验证。** 唯一正式Attempt已经停止，未重试。阶段A的两张G10像素物化、阶段B的两条历史H7 Unicode物化通过；阶段C在IRRA恢复之前被CPU隔离闸门拒绝。不能将局部通过写成Stage4C-2 PASS。

基准4d3bc025a7da593368ebd22cc05db1ebe787b446；分支codex/tbps-framework-stage4c2；树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4c2-20261017。新增Runner、来源/比较支持、固定配置、CPU测试及本轮统一文档；没有修改旧prototypes、G/H、IRRAAdapter、历史Loader、C1存储、模型、Checkpoint、条件缓存或任何旧实验结果。

## 严格CPU与资源前检

- G/H真实工厂全部入口、传递导入与75项冻结runtime来源通过；禁止torch.load/jit.load、神经构造/前向、CUDA和导数，命名空间恢复。
- 官方IRRA真实build/clip/Tokenizer导入链逐执行文件SHA核验；没有用两次Adapter调用冒充独立旧路径。
- C1完整计划/来源回执与B2 METRICS_REPLAY_VERIFIED回执校验；3074 Gallery/6156 Query顺序一致。
- 固定Gallery index0/15与固定Query编辑/NOOP，在GPU运行前锁定，未按攻击效果选择。
- 两条Query原顺序index0/31；原Caption、Span、动作与来源一致。
- GPU资源：RTX4090，无计算进程，free24051MiB/used40MiB/utilization0%。
- FP32/math/no_tf32、seed42、误差1e-6、每进程RSS≤2147483648 bytes、Attempt总时长≤300秒；无临时放宽。
- 初版CPU预检读取B2报告字段错误，以实际的status=PASS和metrics_replay_status=METRICS_REPLAY_VERIFIED修正后通过；没有改变B2证据。

## 两张真实RGB及delta

全部shape=[1,3,384,128]，torch.float32，像素[0,1]，effective_delta严格等于Adv−Original。生成时与保存后读回的三Tensor逐值bitwise exact、max_abs_error=0。两张L∞均为**0.0313725471496582≤8/255**。

| Gallery | 原始像素内容SHA | 对抗像素内容SHA | effective_delta内容SHA |
| --- | --- | --- | --- |
| cuhk:37132:0 | 90218e2189d60abd931eaaf0e5b8a49c24777424e37ee7d3901b74af63129045 | 54c594950652698fb62925eee8e50bf87c3505f55b66850bdc332b2572cf4be6 | 23c315645350476dbbd42978b2c8f3904f7d124fa12ccb8a5a9e3086aa160650 |
| cuhk:37147:0 | 64e176c41cf3c61ed672b7a488ec7390183900cae3238e6b7b113a029382d713 | bad51abb2f0e46affbb7ec745f9f436b1f63a9fcb61a976c9c439a6caffcaabe | 7f6db236fdd9dba938e5d9a82a746e53a07d65b0730730bd808d92570a764604 |

序列化pixels.pt文件SHA分别为366adca54fbd5336b1defb66e542a381cd61709fbc892e7b494d876609373fd1、9a15eeb4c0dbbeb1b30e5b174951295886096db70b6aac5d46eb362ae5d18aa4，不能与像素内容SHA混用。

原始G输出raw delta单独保存：内容SHA分别为b124f95a401a76ee2b9d0b73ac2825ea4fdccc92cf86d3eea1e50f222afcb3ab、48a723c7ef4222edf068250559bd0f0572eb3cf4846a2f65aaa3b5d3c455430c；raw delta L∞均0.0313725471496582。effective_delta会受投影/像素裁剪影响，未将两种delta混同。

## 历史Unicode与NOOP

cuhk:37132:0：eligible=true，budget_k=2，实际2次编辑，完整action、来源metadata及读回exact。

- 原文：A man wearing a blue and white stripe tank top, a pair of green pants and a pair of pink shoes.
- 攻击文：A mаn wearing a blue and white stripe tank tоp, a pair of green pants and a pair of pink shoes.
- position3：man，原Span[2,5]，a U+0061→а U+0430（Cyrillic），gender/male。
- position44：tank top，原Span[38,46]，o U+006F→о U+043E（Cyrillic），garment_type/tank_top。
- UTF-8原文SHA24af8c725b92c5e48ff3e77927e7f806d165c0a83696bb33371ff31bd95865b9；攻击文SHA76684487b0add18b0e5d5dfbc5cb5a92d08450626cee22729017b6dc389b134f。

cuhk:37147:1：固定NOOP，eligible=false，budget_k=0，edits=[]，reason=no_legal_token_changing_edit。原文/攻击文字符串exact，UTF-8 SHA均ae57e322322decc31f9491e17fa36f27837fb5b91b6156d1fdb9bcda9747f400。完整原文和动作保存在stage-text/report.json与对应物化metadata，未删NOOP。

H权重恢复及策略前向均0；没有重新选属性/Unicode字符，没有目标模型反馈。

## IRRA FAIL定位及未完成比较

失败原始错误：`ArtifactError: isolated CPU process required`。

真实调用链：

1. smoke.py::child先调用torch.cuda.reset_peak_memory_stats()；该调用初始化CUDA。
2. stage_irra→selected_features→native_action_validator。
3. 原runtime_closure.py::cpu_only要求torch.cuda.is_initialized()==false，正确拒绝。
4. IRRAAdapter.from_source尚未执行，IRRA_load_attempts未开始、IRRA_loads=0、image/text forward=0。
5. 父Runner保存stage-irra FAIL与最终FAIL，不调用后续阶段或自动重试。

源码位置：smoke.py:160提前初始化CUDA，stage_irra在smoke.py:90进入CPU读取。

这是本轮新增Runner的启动顺序错误，不是IRRA数值、预算、Checkpoint或攻击算法异常。严格CPU导入门禁未覆盖GPU子进程“初始化CUDA后再次进入CPU规则读取器”的时序，这是CPU测试覆盖缺口；新增合成复现确认CPU门禁确实拒绝、模型加载不发生。失败Runner源码保持本次Attempt所锁定的字节，未事后修复来改变证据。

**原/Adv图像特征、原/Adv文本特征误差和四组2×2 Score均未产生，状态PENDING。** 不填0、不采用旧指标替代，不宣称新旧特征等价。阶段C未载入任何历史特征Tensor即被拒绝。

建议后续最小修复（未实施、需新授权）：在IRRA独立子进程中，先完成历史特征/动作与物化输入的CPU只读检查，退出这些上下文后才初始化CUDA/峰值统计/加载IRRA；为真实启动顺序添加CPU Mock测试。保持G/H算法、样本、权重、精度与阈值不变，保留本次FAIL及已生成单元，另行批准新Attempt，不恢复或覆盖materialized_smoke_v1。官方IRRA检查点含optimizer等大Storage，加载期RSS≤2GiB尚未实测；修复前需做CPU-only资源分析，不能自行提高门限。

## 实际计数、冻结与资源

| 阶段 | 权重恢复 | G图像前向 | H策略前向 | IRRA图像/文本前向 | 耗时秒 | peak RSS bytes |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| G | G10一次 | 2 | 0 | 0/0 | 14.373739369213581 | 1015422976 |
| 历史文本 | 无 | 0 | 0 | 0/0 | 12.154895596206188 | 611213312 |
| IRRA | 无，加载前FAIL | 0 | 0 | 0/0 | 6.954171933233738 | 629682176 |

Attempt墙钟35.17172169685364秒；父进程peak RSS741126144 bytes。G GPU peak allocated42.3857421875MiB/reserved52MiB；IRRA会话仅CUDA上下文已初始化，allocated/reserved均0；不能当作IRRA模型显存测量。

G模型54个参数Tensor，eval、requires_grad=false、所有grad=None；前后参数摘要e247df82150c4b6757e4eaffef421226fa6ed3ed8129ac5e51c289ea777bbb33 exact；无buffers。H与IRRA未恢复，不声称测得其GPU运行前后参数摘要；原Checkpoint字节SHA只读校验保持。

G10 SHAf59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e；H7 SHA487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df；IRRA SHA9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。Backward、Optimizer、JVP全部0。GPU子进程已退出，无自动后台运行。

## 证据与保护

根目录：/home/lzf/ldx/outputs/tbps-framework-stage4c2/materialized_smoke_v1/。

- attempt/manifest.json SHAff0c62a387da4f0f9de11ceedf9c6bfae0ad10ca7eee28a09326cc1e4b762c1a，内含严格CPU前检和固定样本。
- stage-g/manifest.json SHAe4708ed933f01c7b0681ee2da96944b4092167e5dd563807725cfa3131526973。
- stage-text/manifest.json SHAce25b5473d978e34cdc270295ecd8a88bb74a74f592b498f50aaaa07c9609ece。
- stage-irra/manifest.json SHA3857d0e25af7e0ff667481954a74634600fd476e40f382768960a0dbe0580bfb。
- result/manifest.json SHAc1922bdf32935623dc9e85b1aeffe8a817cef2ab42b3ba767f62eda90e45155b。
- g.log/text.log/irra.log保留；原始启动日志/home/lzf/ldx/tmp/stage4c2_gpu_attempt_v1.log。
- artifacts/含两图/两文、原子回执及G前置生成Claim/生成证明；raw-delta-0/1为原G输出补充证据，不冒充集合seal。
- 后置CPU保护证据：/home/lzf/ldx/outputs/tbps-framework-stage4c2/post_cpu_protection_v1/result/，Manifest SHAd4d5ceb47270800589b6321036b7e92617f8e1d11ac57d05b3f06f2fb2a5f4e2。
- 最终CPU验收：/home/lzf/ldx/outputs/tbps-framework-stage4c2/cpu_verified_v1/result/，Manifest SHAc39be04bb65e9c25417e2145f3645363facfe34fee211c390f85e4e87ce84a80。

最终501框架+36历史基础=537 CPU tests通过（23项新增）。后置7991项历史文件及3472个Joint缓存SHA/mtime不变，C1/B2完成来源和本次所有生成证明有效，无pending；六个受保护工作树HEAD/clean保持。既有Stage3B FAIL、Stage3B2 PASS、B2 METRICS_REPLAY_VERIFIED、C1 CPU来源及所有旧资产不覆盖。本次没有真实Full特征聚合、排名或正式指标重算。

## 全量成本与后续安全边界

两图的真实结果证明FP32三Tensor每图589824×3 bytes；3074张纯Tensor总量仍5.0657958984375GiB，建议独立磁盘8–10GiB并额外预留文本/动作/文件系统空间。本轮G会话peak allocated42.39MiB、CPU约968.38MiB，可作为小样本量级；不包含后续IRRA真实加载成本。

14.37秒包含完整来源核验、模型恢复、两图解码、原子存储和读回，不是纯G forward时长，不能线性外推成Full基准。此前10–60分钟仍仅工程估计，尚无精确全量耗时数据。

全量物化前至少需要：单独批准CPU-only时序修复、审查后批准新的两样本GPU验证、确认特征/Score阈值通过、核验实际IRRA加载资源；之后再独立批准G-only流式3074张物化及历史6156文本复用。全量入口应保持逐单元Claim/不可变证明、完成项严格校验跳过、未提交前向不自动重做；目标评分与攻击产物分离，不向策略反馈。

本轮停止，不进入全量物化、Stage5、RaSa/APTM或训练。
