# AnyAttack官方权重的TBPS免训练迁移

## 研究证据与范围

用户授权直接使用官方训练好的Decoder，不进行TBPS训练或Adversarial Noise Fine-tuning。本次固定CUHK-PEDES × IRRA、RaSa、APTM；Smoke10 Gallery/10原Query，全部生成/原生评分/独立CPU审计通过后，自动Full3074 Gallery/6156原Query。扩展三数据集九组需各自原生权重与独立协议，不能把CUHK模型误作其他数据集模型。

AnyAttack官方master固定91b209120c61261a51a45ce217dae7f1daac081d，README明确建议未知任务使用coco_cos.pt。作者README链接的Jiaming94/anyattack当前重定向jiamingzz/anyattack；固定HF revision4c7172a1ada67ab01e6606ddda4ccacb1362ea4a，checkpoint路径为checkpoints/coco_cos.pt。335270958字节，SHA256 bed1f25d46b4eaef0c3d4eb39c3853b2241991cc048b3b2fff21b7b8c38e0070。CLIP ViT-B/32固定OpenAI官方SHA40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af。

AnyAttack论文区分LAION大规模预训练与下游20轮微调。R²TUA论文表1在不同模型/数据集得到不同分数，不能证明使用了不同Decoder；其第8页表5仍记AnyAttack训练16h、攻击10.8ms。3090环境不能证明没有训练。已检索公开论文、GitHub与关联目录，尚未找到作者披露的AnyAttack适配实现或Decoder身份，精确适配方式记UNKNOWN。本次结果称“官方COCO Decoder零样本TBPS迁移”，不称R²TUA表1精确复现。

## 数值与信息协议

保留官方models/model.py和CLIP实现逐字节内容。原仓库BPE文件为0字节，在独立runtime overlay补入已验证的同一OpenAI BPE词表；原始解压源码不改，overlay每份Python与原Git blob精确对应。使用原CLIPEncoder.encode_img的**未额外L2归一化**特征及Decoder(embed_dim=512)，不换成IRRA特征、不加tanh。DataParallel前缀只允许全有或全无，strict加载。

AnyAttack需要目标参考图像和扰动承载图像。本协议使用全Test Gallery预先固定的异PID一一映射：seed42打乱PID组顺序，按最大组长度循环移位，逐项验证不同PID。PID仅用于映射，不进入CLIP/Decoder；无caption、检索分数、梯度或Victim反馈。必须披露此Test PID访问，与Idea的G(I)信息条件不同。Smoke使用同一Full映射的前10项，不按Smoke分数选参考或权重。最终只计算原PID的常规检索指标，未报告targeted命中率。

参考图RGB经PIL BILINEAR到224×224；承载图经同一PIL流程到384×128。官方Decoder224噪声先按原16/255 FP32 clamp，再CPU bilinear(align_corners=False)缩放到384×128、再次限定同一浮点端点，再加RGB并clamp[0,1]。保存FP32张量供Victim直接读取，不经JPEG/PNG量化；CPU逐图exact重放全部桥接。实际double测量L∞、名义16/255及浮点端点差分别记录，不声称严格实数预算无舍入。文本完全原样。此次预算不是Idea的8/255，不能直接作等预算优劣结论。

三模型使用同一冻结图像集合，IRRA原生归一化点积；RaSa Top128候选+原ITM；APTM原ITM及全Query/Gallery归一化，由既有APTMSetScorer处理，不能逐块归一化。完整Gallery与全部同PID正例排名保留，报告R@1/R@5/R@10/mAP及内部mINP。模型参数/buffers前后SHA、真实神经hook计数和原始输入来源均核验；独立CPU进程验证全部Manifest/SHA、像素桥接/预算、固定输入身份及完整分数重排名，不调用模型/CUDA/优化器/反向/JVP。

## 执行身份

独立工作树/home/lzf/ldx/projects/idea-test-tbps-anyattack-pretrained-20261108；分支codex/anyattack-tbps-pretrained。输出/home/lzf/ldx/outputs/tbps-framework-anyattack-pretrained-20261108；控制/home/lzf/ldx/tmp/anyattack_pretrained_20261108。所有GPU调用使用共享观察策略，无VRAM floor，不干预外部进程。真实失败停止且不自动重试；原有G/H、SGA、日志、Checkpoint与产物不改。服务器直连GitHub失败日志保留在本地asset_download.log，下载转为Windows获取、SHA验证、SCP传入后服务器再验，未启动任何训练。

默认CPU预检：PYTHONPATH=src python -B -m tbps.anyattack.cli --config configs/framework/anyattack_pretrained.server.json。真实推理只能显式--action generate/evaluate --allow-inference；监督器依次门控Smoke、Full。初始状态：资产与CPU验证进行中，未声明Smoke或Full成功。

## 公开来源

- [AnyAttack官方仓库固定版本](https://github.com/jiamingzhang94/AnyAttack/tree/91b209120c61261a51a45ce217dae7f1daac081d)
- [官方推荐权重](https://huggingface.co/jiamingzz/anyattack/tree/4c7172a1ada67ab01e6606ddda4ccacb1362ea4a/checkpoints)
- [AnyAttack CVPR2025论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Anyattack_Towards_Large-scale_Self-supervised_Adversarial_Attacks_on_Vision-language_Models_CVPR_2025_paper.pdf)
- [R²TUA CVPR2026论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_R2TUA_Reconstruction-residual_Based_Targeted_and_Untargeted_Attack_Against_Text-Image_Person_CVPR_2026_paper.pdf)

CPU导入预检首轮使用旧cpu_only护栏时，拦截Transformers导入时创建的无权重GELU常量，未发生模型加载/前向或GPU实验。新入口只在CPU导入闭包允许构造，仍拒绝全部神经前向、torch.load、CUDA和导数；未改变已有全局护栏。此为正式Attempt前接线修复。

## CPU验证回执

官方权重在服务器再次SHA通过；Decoder checkpoint含epoch/global_step/decoder_state_dict/optimizer_state_dict/scheduler_state_dict/best_val_loss，本入口只strict加载142个Decoder状态键，不实例化或恢复优化器。fc.0.weight形状[50176,512]。CPU来源/导入/原生Tokenizer预检PASS，CUDA未初始化；12项新增协议测试PASS，适用Adapter回归总67 PASS。额外历史test_final_APTM_config_dry_no_model_or_cuda要求旧materialized_aptm.server.json冻结执行闭包，在新独立工作树会拒绝，保持历史配置不变，未把该旧执行身份当成本次PASS证据。CPU预检日志/home/lzf/ldx/tmp/anyattack_cpu_preflight_20261108.log。
