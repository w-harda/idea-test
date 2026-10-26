# R²TUA参考对齐的CUHK-PEDES / IRRA评测标准

状态：设计与CPU核验完成，**未修改或执行任何评测入口**。这是可公开复查的官方IRRA参考标准，不等于确认了R²TUA作者未公开的实际运行配置。
版本日期2026-10-26；审计基准c2062e430c664d626c07782490128fc3316dfa61。
目标为不重跑SGA/MTGA/AnyAttack等所有算法，保留可追溯的跨论文引用。

## 1. 应固定的数据与身份

| 项目 | 规范 |
| --- | --- |
| Dataset | CUHK-PEDES官方reid_raw.json，使用split=test；不凭文件夹名判断split |
| Gallery | 每个Test图像一次，共3074；不按Query复制Gallery |
| Query | 所有原始Caption，共6156；不假定每图恰好2条Caption |
| PID | Test1000 PID，全部同PID图像为正例 |
| 顺序 | 保留annotation中的图像顺序，再按每图原Caption顺序展开；固定Manifest SHA |
| 隔离 | Train/内部Validation/Test身份隔离；Test仅评测，不选权重或调参 |
| Caption | Raw Caption进入官方SimpleTokenizer；不额外调用pre_caption(...,30)、删标点或删长句 |
| 图像 | 原JPEG只读，PIL convert RGB；不增广、不翻转、不多裁剪、不重压缩 |

官方读取器datasets/cuhkpedes.py::_split_anno/_process_anno与我们的构造规则一致。本机数据实际数量可确认；R²TUA声称官方划分，但未公布其逐成员Manifest或测试数量日志，不能写成“已确认它与本机每个成员完全相同”。

当前原始annotation字节SHA：fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0。
当前Raw Test Manifest SHA：a338f5429b64393b8062e0a7127773d72adb5dd6832eeb89d24c2c63c2f4d9eb。

## 2. 唯一IRRA模型及官方来源

固定anosorae/IRRA公开代码Commit，不跟随main漂移；本机实际加载的独立冻结副本与该Commit的11份已核查文件逐字节一致。

- 官方Commit：cfc23785545ead82dee2749fa0fc0ad948c96430。
- 实际来源根：/home/lzf/Attack/IRRA-main/IRRA-main。
- 权重：/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth。
- 权重SHA：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。
- 冻结配置SHA：8a9c1921ebcf93babcfdc07d35525a8d13572aeebdec5605e390289c536c5175。
- BPE字典SHA：924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a。

模型为官方ViT-B/16 / CLIP text Transformer，图像stride16、文本77、词表49408、cmt_depth4；不要替换为本机其他含Prompt扩展的IRRA工作树。
官方README的CUHK模型与日志下载入口为：
[IRRA官方Model & log](https://drive.google.com/file/d/1OBhFhpZpltRMZ88K6ceNUv4vZgevsFCW/view?usp=share_link)。
入口已由官方README确认；Google Drive网页本轮工具无法直接读取档案内部，不声称重新下载并逐字节匹配了官方压缩包，也不声称R²TUA使用相同权重SHA。
只固定已使用并验证的权重，不搜索能得到更高Clean的替代Checkpoint。

## 3. 图像与文本输入

| 参数 | 固定值 |
| --- | --- |
| 图像尺寸H×W | 384×128 |
| Resize | 官方torchvision Resize((384,128))，PIL RGB输入，默认bilinear；固定Pillow/torchvision版本 |
| Tensor | ToTensor，RGB，[0,1]；生成与物化保持FP32，不做额外PIL/JPEG重新量化 |
| Mean | 0.48145466, 0.4578275, 0.40821073 |
| Std | 0.26862954, 0.26130258, 0.27577711 |
| Tokenizer | 官方SimpleTokenizer，CLIP Byte-BPE；不换BERT |
| 文本处理 | 官方ftfy/html-unescape/whitespace/lower流程；不叠加Baseline自己的pre_caption |
| 上下文 | 77，SOT49406、EOT49407、零填充；超长保留头部并将末位改EOT |
| 正例规则 | 图文PID相同即可，多正例全部保留；无camera过滤 |

CPU实测：6156条Raw Caption的官方tokenize与本项目桥接Token ID全部一致；15条发生官方77上下文截断，最大未截断长度115。不删除这15条。
固定前3张Clean图像，两条IRRA预处理路径像素差为0；这不是整个3074图像集合的神经特征验证。

## 4. 必须显式区分的两个数值执行Profile

### PUBLIC_IRRA_REFERENCE（推荐未来用于官方参考对齐，尚未启用）

直接保留官方build_model的convert_weights语义：适用的Conv/Linear/Attention参数FP16、LayerNorm等按官方实现处理；不要把“混合参数精度”误写成一律FP16。
encode_image/encode_text取官方全局/EOT输出并返回float，后续单位化和相似度为FP32。
依官方test.py路径构建、严格加载唯一best.pth、eval/no_grad。固定运行库版本、硬件、编码batch与评分设备，记录实际dtype分布。
官方README当时参考torch1.9.0/torchvision0.10.0；R²TUA正文写torch2.1，未给torchvision/Pillow/CUDA版本。两者不能混称同一个环境。
本轮没有安装旧环境、改动Loader或执行Clean。当前框架有强制model.float()，因此**现有入口不能未经独立实现/来源绑定就宣称支持本Profile**。

### LEGACY_RAW_FP32（当前既有G10/G60结果的冻结Profile）

继续保持原model.float、FP32/math/no_tf32、image_batch1、text_batch1、query_chunk128，以及现有模型空间SHA。
历史G10、G10+H7、G60、G60+H7只属于此Profile，不回写、不重贴标签。
IRRA的共同Raw物化图像/文本定义与上节一致；与PUBLIC_IRRA_REFERENCE已确认差异在编码数值路径，而非已证实的图像Resize错误。

两个Profile都不能直接标为“R²TUA实际精度”。不根据哪个Clean更接近论文而事后选Profile或Checkpoint。
今后若启用推荐Profile，先预注册、只运行一次官方Clean核验；应独立保存并报告新Clean。现有G固定图像产物若像素空间/预算/身份满足协议，可复用后重新编码评分，不需要重训G或复现全部Baseline。任何GPU行为需下一次明确授权。

## 5. 检索评分标准

IRRA分别输出512维图像全局特征、文本EOT特征，L2单位化，S=Q @ I.T。
只统计Text→Image，不加入训练classifier、ITM、跨编码融合或Top128重排序。
对全部3074 Gallery降序完整argsort；沿官方默认非stable行为，不加自定义ID打破并列。
R@k=所有Query中前k包含任一同PID正例的比例×100。
AP(q)=该Query全部正例所在rank的precision均值；mAP=6156个AP均值×100。
无camera过滤，不删失败/长Caption/NOOP；不以image平均或batch均值代替Query平均。
保留未舍入指标，最后展示两位小数。论文原表只有两位精度，不能推算其未舍入值。
本轮仅源码核验，没有重新生成分数矩阵、执行任何真实排名。

## 6. 8/255及攻击面

本项目G继续使用原始FP32 RGB原图为参照，在归一化前投影，使最终物化像素L∞≤8/255且处于[0,1]，不修改原generate_once或投影算法。
Mean/std归一化在攻击后执行；不能把归一化空间的8/255当作原RGB的8/255。
G Image-only：只修改Gallery，Raw Query保持原文，G(I)不访问Test正确Caption、PID或源梯度。
G+H Joint：同一套Gallery加H(T)文本，H预算/合法性另列；不能把它当作Image-only8/255的同攻击面。
评测PID只用于正例统计，不得反馈攻击器。
对R²TUA仅能确认名义最终8/255；其具体归一化前后位置、图像范围和保存量化没有公开代码证明。其其他Table1 Baseline的epsilon也没有逐行公开，不能把整张引用表标题写成“所有方法均8/255”。

## 7. Clean与引用门禁

保存本项目实测Clean和论文Clean两组，各自标明来源；不以论文Clean替换本项目Clean，不用任意容差宣称权重/环境等价。
Clean接近支持基线水平接近，不能单独证明攻击侧权限、实际epsilon、成员身份及Source相同。
最小后续成本是一个已固定官方IRRA参考Profile的Clean核验和必要时本方法固定产物评分；不是复现所有攻击算法。
当前论文Table1的Untargeted行可以带author-reported及差异脚注引用；未核实条件保持UNKNOWN。
Targeted↑、其他数据集、Table2 Market1501及Table3不同源池/16预算不得混进当前CUHK Untargeted8/255公平比较表。
