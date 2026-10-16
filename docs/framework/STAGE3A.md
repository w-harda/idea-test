# Stage 3A：冻结G/H攻击插件

本阶段新增攻击侧薄插件。没有训练、GPU、真实G/H模型前向、IRRA前向或Full指标计算。CPU Fixture通过不等于真实G/H等价验收。完整依赖、核验结果和Stage 3B建议见[报告](STAGE3A_REPORT.md)。

## 文件与接口

| 文件 | 职责 |
| --- | --- |
| src/tbps/attacks/base.py | PluginInfo、公开Request、独立AttackBinding、像素/文本AttackOutput；可选训练能力默认不可执行 |
| registry.py | 注册/发现/显式依赖工厂；无自动模型加载 |
| configuration.py | 冻结生成阶段权限、预算、代理/目标角色与目标就绪状态 |
| provenance.py | 固定G10/H7、Selection、协议、相关producer及原条件元数据只读索引；CPU CLI |
| legacy_runtime.py | 显式加载门禁、旧模块绝对import隔离、原模型只读加载与会话冻结检查 |
| gv2.py | 调用原generate_once；验证一次G前向、实际RGB像素、8/255、冻结和输入SHA |
| hv1.py | 调用原TextAttacker/validate_action；保留资格检查、Dynamic-K、STOP/NOOP和Unicode规则 |
| joint.py | 组合G/H同一实例；image_only/text_only/joint，独立追踪两种产物 |
| storage.py | 复用新ArtifactStore原子发布紧凑像素/扰动或Unicode文本及动作；不保存模型Embedding |
| __init__.py | 无模型副作用的最小导出 |
| configs/framework/stage3a_plugins.server.json | 固定历史资产索引、74项旧运行依赖、训练侧Tokenizer与三组预算 |
| tests/framework/stage3a_fixtures.py、test_stage3a_plugins.py、test_stage3a_provenance.py | CPU Mock/真实旧纯函数、权限、预算、Span/Unicode、来源及存储测试 |

src/tbps/config.py仅更正当前可用攻击插件列表及“缺少统一Evaluation Engine”的拒绝信息。原实验源码、Stage 2 Adapter、训练程序及资产不改。

## 能力与边界

| 插件 | 在线输入 | 在线输出 | 在线代理/梯度/奖励/优化 |
| --- | --- | --- | --- |
| gv2 | PublicImageInput：原图FileRef及RGB空间；绑定身份另存 | 原G工作分辨率的RGB [0,1]像素、effective_delta、来源/预算/调用记录 | 全部不需要 |
| hv1 | PublicTextInput：原始Caption；不接外部预选属性，Canonical由原前端复算 | 原文/对抗Caption、完整属性Span、K、选择/STOP/NOOP、Unicode编辑和Tokenizer审计 | 全部不需要；只做固定训练侧Byte-BPE约束 |
| joint | 独立的ImageRequest/TextRequest，不要求二者正确配对 | 分离的image/text AttackOutput | 委托同一G/H组件 |

线上Request、Binding和策略之间分层：Dataset Manifest的PID/配对信息只用于上游身份校验，Binding只留Manifest摘要、sample_id、模态及输入SHA；后端G只收到像素，H只收到Caption。Label/paired_index/reward/S_raw/JVP/victim参数不在生成接口。

预算固定：
- G：原始RGB攻击工作空间、384×128、L∞≤8/255，锚点是原图经既定Resize/ToTensor后的干净像素；不在标准化特征空间扰动。没有恢复JPEG原分辨率像素的隐式转换。
- H：原Natural-K gamma=3、每属性最多1字符、共享词最多1字符、总编辑≤K；保留旧Unicode白名单、源Tokenizer改变/不截断掩码；不增加固定K=4或目标模型反馈。
- Joint：以上两个预算并列，三种模式共享固定权重及同一已核验来源bundle。

PluginInfo.spec.training_kind描述历史方法；training_executable和offline_preparation_executable均False。没有train/Optimizer的通用强制接口。require_preparation明确拒绝；条件来源审计不等于执行准备。

## 最小使用方式

发现与描述（无Torch/旧模块导入、无CUDA）：

```python
from tbps.attacks import default_registry
registry = default_registry()
for info in registry.discover():
    print(info.as_dict())
```

已核验Dataset Adapter创建Manifest及公开输入后，绑定真实成员（标签不会进入G/H）：

```python
from tbps.attacks import AttackBinding, ImageRequest, TextRequest
image_binding = AttackBinding.from_manifest(manifest, image_id, "image")
image_request = ImageRequest(dataset.public_image(manifest, image_id), image_binding)
text_binding = AttackBinding.from_manifest(manifest, query_id, "text")
text_request = TextRequest(dataset.public_text(manifest, query_id), text_binding)
```

真实加载API为load_legacy_plugins的上下文会话，allow_model_load默认False且先于任何来源/权重访问拒绝；**Stage 3A没有调用该API的授权分支**。将来单独授权后可显式选择components=("gv2",)或("hv1",)；H保持CPU，G可显式指定cpu/cuda:index。全会话冻结验证，退出恢复旧import命名空间、sys.path和pyc标志；不使用load_latest/current/rollback。

Fixture通过后，可以用registry.create("gv2", backend=..., provenance=...)注入明确cpu_fixture后端；不把Mock打包为真实权重。Joint通过registry.create("joint", image_plugin=g, text_plugin=h)组合。

Joint.generate(mode="joint", image=image_request, text=text_request)不检查二者是正例。未来完整Gallery和Query必须分别遍历image_plugin/text_plugin；不要把3074 Gallery与6156 Query按一一配对zip。产物发布分别调用storage.publish_output，不合成一个特征文件。

## 新攻击方法最小扩展

只增加新插件模块/配置及注册，不修改G/H文件。例子是CPU回显Fixture，不是SGA/MTGA实现：

```python
from tbps.contracts import AttackSpec
from tbps.attacks import PluginInfo
spec = AttackSpec("example", "v1", frozenset({"text"}), "policy", None,
                  False, frozenset(), frozenset({"captions"}), None)
info = PluginInfo(spec, frozenset(), frozenset({"captions"}))
class Example:
    def __init__(self, budget):
        self.info, self.budget = info, budget
    def generate(self, request):
        return request.public.caption
registry.register(info, Example)
```

迭代攻击可以自行声明attack_time_optimization及source_gradients等能力，提供自己的算法/阶段配置。当前validate_generation只接受冻结公共输入生成，迭代插件不得借该校验伪装可执行；未来另接专属权限校验，不改G/H算法。

## 跨模型约束

validate_generation可接受37维或Fusion Mock ModelSpec作为目标契约，不要求512维/77 tokens/余弦双塔；未实现RaSa/APTM Adapter时victim_adapter_ready=True会拒绝。未实现Dataset Adapter同样拒绝。没有统一评分Engine。

G/H训练代理及H候选约束Tokenizer固定为IRRA来源；切换目标模型不能传入目标Tokenizer修改H动作。攻击产物是像素/文本，不含IRRA Embedding。未来评分方独立处理原生预处理/Tokenizer并记录变化，不反馈给已冻结攻击器。G的384×128工作像素与未来目标图像尺寸的映射需要另行验证，当前不宣称跨模型评测已可执行。

## 服务器CPU命令

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage3a-20261016
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m unittest discover -s tests/framework -v
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -p no:cacheprovider tests/test_dataset_adapter.py tests/test_dynamic_topk.py tests/test_generator_framework.py tests/test_joint_baseline.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.attacks.provenance --config configs/framework/stage3a_plugins.server.json --condition-row cuhk:9955:0 --condition-row cuhk:26530:0
```

最后一项成功标志STAGE3A_PLUGIN_SOURCE_AUDIT_PASS；无GPU/run参数，不反序列化模型或条件Tensor。缺失路径明确NOT_READY并拒绝；SHA/来源/身份冲突拒绝。

本轮CPU证据仅写/home/lzf/ldx/outputs/tbps-framework-stage3a/，不提交Git。没有可自动执行Stage 3B或Full的脚本/后台任务。
