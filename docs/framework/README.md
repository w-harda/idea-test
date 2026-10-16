# Stage 1：基础契约、配置校验与旧资产只读桥

审计基准：40c7476f92d6bc2458d86029cf7bf8317da0a086。状态唯一入口：[PROJECT_STATE](../PROJECT_STATE.md)；保护规范：[AGENTS](../../AGENTS.md)。本阶段没有训练、模型加载或评测执行引擎。

## 文件职责

| 新文件 | 职责 |
| --- | --- |
| src/tbps/contracts.py | frozen/slots契约、显式预算、公开输入与身份/奖励分层 |
| src/tbps/config.py | 严格JSON字段、来源角色、能力/权限与迁移轴校验；仅dry-run |
| src/tbps/compat/legacy_assets.py | 原路径只读SHA/JSON关联、旧完成证据描述；不加载Checkpoint |
| src/tbps/artifacts.py | 新输出命名空间的不可变Manifest/回执、原子发布、SHA、CPU Tensor clone |
| configs/framework/example_irra_gv2.json | 真实服务器路径的配置解析示例，不是可运行训练配置 |
| configs/framework/legacy_assets.server.json | 8类旧资产、192个唯一文件的Git固定SHA及协议关系 |
| tests/framework/ | 合成CPU Fixture、权限/身份/来源/原子性/Storage测试 |
| docs/decisions/0001、0002 | 冻结只读桥、阶段权限及原生评分边界 |

## 契约最小示例

```python
from tbps.contracts import AttributeSpan, PublicTextInput, ModelSpec
caption = "red red shirt"
public = PublicTextInput(caption, (
    AttributeSpan("color", "red", "red", 4, 7),
))
# public只有Caption与准确原Span；不能附加pid/paired_index/reward。
native = ModelSpec("future-fusion", "design-only", None, None, None,
                   "fusion", "model_native",
                   frozenset({"score", "fusion", "local_features"}))
# 这只是能力契约，绝不意味着future-fusion已实现。
```

DatasetManifest保留原始Gallery/Query顺序；image_id/query_id可使用原row_id，配对由paired_image_id表示。PersonID(dataset, value)禁止把两个数据集的同数字PID当作同一身份；多Caption和全部同PID正例均保留。当前relevance只接受all_same_pid，其他数据集规则必须由未来Adapter明确扩展，不能静默套用。

EvaluationIdentity仅给评测层，TrainRewardContext仅给已验证的official train/attack_train奖励绑定层；后者要求正确配对正例及至少一个负例，拒绝Validation/Test角色。它不代替未来Dataset Adapter的真实来源验证。Policy接口必须只接收PublicTextInput，不接收上述两个上下文。

ModelSpec允许不同表示维度、Tokenizer长度/单位/截断策略、H/W、dot/native/fusion；native模型可没有独立global embedding。RepresentationArtifact携带模型/预处理/Tokenizer/空间SHA，ScoreArtifact保存原生评分协议和有序身份。AttackSpec可描述generator/policy/UAP/iterative/composite，不强迫所有方法都有训练阶段。

ImageBudget必须声明norm和pixel_space；TextBudget在固定max_edits与动态selection_rule之间二选一，并单独声明约束与每属性/共享词上限。框架通用契约不固定8/255；历史G/H描述配置明确锁定其预算和约束。

## 配置校验

完整最小JSON见[example_irra_gv2.json](../../configs/framework/example_irra_gv2.json)。显式区分：

- attack_method、surrogate_model、victim_model；
- attack_train_dataset、attack_validation_dataset、attack_test_dataset的官方split和内部role；
- attack_checkpoint及其攻击训练/选优来源；
- surrogate_checkpoint、victim_checkpoint及各自model_training_dataset；
- attack_mode、attack_budget、seed，以及train/attack_time权限。

white_box要求代理/受害者模型身份及Checkpoint SHA相同；cross_model/cross_dataset两个迁移轴必须与实际声明一致。Checkpoint自身训练来源不自动等于攻击器训练来源。配置解析不读取大规模数据或核实真实PID隔离；这由已绑定历史协议和下一阶段Dataset Adapter负责。

已知gv2/hv1/joint/ap_attack、irra、cuhk_pedes仅有历史描述符，全部executable=false。RaSa/APTM、ICFG/RSTP及未知Baseline明确拒绝；require_executable()始终拒绝，不能把“配置合法”解释为“插件已支持”。

服务器CPU dry-run：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage1-20261016
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.config --config configs/framework/example_irra_gv2.json
```

成功标志CONFIG_VALID_DRY_RUN。PRESENT_NOT_SHA_VERIFIED只表示输入路径存在；路径缺失为NOT_READY，始终不会加载模型或执行实验。

## 旧资产只读桥

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage1-20261016
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.compat.legacy_assets --catalog configs/framework/legacy_assets.server.json
```

只输出描述，成功标志BRIDGE_METADATA_VERIFIED_NO_MODEL_EXECUTION；缺失路径为NOT_READY/exit2；SHA/schema/协议/来源冲突为BRIDGE_REJECTED/exit1。不调用旧load_latest、恢复、prepare、旧Store或模型forward。

需要保存一个**新的**不可变桥快照时，在相同命令后追加：

```bash
--output-base /home/lzf/ldx/outputs/tbps-framework-stage1 --run-id review-history-v1
```

追加参数仅写新命名空间；不复制、转换或改写旧资产。已存在run-id拒绝覆盖。生产者只锁定实际相关的contracts/artifacts/legacy_assets三文件，不枚举整个目录。新加无关训练文件不会改变锁定集合。完成后新目录含manifest.json、completed.json、bridge_manifest.json；manifest绑定目录配置文件及生产者SHA。

实际输出片段（完整版本在新桥目录）：

```json
{
  "schema": "tbps-legacy-bridge-1",
  "baseline_commit": "40c7476f92d6bc2458d86029cf7bf8317da0a086",
  "assets": [
    {
      "id": "gv2_epoch10",
      "status": "METADATA_VERIFIED",
      "protocol_digest": "1ada0c31ee4fd4bf35eaf02fcc5bcebc7a8276c2714f34aaa6d9a88e3501f50b",
      "models": {
        "G": "f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e",
        "IRRA": "9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009"
      },
      "reuse_scope": "read_only_reporting",
      "full_feature_audit_performed": false,
      "executable": false
    }
  ]
}
```

完整files映射包含原路径、SHA、大小和mtime_ns，datasets区分攻击训练/Validation/评测来源。此片段省略其余字段和7个资产，不能作为独立完整Manifest使用。

核验内容：

- G-v2 E10文件SHA、sidecar/训练契约及2048+128缓存Manifest绑定。
- H Epoch7选择、14336 visit（14259实际更新）、Checkpoint/intent/step日志绑定，10份固定Validation候选的已有结果及最小Joint mAP/最早visit关系。只比较已存标量，不重算检索。
- G-v2 Full、Joint Full、消融Full完成回执/CSV/相关JSON；消融11个trial子协议→父协议关联和独立审计日志双成功标志。
- AP B1/B2/B3只做metadata级来源描述，未把它们升级为新框架全资产审计通过。

METADATA_VERIFIED不表示重新审计全部特征/Checkpoint内部Adam/RNG或训练轨迹。旧IRRA特征不能给RaSa/APTM使用；require_feature_reuse()即使模型/协议相同也拒绝直接执行，等待后续已审计Adapter。旧Storage兼容记录保持原样，只检查其旧协议绑定，不重新执行修复。

## 新资产存储与中断

```python
from tbps.artifacts import ArtifactStore
base = "/home/lzf/ldx/outputs/tbps-framework-demo"
store = ArtifactStore(base + "/artifacts", new_base=base)
store.publish("unit-v1", {"notes.json": b'{"value":1}'},
              {"schema": "demo-source-1"})
store.verify("unit-v1", expected_provenance={"schema": "demo-source-1"})
```

new_base必须是专属tbps-framework*命名空间，默认拒绝prototypes/原项目树/数据/官方模型树，并可追加protected_roots。构造、pending、verify不写文件；发布才获取Linux flock并创建输出。数据fsync→Manifest→完成回执→目录原子rename→父目录fsync；已发布版本不可覆盖。

故障或进程退出留下.pending-*证据，未提交目录不参与verify；同ID或无有效intent的pending阻止静默重试。Stage 1不提供自动恢复/删除pending；先检查新输出证据，再由用户决定独立run-id或后续恢复设计。禁止把这一机制用于旧Checkpoint/旧缓存目录。

publish_tensor只接收CPU Tensor；保存前detach().clone()，load_tensor读取受SHA保护的Tensor后再次clone。测试证明旧view格式数值exact，返回Storage紧凑，新旧格式可共存。没有旧缓存迁移或改写。

## CPU验证

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage1-20261016
TMPDIR=/home/lzf/ldx/tmp CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m unittest discover -s tests/framework -v
```

本阶段114项框架CPU测试；另有36项原数据/Top-K/生成器/Joint基础CPU回归。详见[阶段报告](STAGE1_REPORT.md)。不在测试/安装脚本中放置GPU或Full命令。

## Stage 2建议（未授权、未实现）

1. Dataset Adapter：先CUHK-PEDES官方annotation只读解析，复现原Gallery/Query顺序、多Caption及全部正例；输出Manifest和公开输入投影，核验真实split/PID/源图SHA。
2. Model Adapter：先IRRA显式像素预处理、原Tokenizer/归一化及官方排名契约；返回带模型空间身份的表示或Score。默认不加载模型，真实执行必须另行授权。
3. 为RaSa/APTM设计原生融合Score接口后再接真实源码/权重，逐个声明能力；不伪造512维cosine适配。
4. 新旧特征/指标等价验证通过后再设计薄Attack Plugin；保持G/H算法循环和各自恢复边界，不合并成一个通用训练算法。
