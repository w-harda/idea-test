# Stage 3B-R：CPU运行导入闭包修复

本阶段从失败Commit `0fb90cd1b280da886aec08834dc13d28eacd5848` 派生，分支 `codex/tbps-framework-stage3br`。只修复来源清单并强化CPU预检，不修改旧算法，也不执行真实模型。

## 文件和来源

- `configs/framework/stage3br_plugins.server.json`：Stage 3A来源配置的独立派生版本；74个原锁定项全部保留，只新增1项，总计75项（包括非Python来源）。
- `configs/framework/stage3br_derivation.server.json`：绑定原配置、派生配置SHA和唯一允许新增的路径/SHA；任何原字段或原pin变化均拒绝。
- `configs/framework/stage3br_preflight.server.json`：原Stage 3B固定Train样本/阈值/预算不变，仅切换到新来源配置；只供下面的CPU入口使用。
- `src/tbps/runtime_closure.py`：工厂入口AST核对、完整导入入口解析、动态文件执行SHA检查、CPU硬闸门和命名空间恢复核验。
- `src/tbps/check_attack_runtime.py`：无GPU执行选项；成功必须显式提供派生证明。原缺项配置仍拒绝。
- `tests/framework/test_stage3br_runtime_closure.py`：25项新增CPU测试，另在循环中逐个检查15份合成依赖的缺失和SHA篡改。

唯一新增历史来源：

```text
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/condition_schema.py
SHA-256: f13932c2688cab4e135d063eed6f066016f6ba26e2316cbc201f19b246b298b4
```

原Stage 3A配置SHA仍为 `d2c01ea7cc70c3c4dd56dceb5de6d654069e129201b54b468e48ff7a2c896123`。不改原配置，不放宽原SHA校验，不复制历史源码、权重或缓存。

## 严格CPU命令

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage3br-20261016
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.check_attack_runtime --config configs/framework/stage3br_preflight.server.json --derivation configs/framework/stage3br_derivation.server.json
```

预期：退出码0，JSON中 `status=PASS`、`forbidden_attempts={}`、`CUDA_initialized=false`，最后一行 `GH_RUNTIME_PREFLIGHT_PASS`。没有run/GPU授权参数；不会执行G/H/IRRA、torch.load、排名或条件准备。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage3br-20261016
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m unittest discover -s tests/framework -v
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -p no:cacheprovider tests/test_dataset_adapter.py tests/test_dynamic_topk.py tests/test_generator_framework.py tests/test_joint_baseline.py
```

## 覆盖范围和闸门

依据真实 `load_legacy_plugins()` 的AST提取所有 `old("...")` 入口，要求恰好与7项显式符号表相符；新增入口或不可静态检查的动态名称必须拒绝，避免预检与工厂再次漂移。

实际解析11个函数/类符号：
G的load_generator/generate_once/ImageOnlyGenerator/pixels_from_pil；
H的TextAttacker/validate_action/official_probe/TokenizerProbe/CaptionFrontend/CharacterPolicy/state_sha_from。

完整预检在同一个CPU硬闸门中运行已有身份/SHA预检和扩展导入检查。阻断torch.load、torch.jit.load、神经Module构造/前向、autograd导数及CUDA初始化入口；即使内部捕获禁用操作异常，最终也不能报PASS。不构造真实G/H，不读取Checkpoint payload。

除普通MetaPath导入外，还拦截SourceFileLoader.get_code，在执行前核对CaptionFrontend直接文件加载的v55/v54/v53及官方SimpleTokenizer。前端自身6项来源声明与派生pins逐项对应。退出时检查sys.path、sys.meta_path、dont_write_bytecode及已有模块不变，不残留旧模块；第三方导入保留正常Python缓存。

实际两条Train Caption的资格/Span/BPE记录仍由原preflight只读核对；不生成攻击动作，不使用Test。75项来源逐条SHA/mtime前后核对。报告记录真实执行到的33份源码及来源摘要，不声称75项均被执行。

## 停止边界

本次只证明固定加载工厂的当前导入闭包及CPU前端检查通过。Checkpoint payload反序列化、真实网络实例化、设备迁移、G像素输出及H完整动作等价仍PENDING。

已具备提交审查、重新申请一次受控GPU Smoke的CPU前置条件；不是新的GPU许可。后续必须重新获得授权、使用新的不可变attempt，并在执行前运行此严格预检。原 `equivalence_v1` 永久保留为FAIL，不恢复、不覆盖；本阶段不提供GPU运行命令，不进入Stage 4。
