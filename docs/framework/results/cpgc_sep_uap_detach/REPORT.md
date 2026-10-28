# C-PGC只读UAP加载桥最小修复

## 修复与算法边界

基于执行db7e3e2建立独立codex/cpgc-sep-uap-detach工作树。生产源码只改target.py和audit.py各一行：torch.load(...,map_location="cpu",weights_only=True)后立即.detach()，在摘要、unsqueeze和F.interpolate之前取得无梯度推理视图。未修改provenance.py、官方UAP、模型、Loss、文本候选、DPP、预算、评分或资源门槛。

作者真实UAP文件SHA 3701c038bb1db7352eda31edbef0c6acdb99b7584c05896266b961ef023f1a2f，FP32 [3,384,384]，原文件保持requires_grad=True；只在内存只读加载桥detach，不重新保存。旧非法Tensor继续被payload_digest拒绝。

## 新旧来源与Attempt

原v1全部45份控制/Source/Target文件SHA、大小、mtime、inode复核不变。旧FAIL及Source PASS完整保留。新的全局配置SHA和producer身份不同，现有固定Source输出契约没有显式跨身份复用入口，本轮不强行移植旧completed；在v2新空间重复必要的两Query Source Smoke，仍读取同一原始UAP，绝不重新生成或优化UAP。

新输出：/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v2/
新控制：/home/lzf/ldx/tmp/cpgc_sep_full_v2_20261029/
执行工作树：/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-uap-detach-20261029

## CPU验证

原101项及新增6项CPU回归全部PASS；真实Source/Victim环境预检PASS，bash -n与git diff --check通过。新增测试直接加载作者真实文件并执行Target/Audit实际AST加载语句，验证数值/形状/dtype/文件SHA和metadata不变、摘要稳定、两模型尺寸插值无梯度；拦截CUDA/backward/autograd.grad/Optimizer.step，未调用神经模型。CPU PASS不等于GPU PASS。

## GPU与Full门禁

资源配置维持db7版本：最低空闲8192MiB、RAM32GiB、磁盘100GiB、inode80000；RaSa融合16但完整Top128、Query Chunk32，精度不变。对应方法六组同配置真实Smoke与独立CPU审计全部PASS，且实测峰值reserved+2048MiB≤8192MiB后才允许Full。任意真实FAIL首错停止、不重试。

nohup+setsid一次性串行监督器执行Source/Target Smoke与审计→C-PGC六Full→SEP Smoke及六Full；正式结果不使用Smoke或论文值填充。成功后独立文档分支codex/cpgc-sep-full-v2-results发布实际指标。此准备报告尚不宣称GPU或Full成功，实际状态看控制phase与不可变completed/FAIL。
