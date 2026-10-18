# SGA首轮IRRA Smoke：严格数值投影FAIL（保留）

官方SGA f49e6926cd13148e49bdbc2c3e537660e096230c，初始源码/config被冻结于本工作树。三个CPU来源预检PASS，36项专项CPU及113项相关回归通过（专项后来增加2项），但未覆盖clean像素8/255附近的低端舍入问题。

唯一初始IRRA GPU尝试：/home/lzf/ldx/outputs/tbps-framework-sga-whitebox/smoke/irra/，status=FAIL，victim_loads=1，Claim1/Image0；RaSa/APTM未启动。execution/report.json报strict global RGB attack budget，不是OOM。耗时16.471906285732985秒（不含CPU预检），GPU allocated958375936/reserved1088421888 bytes。启动空闲3865MiB/利用率92%，Ollama未干预。日志/home/lzf/ldx/tmp/sga_smoke_irra.log。

CPU复现：clean=arange(256,float32)/255，project到负端时像素8、9、10、11越界；max_linf=0.0313725508749485，大于数学8/255=0.03137254901960784。原因是先做FP32 clean-epsilon，再一次nextafter；低端接近0时其ULP不足抵消epsilon先转FP32的误差。必须先在FP64中算clean±数学epsilon，再转FP32并内舍入；不允许加容差或改变预算。

初始报告未在异常路径持久化实际视觉Hook计数，因此不编造精确次数；发生过输入梯度计算，IRRA样本前向界为5到50（首图、最多10轮×5尺度），精确值UNKNOWN。G/H/JVP/Optimizer为0，无成功像素或检索指标。后续修复需完善异常计数。

本树、原config、FAIL文件与Claim保持，不恢复或改签。数值修复在新的独立派生工作树及输出目录验证；不覆盖原失败，也不启动Full。
