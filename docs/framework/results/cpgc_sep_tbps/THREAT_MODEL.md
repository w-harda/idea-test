# 权限与比较边界

C-PGC：作者ALBEF/Flickr30K源模型及公开预训练通用UAP，测试时Source文本推理选词位置，不用IRRA/RaSa梯度、PID或检索结果调整UAP。图像12/255 FP32，文本至多1词，Gallery与Query均受攻击。

SEP：按要求IRRA+RaSa代理集合包含Target，读取Target输出及真实Test Query PID/图库PID，将正确PID与错误PID原生分数转换为原分类接口；无backward或Optimizer。仅Query文本攻击，Gallery保持Clean。不是严格未知Target的迁移黑盒。

两方法输入文本前处理不同，各自保留Raw Clean/Matched Clean控制。不同模型的数据域权重、原生预处理和精度路径分开绑定，不能拿CUHK权重冒充ICFG或RSTP。本文结果不是G/H新训练，禁止重用旧Baseline指标填充本次结果。
