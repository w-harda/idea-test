# SEP-Attack TBPS适配审计

官方仓库https://github.com/ZevineXu/SEP-Attack ，Commit3a6d848a48097aee79be0299da8959e78db543c7。WWW2026论文https://arxiv.org/pdf/2605.24958 。这是原分类攻击的检索适配，不称原论文完全同设置复现。

原Ensemble/DPP、importance、optimize、attack函数体保持官方定义。仅_parse_models加载接口接入两个原生检索模型；官方顶层缺少的无关分类包通过只加载原AST定义隔离，不重写候选机制。

## 数学接线

每个模型原生图文分数s_m(t,i)，将同PID所有图库图像max聚合为z_m(t,p)=max_{i:pid(i)=p}s_m(t,i)，作为原分类logits。多正例不挑某个配对图片。原函数自行softmax，c_m=p(y)-max_{p≠y}p(p)，loss=sum_m w_m max(c_m,-k)。仍是原有margin，不新增检索Loss或梯度。RaSa的每次候选评分实际执行完整图库Top128+ITM，绝不用余弦代替融合。

DPP保留100随机权重池→原greedy MAP DPP→keep=len(models)，指定两模型因此保留2个权重，论文原设置4模型/4权重不同。Target在代理中，访问其输出和真实Query PID、全Test Gallery标签；不访问模型梯度。每个Target仍保留原clean eligibility和成功统计调用，但不以成功与否事后选最终文本。

## 默认冲突与固定最终文本

论文epsilon=.15、eta=.2、T10、gamma=.02；原代码ceil与floor存在差异，实际保留代码floor/minimum1逻辑，不静默更改；迭代10、patience5、top50原同义词。CLI k=.9、README例子1，采用实际CLI默认.9并记录。原脚本运行十组gamma并统计Top10成功率，不能直接变成一条检索输入。当前固定gamma=.02分支原tau排序第一条，在该分支任何Victim成功测试前只读捕获，不依据Victim成败筛选；仍执行原其他分支用于原统计。这个top1输出协议区别于论文TopK攻击成功率，必须说明。

Python/NumPy保持官方seed0；原Torch未固定，新增预注册Torch0仅使DPP随机权重可追踪。每Query保存原文、clean_str(...256)后的Matched Clean、最终文本、编辑位置/比例、NOOP原因、随机状态与候选tau。失败/无候选/原Clean错误Query全部保留分母。

## 语义与失败边界

官方counter-fitted词资源包含服装/颜色/性别不严格等价词，不能把同义词表当作身份语义保持证明。保存实际编辑和身份词变化标记；未确认的语义明确NOT_VERIFIED，不通过过滤候选来偷偷改变算法。语义存在改变时，报告实际攻击结果与语义合法性为两个不同结论。

真实CPU边界Fixture已发现：当importance完全并列时原缩放产生NaN，np.random.choice报ValueError: probabilities contain NaN。所有28004条真实Test Caption的CPU词性检查均至少6个可编辑词，无0/1候选，但不能证明真实模型不会全并列。保留原代码、不加epsilon或改归一化；若真实Smoke/Full触发，保留FAIL并停止，不能宣称通过。
