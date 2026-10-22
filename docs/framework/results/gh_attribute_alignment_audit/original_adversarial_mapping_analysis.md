# 原文—Unicode攻击文本—G属性监督映射

本轮只处理既有原文/攻击文本。原128条Caption经冻结CaptionFrontend规则重提取，逐条与原资格属性exact；
另对既有H9/H20/H60共384条攻击文本重提取，总规则转换512次。
不执行Tokenizer、IRRA/G/H神经前向或新攻击。

| H | 原属性 | 重提取属性* | 原语义/坐标找回 | 未找回 | 其中自身编辑 | 其中未编辑 | UTF8起点移动 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 612 | 319 | 308 | 304 | 278 | 26 | 472 |
| 20 | 612 | 370 | 363 | 249 | 243 | 6 | 396 |
| 60 | 612 | 369 | 362 | 250 | 238 | 12 | 374 |


*重提取可能新建/重绑定其他属性，因此其总数不等于原属性找回数。
逐原属性以semantic_axis、region、canonical_value、原字符起止坐标严格匹配；多候选标AMBIGUOUS。
本Validation重提取歧义均0，不能与Train原始G/H候选的1个歧义混为一谈。

## 字符坐标、字节与Tokenizer

全部实际替换为1个code point→1个code point，所以字符串字符数量、半开字符Span和后续字符坐标不变。
本白名单由ASCII变Greek/Cyrillic，UTF-8由1字节变2字节，每个编辑增加1字节；
H9/H20/H60整组增加278/243/238字节。必须重算新UTF-8偏移，不能沿用original_utf8_offset。
612个属性中，新字节起点移动472/396/374个。

现有EditRecord保存attribute_id、Canonical、region、原Span/raw、position、position_in_attribute、
original_utf8_offset、原/新字符、码位和Unicode名。Tokenizer前后IDs、长度、新截断等是旧已存审计，
本轮只读；字符Span可保持不代表BPE token坐标或EOT保持。

## 重新提取的实际风险

H9/H20/H60所有自身被编辑属性278/243/238项，都未被原词法规则按原语义/Span重新找回。
还有26/6/12项未被直接编辑的属性消失，提示衣物/对象词改变还可能影响颜色绑定。
这不证明人类语义或神经理解一定丢失；它证明直接重新提取再匹配旧属性不可靠。

## ID与统一Schema

G ID包含row_id、旧slot、linked_object等；H ID只哈希六个语义/Span字段，且不含Caption/PID。
建议新anchor_id绑定original_caption_sha256+六字段，同时保留原G/H ID列表和linked_object；
不要替换旧ID，不要把相同词不同位置强配。

proposed_attribute_schema.json是本轮草案，不是已接入训练的接口。
含original_caption、adversarial_caption及各自SHA、original_span、edited_span、原/新字节Span、
semantic_axis、region、canonical_value、attribute_id、character_edit、unicode_codepoint、
G_selection、H_selection、linearization_caption_sha256及来源。
G/H选择和H实际编辑分开；UNRESOLVED/AMBIGUOUS保留列表，不能伪造唯一映射。
PID/配对只在独立training_binding_ref，不能进入H公开策略输入。

## 旧JVP的含义

v_i(T)是在原Caption的block8激活H_T上，按原token定位w_i和语境方向e_i计算的线性化。
Unicode改变后，即使本属性未编辑，整句BPE/激活/reader定位都可能改变。
原z_T不能冒充z_Tadv；旧v_i(T)不能改签为v_i(Tadv)。维度都是512或77不能证明可复用。

可保留Q/basis和原Canonical作为固定语义锚点，新语境token映射、reader w/e、z和JVP要按新来源重算。
若仍保留旧JVP作原文语义参照，必须如此命名；该参照本身不构成H→G反馈。
现有prep_text_frontend.clean==old_z_t及原caption/selection门禁会拒绝直接改Caption；
未来只能建立独立新条件协议，不能关闭旧门禁。

例如原blue的字符Span保持，H输出bluе可通过编辑trace精确回放；
保存edited_raw=bluе并继承原Canonical blue锚点，不要求词法规则重新认出bluе。
可追踪性由既有动作证明，新语境JVP数值仍需未来授权验证。
