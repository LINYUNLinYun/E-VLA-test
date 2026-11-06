# 修正周期3 Jacobian：评分来自Full-D10

2026-10-30。用户澄清：在完整Full路径上对动作求各层敏感度，保留高影响层，跳过低影响层得到V14/V12，之后两个Fast复用。原cycle3 Jacobian虽使用全18层VLM，但其额外可微AE仍为D2，未与执行Full-D10对齐；保留原结果，不能把它概括成Full-D10 Jacobian方案的成败。

修正Jacobian评分路径的FM日程D2→D10，并修正Full结束后的输入重建；不擅改为未来第6–15步动作。核验发现原cycle3重放只重投影前512个视觉token，后续语言token仍是Full更新过的隐藏状态。单输入探测确认视觉最大差0、语言最大差3426，恢复原语言嵌入后初始输入逐位一致，证据见input_reconstruction_probe.json。新实现必须在重放前恢复原始语言嵌入，四suite原始输入身份检查作为上线门；不是把这一错误称为跨帧敏感度失效。两处同时修正，后续结果变化不能单独归因于D10日程。

沿用前5步指数时间权重[1,2,4,8,16]/31、七维全+1、一次VJP近似层影响；这不是完整35行Jacobian范数，后续报告明确区分。全部18层参与评分，在17个可跳层的合法skip4/skip6组合中最小化abs门控梯度和，最多连续跳2层，保留14/12层；L18固定KV-only不参与跳过。

可微重放使用同checkpoint、同输入、同噪声和原生Full-D10的10步时间偏置/输出权重，完整反传10个FM步骤，不detach、不沿用D2后向。原生Full-D10执行动作不改变；原生Triton不支持autograd，梯度来自PyTorch BF16/FP32 RMS的同结构D10重放。预先明确：Graph/eager raw atol=rtol1e-5、梯度atol1e-4/rtol1e-3、mask相同为上线门；PyTorch与原生Full的数值差另测并公开，不声称原生Triton算子精确反传，也不因误差静默替换执行后端。

执行周期仍Full-D10→Fast-D2→Fast-D2，H50/R5、max2，Jacobian仅在Full轮评分，Fast零探测，评分后的mask在后两个Fast保持。原cycle3余弦和随机基准在初态/首Full动作一致性审核后复用；只新跑Jacobian V14/V12各40条，共80条，固定相同四suite奇数task1/3/5/7/9×init19/20。环境构建前seed7、原步数上限和噪声流不变。有效失败不重试，技术失败最多恢复一次。

上线前四suite固定输入：D10重放实际执行10步、时间日程/输出权重与原生Full一致；Graph/eager输出与梯度一致，排序高影响保留方向正确；原生执行Full与旧参考不变。记录重放/原生差值、选层新开销、冷捕获和显存。仅Full轮付出必要的额外D10重放/反传成本，并计入真实周期；不直接套用旧65ms。

最终结果同时列原D2代理Jacobian、新Full-D10 Jacobian、余弦/随机基准成功率和配对探索区间；分Full/Fast/完整三轮稳态成本，保留全部episode总墙钟。解释仅限此单次投影候选，不混淆完整Jacobian矩阵和完整模型路径。
