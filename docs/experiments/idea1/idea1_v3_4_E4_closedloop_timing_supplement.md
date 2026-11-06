# E4闭环：热态计时技术补充

2026-10-30，在闭环运行中增加，不改变任何科学条件、case、seed、缓存或刷新日程。

首轮固定输入测速只对每suite task0预热。检查逐请求分解发现，其他prompt/shape第一次调用某些补偿kernel时触发编译，VLM计时最高约364ms，而相应热态中位数约15ms。原数据完整保留为startup_mixed_benchmark.json，不能当纯热态部署速度。

闭环继续执行；完成后在GPU0/2/3重新分别启动同一Backend和相同工程gate，每个固定输入对六条实际计算路径均先预热5次，再原顺序/逆序各测一次。40输入×6路径×2遍=480个测量请求/卡，预热1200次/卡；重复同一路径保持相同obs、state、prompt、显式noise、history。时间包含视觉前向、模型、缓存提取/补偿和解码。预热及服务加载成本另列。

原manifest、原闭环代码及结果身份不变；补充脚本独立保存source hash和gate。只替换报告中被编译污染的热态表，保留原始计时、原始报告与补充运行记录。三个周期条件依旧使用(Full+draft1+draft2)/3摊销，质量结论仅来自真实闭环。

GPU0/2在闭环释放后被其他任务使用，补测等待空闲卡，不切换至用户排除的GPU1。GPU3首轮热态补测中途出现外部compute进程；该轮保存在warm_timing_shared及shared_gpu_benchmark.json，主测速不采用。待其空闲后按相同输入和参数重做GPU3。GPU2、GPU0分别在空闲后启动；timing_resource_samples.jsonl记录后续0/2/3的compute进程用于识别并发。主报告只使用最终独占测量。
