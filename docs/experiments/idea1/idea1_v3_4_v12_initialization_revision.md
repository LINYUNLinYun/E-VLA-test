# V12实验：四个初态不一致case的技术修复

最终核验发现：同一官方init向量、env seed7、10步settling下，环境复用使4/40个case的七条件首请求input_hash不一致。涉及Spatial task4，Long task3/8/9。其中Long task3出现约4.7cm物体位置差，不能把这些记录当相同初始观测的配对。

科学条件、mask、参数、rank、noise、period2、case、终止上限均不变。仅这四个case全部七条件作技术替换，按case选择与成功/失败无关；原28条记录保存在initialization_revision/original，其他36case不重跑。新运行每条件创建独立环境，在环境构建前重置Python/NumPy seed7，再沿用原seed、官方init和settling。先对四case各两次独立构建确认首观测一致，后重跑28条并再次比较每case七条件首观测。保留修复源码hash、初态门禁、case清单、原/新数据及验收。不修改原冻结的模型实现、补偿参数或原始记录。

GPU仅0/2/3；不重做已通过的纯模型热态测速。新环境构建开销会改变这28条的回合墙钟，报告中单列技术修复开销，七条件在相同case上均使用相同修复。
