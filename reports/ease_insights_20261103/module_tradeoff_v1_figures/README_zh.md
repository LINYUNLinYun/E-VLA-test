# 已完成模块发现集的静态科研图

[PNG](module_discovery.png)、[PDF](module_discovery.pdf)、[SVG](module_discovery.svg)和[逐suite CSV](suite_counts.csv)从已通过整批核验及独立统计复算的模块实验生成。左图为成功数，中央为AE16−VLM_c的发现集差值，右图为八个完成服务的固定输入成本比与中位数。

AE16在四suite均不低于VLM_c，未获得反向偏好；图仅展示发现证据。Full与VLM_c的160个分析单元保留原身份引用，另四条路径共320条新案例；引用不算新复验。计时不代表闭环或独占GPU加速。未读取正在运行的相机实验成功率。

[生成脚本](make.py)校验三个已接受输入哈希，[provenance.json](provenance.json)保存输入、脚本和所有导出物的哈希。
