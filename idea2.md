# idea2

## Realtime VLA: 

### background & motivation
做推理pipeline上的加速，主要是针对pi系列，xr-1，cosmos3这种MoT架构。

首先要先看看这类架构的数据依赖关系：[跳转下](./zrh的vla调研.md#123-vla的特征融合)

MoT架构的推理瓶颈主要action generate 单向依赖于vlm的kv cache（cosmos 除外），理论上可以做两种流水线并行。

一种是模块级的并行，AE在前台依赖于旧的VLM KV cache进行动作生成，而后台则在根据最新的观测更新VLM KV cache。理论上可以把两阶段进行一定的重叠，以减少端到端延迟，且因为两阶段的性能负载特性不一样，可以更好地利用硬件资源。

另一种是更细粒度的并行，以pi系列为例，AE的每一层都依赖于VLM计算的KV，而且只依赖KV不需要计算attention和FFN，所以理论上在VLM的第一层做完KV的映射后，就可以开始AE的第一层计算了，理论上岂不是能把AE的计算和VLM的计算完全重叠起来？

![alt text](vla_pipeline_comparison.svg)

#### background



#### motivation
##### bottleneck & observation


### designs




### v0.0 新增主线
[](./idea2_探索性实验.md)

### v0.0 新增主线
对v3进行进一步地探索，做出最好的效果！

<details>
  <summary>实验方案以及结果</summary>
  
  - 实验要求：无
  - 实验1：无依赖，高优先级
    -  背景：
    -  目标：
    -  设计：
    -  预期结果：
  - 实验2：
    - 背景：
    - 设计：
    - 目标：
    - 预期结果：
  - 实验3：
    - 背景：
    - 设计：
    - 目标：
  - 实验4：无依赖，高优先级
    -  背景：
    -  目标：
    -  设计：
    -  预期结果：

### v0 实验结论


