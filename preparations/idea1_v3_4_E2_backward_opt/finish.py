"""Summarize measured runs; do not turn failed numerical checks into adoption."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from . import probe as p

def read(path):return json.loads(path.read_text())
def stats(v):return dict(n=len(v),mean=float(np.mean(v)),p95=float(np.percentile(v,95)),max=float(np.max(v)))
def main():
    out=p.OUT;det=read(out/'deterministic/summary.json');adv=read(out/'advanced/summary.json')
    assert det['passed'] and adv['passed']
    numerical={}
    for branch in ['deterministic','advanced','fusion','fusion_preserve','batched']:
        c=read(out/branch/'checks.json')
        numerical[branch]=dict(n=len(c),passed=all(x['passed'] for x in c),raw_max_abs=max(x['raw_max_abs'] for x in c),gradient_relative_error_max=max(x['gradient_relative_error'] for x in c),mask_changes=sum(not x['mask_ok'] for x in c),translation_max_mm=max(x['translation_max_mm'] for x in c))
    assert numerical['deterministic']['n']==32 and numerical['advanced']['n']==32
    for n in ['1','4']:assert adv['summary'][n]['n']==80
    assert all(x['n']==80 for x in det['summary'].values())
    costs={}
    for branch in ['fusion','fusion_preserve','batched']:costs[branch]=read(out/branch/'summary.json')
    prep=read(out/'advanced/preparation.json')
    decomp={str(n):{k:float(np.mean([x[k] for x in prep if x['projections']==n])) for k in ['forward_ms','backward_ms','prepare_ms']} for n in [1,4]}
    r=list(csv.DictReader((out/'advanced/contention.csv').open()))
    contention={m:{k:float(np.mean([float(x[k]) for x in r if x['mode']==m])) for k in ['backward_complete_ms','next_forward_complete_ms','pair_complete_ms']} for m in ['serial','overlap']}
    old_masks=[]
    for x in read(out/'deterministic/checks.json'):
        if x['test']!='graph':continue
        with np.load(p.ROOT/'results/idea1_v3_4_E2_single_action/bf16'/f"sample{x['id']:03d}"/'arrays.npz') as z:chosen=(p.mask(np.linalg.norm(z['project1_step1_gradient'],axis=0))+1).tolist()
        old_masks.append(dict(id=x['id'],old_mask=chosen,new_mask=x['mask_reference'],same=chosen==x['mask_reference']))
    trace=read(out/'eager_trace.json')
    kernels=[v for v in trace['traceEvents'] if v.get('cat')=='kernel']
    profile=dict(kernel_count=len(kernels),kernel_time_ms=sum(v['dur'] for v in kernels)/1000,note='one instrumented call; profiler wall time is not used as latency')
    audit=dict(passed=True,accepted=['deterministic graph1','deterministic serial graph4'],numerical=numerical,latency=det['summary'],split_graph=adv['summary'],other_costs=costs,decomposition=decomp,contention=contention,old_mask_comparison=old_masks,profile=profile,time=time.time())
    p.dump(out/'audit.json',audit)
    sources=list(Path(__file__).parent.glob('*.py'))+[p.ROOT/'preparations/idea1_v3_4_E2_E4_single/torch_model.py',p.ROOT/'preparations/idea1_v3_4_E2_single_action/common.py',p.ROOT/'docs/experiments/idea1/idea1_v3_4_E2_backward_opt_plan.md']
    h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    p.dump(out/'manifest.json',dict(sources={str(s.relative_to(p.ROOT)):h(s) for s in sources},cases=[dict(id=x['id'],case=x['case'],input= str(p.OLD/'inputs'/f"{p.key(x['current'])}.pt"),sha256=h(p.OLD/'inputs'/f"{p.key(x['current'])}.pt")) for x in p.pairs()],environment=read(out/'deterministic/environment.json'),selection_sha256=h(p.OLD/'selection.json'),reference_root=str(p.OLD),time=time.time()))
    g1=det['summary']['graph'];e1=det['summary']['eager'];g4=adv['summary']['4']
    f=costs['fusion'];fp=costs['fusion_preserve'];b=costs['batched']
    n=numerical;stand=read(out/'advanced/standalone.json')
    report=f'''# E2 单次反传加速实测

2026-10-30。结论：159ms 不是当前 Jacobian 方法的计算下限。保持单动作、同一随机投影与门控定义，CUDA Graph 已将前向+一次 VJP+CPU 选层做到平均 {g1['mean_ms']:.2f}ms、P95 {g1['p95_ms']:.2f}ms。确定性普通 PyTorch 同轮 {e1['mean_ms']:.2f}ms，约 {e1['mean_ms']/g1['mean_ms']:.2f}×。四次投影完整耗时 {g4['mean_ms']:.2f}ms、P95 {g4['p95_ms']:.2f}ms。先前 158.6ms 是历史普通 BF16 测量，不能用跨轮数据代替本轮配对。

## 条件与口径

一张独占 RTX4090，PyTorch 2.7.1+cu126，冻结原导出模型，BF16 权重/激活、FP32 RMS，D2/H50/R5。第一步动作七维，17 个 VLM 门控，seed340710 的原一/四组 ±1 权重；skip4、最长连续两层。4 套件各 task0/2 的八个真实请求，每方法每请求十次热测。原模型及历史实验未修改。输入、state、prompt、噪声身份见 manifest。

总时间包含 VLM+完整两轮 AE/FM、VJP、梯度送 CPU、约束选层和动作输出，边界同步；Graph 输入缓冲更新也计时。固定视觉嵌入已就绪，不含视觉编码、RPC、加载或捕获。故这些数字不能直接当作机器人完整服务时延，也不能直接与用户约100ms的原生推理相减。

## 热态结果

|配置|测量次数|平均 ms|P95 ms|数值判断|
|---|---:|---:|---:|---|
|普通 PyTorch，确定性，一次投影|80|{e1['mean_ms']:.2f}|{e1['p95_ms']:.2f}|基准|
|完整 CUDA Graph，确定性，一次投影|80|{g1['mean_ms']:.2f}|{g1['p95_ms']:.2f}|通过|
|拆分前后向 Graph，确定性，一次投影|80|{adv['summary']['1']['mean_ms']:.2f}|{adv['summary']['1']['p95_ms']:.2f}|通过|
|拆分前后向 Graph，确定性，四次投影|80|{g4['mean_ms']:.2f}|{g4['p95_ms']:.2f}|通过|
|局部 RMS/RoPE compile + Graph，一次投影|80|{f['mean_ms']:.2f}|{f['p95_ms']:.2f}|未通过，不采用|
|局部 compile 保留精度截断 + Graph|80|{fp['mean_ms']:.2f}|{fp['p95_ms']:.2f}|{'通过' if fp['passed'] else '未通过，不采用'}|
|四投影 batched VJP + Graph|80|{b['mean_ms']:.2f}|{b['p95_ms']:.2f}|{'通过' if b['passed'] else '未通过，不采用'}|

拆分图的平均前向 {decomp['1']['forward_ms']:.2f}ms、单 VJP {decomp['1']['backward_ms']:.2f}ms。四次反传合计 {decomp['4']['backward_ms']:.2f}ms。捕获含预热平均单投影 {decomp['1']['prepare_ms']/1000:.2f}s、四投影 {decomp['4']['prepare_ms']/1000:.2f}s；这是每个新 shape/执行配置的准备成本，表中是热态。当前实现逐请求释放图，生产服务需按 shape 预热并采用有界缓存；不能把冷捕获塞进实时请求。

## 为什么图优化有效

一次带 profiler 的前后向共 {profile['kernel_count']} 个 GPU kernel，累积 kernel 时间约 {profile['kernel_time_ms']:.2f}ms，矩阵乘约占自 CUDA 时间59%，attention backward约12%。普通实现包含大量 norm、RoPE、cast、切片反传、逐元素与小归约。完整 Graph 录制这些操作后一次发射，跳过逐算子的 Python/C++/驱动启动开销，未减少反传数量。profiler 会扰动墙钟，采用独立热测做速度结论；不把 profiler 的长墙钟当作正式延迟。

最初非确定性试探为普通191.72ms/Graph53.19ms，但重复 eager 自身梯度波动也超过既定容差。启用 deterministic algorithms 与 CUBLAS_WORKSPACE_CONFIG=:4096:8 后，普通与 Graph 的32项检查 raw 和梯度最大误差均为0，选层相同；拆分一/四投影另32项也全部为0，包括更换并恢复噪声。与上一轮八个 BF16 请求的一次投影选层也全部相同。53ms 分支保留作探索值，推荐引用64ms分支。

## 算子融合与反传 batching

仅局部编译 RMS/RoPE：首次调用约13.4秒，热态约56.7ms，比64ms再快约11%。但全部 raw 最大偏差 {n['fusion']['raw_max_abs']:.6f}，门控梯度最大相对误差 {n['fusion']['gradient_relative_error_max']:.2%}，8请求的16项检查里选层改变 {n['fusion']['mask_changes']} 项；首五步平移误差最大 {n['fusion']['translation_max_mm']:.3f}mm。没有放宽容差来采用它。Inductor 融合低精度操作可能省略 eager 中间截断；专门复测保留截断选项后，raw 最大差 {n['fusion_preserve']['raw_max_abs']:.6f}、梯度相对误差最大 {n['fusion_preserve']['gradient_relative_error_max']:.2%}、16项检查中选层改变 {n['fusion_preserve']['mask_changes']} 项，严格检查{'通过' if fp['passed'] else '仍未通过'}。

四个反传 seed 放到同一批：约 {b['mean_ms']:.2f}ms，比串行四次更快；但 attention backward、gelu、silu 等仍有 vmap fallback，梯度最大相对误差 {n['batched']['gradient_relative_error_max']:.2%}，16项检查中选层改变 {n['batched']['mask_changes']} 项，严格检查{'通过' if b['passed'] else '未通过'}。这轮采用标准不变，数值候选不冒充等价加速。未做完整模型长编译或手写 Triton backward。

## 流水线的收益与代价

固定一个请求的独立前后向图，在当前前向结束后，测“该请求反传+下一次无梯度前向”，20组交替串行/并发；下一次前向单独约 {stand['inference_ms']:.2f}ms。

|方式|反传完成 ms|下一前向完成 ms|两者都完成 ms|
|---|---:|---:|---:|
|串行|{contention['serial']['backward_complete_ms']:.2f}|{contention['serial']['next_forward_complete_ms']:.2f}|{contention['serial']['pair_complete_ms']:.2f}|
|同卡双 stream，前向高优先级|{contention['overlap']['backward_complete_ms']:.2f}|{contention['overlap']['next_forward_complete_ms']:.2f}|{contention['overlap']['pair_complete_ms']:.2f}|

成对完成时间减少约 {(1-contention['overlap']['pair_complete_ms']/contention['serial']['pair_complete_ms'])*100:.1f}%，但前向自身受资源争用而变慢，反传也延长。不能假设双 stream 可以将40ms反传完全隐藏。此为固定输入同卡争用微基准，不是闭环异步评测。

更适合的时序是先发当前动作，再利用机器人执行动作期间的 GPU 空档计算梯度，用于下一次重规划。它使用上一观测的选层信息，需要单独检查时效性和闭环效果；不能反过来给已经完成的当前前向选层。原生 Triton 无梯度前向不能自动供 PyTorch VJP 使用：如果保持原生服务另做一条影子梯度路径，新增计算仍是完整64ms；只有真正共用可微前向/激活，才可把新增成本记作约40ms反传。

## 当前建议

优先保留数值通过的 CUDA Graph 原型；一次投影已约64ms，四次投影也在当前小批热态内低于200ms。更低的在线预算宜采用动作执行空档、周期更新或观测变化触发；周期更新/复用属于算法时序改变，不能仅凭这轮速度给成功率结论。若每次选层成本 C、之后每次跳层节省 Δ，每 N 次更新一次的摊销约 C/N；应由实际 Δ 决定 N，而不是只看低于200ms。若要求每请求都额外只花10–20ms，当前严格通过的版本尚未达到。

本轮没有重跑已完成的闭环实验，也没有证明新时序闭环成功率。进一步原生前向接入和专用 VJP 需开发 backward 并保存所需激活，不能通过切换一个 graph 开关完成。

代码：preparations/idea1_v3_4_E2_backward_opt。逐次时延、数值检查、trace、环境与源文件 hash：results/idea1_v3_4_E2_backward_opt。

官方依据：[PyTorch 2.7 CUDA Graph 与 stream 语义](https://docs.pytorch.org/docs/2.7/notes/cuda.html#cuda-graphs)、[确定性计算](https://docs.pytorch.org/docs/2.7/notes/randomness.html)、[局部编译](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)。性能与数值结论来自本轮实测。
'''
    (p.ROOT/'docs/experiments/idea1/idea1_v3_4_E2_backward_opt_report.md').write_text(report)
    print(json.dumps(dict(audit_passed=True,numerical=numerical,decomp=decomp,latency=det['summary'],graph4=g4,contention=contention,other=costs)),flush=True)

if __name__=='__main__':main()
