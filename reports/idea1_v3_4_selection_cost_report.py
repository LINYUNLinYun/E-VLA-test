"""Publish measured outcomes, keeping graph and no-graph scopes separate."""
import csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/idea1_v3_4_selection_cost'

def read(name):
 with (OUT/name).open() as stream:return list(csv.DictReader(stream))


def main():
 graph=json.loads((OUT/'batch200_graph/complete.json').read_text())
 batch=json.loads((OUT/'analysis.json').read_text())
 summaries=read('summary.csv');pairs=read('paired.csv')
 prep=read('batch200_graph/preparation.csv')
 checks=json.loads((OUT/'benchmark/checks.json').read_text())['checks']
 graph_checks=json.loads((OUT/'batch200_graph/checks.json').read_text())['checks']
 assert len(summaries)==8 and sum(int(r['n']) for r in summaries)==64
 assert all(c['gradient_vs_eager_batch_max_abs']==0 for c in graph_checks)
 lines=['# 选层开销：批量反传、组合评分及200ms图测试','',
 '四方向一批加CUDA Graph仍未达到200ms：8个真实输入、80次热测，均值371.90ms，P95 380.78ms，最大384.90ms。图保持了批量版的数值结果；批量版相对原串行反传的既定一致性核验未通过。', '',
 '沿用π0、Full-D10可微重放、原四组投影seed340610、17层门控及间隔跳层约束。可微前向仅共享一次。各数据行均为选层成本，包含当前输入/噪声更新、可微前向、反传、梯度送CPU及原层RMS评分/选层；不包含前面的原生Full推理、视觉处理、验证或RPC。', '',
 '|口径|均值ms|P95 ms|热测数|峰值allocated GiB|数值结论|', '|---|---:|---:|---:|---:|---|']
 for b,label in [('1','无图，四次串行'),('2','无图，2+2'),('4','无图，四方向一批')]:
  x=batch['batch'][b]
  verdict='基准' if b=='1' else '梯度/层分数8/8未过容差'
  lines.append(f"|{label}|{x['total_ms']['mean']:.2f}|{x['total_ms']['p95']:.2f}|{x['total_ms']['n']}|{x['peak_allocated_mib']/1024:.2f}|{verdict}|")
 lines.append(f"|有图，四方向一批|{graph['mean_ms']:.2f}|{graph['p95_ms']:.2f}|{graph['n']}|{max(float(r['peak_allocated_mib']) for r in prep)/1024:.2f}|与无图批量版相同；相对串行仍未通过|")
 lines += ['',
 '历史约455ms是含局部CUDA Graph的四次串行热态结果，不作无图基线。本轮无图串行为2019.50ms；2+2降32.88%，四方向一批降54.95%。有图分支按用户后续授权单独追加，固定输入相同，测量阶段和顺序不同；371.90ms是本轮实际绝对时延，不把历史455ms当同轮配对因果对照。', '',
 '无图四方向一批分解：前向307.95ms、反向600.23ms，完整909.80ms。简单开图后的371.90ms仍需约1.86倍加速才能到200ms；本轮未实现该目标。不能用跨请求摊销替代单次选层时延。', '',
 f"有图每输入捕获含3次预热平均{np.mean([float(r['capture_with_warmups_s']) for r in prep]):.2f}秒，热测前另20次回放；模型加载{graph['model_load_s']:.2f}秒，整次快速图探测{graph['total_probe_wall_s']:.2f}秒。捕获准备成本未计入热态；峰值reserved {max(float(r['peak_reserved_mib']) for r in prep)/1024:.2f}GiB。无图首次选层调用分别约2.09/1.47/0.99秒，借用模型及常量准备约0.5ms，均保留原始明细。", '',
 '既定梯度/层分数容差atol=1e-4、rtol=1e-3，raw容差atol=rtol=1e-5。两种无图批量方式的梯度和分数都在8/8输入未通过；2+2的V12/V14选层16/16相同，四方向一批15/16相同。四方向一批变化发生在Goal task01/init14当前请求6的V14：串行[5,12,15,17]→批量[5,10,12,17]。图版相对无图批量版raw完全相同、梯度最大差0；该图优化本身没有新增偏差，但未消除批量数值差异。', '',
 '## 组合评分闭环','',
 '4 suites×task0/2×init19，每种原配置8例；baseline和combination共64个有效回合。V12/V14、周期3/5、Full-D10/Fast-D2、H50/R5、tau=.10、t=.10/.05、夹爪保护和外推规则均保持。两组都用无图串行四VJP、每个实际Full刷新，不叠加批量或降低刷新频率。', '',
 '|配置|baseline|组合评分|恢复/新增失败（严格配对）|严格n|', '|---|---:|---:|---:|---:|']
 for a in [r for r in summaries if r['variant']=='baseline']:
  b=next(r for r in summaries if r['base_id']==a['base_id'] and r['variant']=='combination')
  pair=next(r for r in pairs if r['base']==a['condition'] and r['candidate']==b['condition'] and r['scope']=='strict')
  lines.append(f"|V{a['budget']}/P{a['period']}|{a['success']}/8|{b['success']}/8|{pair['recovered']}/{pair['new_failures']}|{pair['n']}|")
 lines += ['',
 '组合目标为sqrt(mean_q((sum_l g[q,l])²))，穷举全部合法组合并按原词典序处理平局；仍为一阶近似，未解决非线性交互。', '',
 '|跳层数|原CPU评分ms|组合CPU评分ms|', '|---|---:|---:|']
 for k in ('4','6'):
  r=batch['scoring'][k]
  lines.append(f"|{k}|{r['layer_rms_sum']['mean']:.4f}|{r['signed_sum_rms']['mean']:.4f}|")
 anomaly=json.loads((OUT/'prefix_checks.json').read_text())['anomalies']
 lines += ['',
 '评分为同一组已算好的带符号梯度，每输入每方法100次纯CPU交替测量；没有新增反传。评分增加约0.25ms，远小于反传开销，未降低选层成本。闭环仅8例/配置，是探索小批；不能将个别配置多成功1例视为总体提升。自由轨迹耗时另见summary.csv，不作同输入因果加速证明。', '',
 f'逐回合核验原评分/组合最低分、四VJP计数、Full/Fast路由、验证、动作执行及每Full残差历史刷新。32项共同前缀检查异常{anomaly}项；有效失败全部保留，未重跑到成功。闭环先在GPU0运行，按用户授权后使用GPU0/1/3；新增服务均通过8真实输入跨卡数值核验，同一case的条件保持在同卡。', '',
 '降低选层刷新频率的第二项按用户要求暂停，已产生记录保留，不纳入本轮完整比较。为优先完成200ms探测，GPU0在回合/场景边界移交给独立图测试；其他卡继续质量对照。图测试完成后已释放GPU0。', '',
 '原始数据：', '',
 '- `results/idea1_v3_4_selection_cost/benchmark/`：无图逐次时延、梯度/分数/选层核验、准备成本和显存。',
 '- `results/idea1_v3_4_selection_cost/batch200_graph/`：图版80次热测、捕获准备、显存、逐方向误差和环境/源码hash。',
 '- `results/idea1_v3_4_selection_cost/summary.csv`、`paired.csv`、`requests.csv`、`audit.json`：64例闭环、逐请求耗时及审计。',
 '- `manifest.json`保留原冻结科学配置；`scope_change.json`、`execution_amendment.json`及`batch200/graph_authorization.json`记录用户调整后的范围与执行方式。', '',
 '[原冻结方案](idea1_v3_4_selection_cost_plan.md)', '']
 target=ROOT/'docs/experiments/idea1/idea1_v3_4_selection_cost_report.md'
 target.write_text('\n'.join(lines))
 (OUT/'report_receipt.json').write_text(json.dumps(dict(report=str(target),created=time.time()),indent=2)+'\n')
 print(str(target))


if __name__=='__main__':main()
