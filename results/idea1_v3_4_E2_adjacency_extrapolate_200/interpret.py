import csv,json,shutil,time
from pathlib import Path

ROOT=Path('/root/nfs/embodied/benchmark')
OUT=Path(__file__).parent
REPORT=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_adjacency_extrapolate_200_report.md'

def main():
    assert json.loads((OUT/'audit.json').read_text())['passed']
    analysis=OUT/'analysis';analysis.mkdir(exist_ok=True)
    assert not (analysis/'interpretation_complete.json').exists()
    shutil.copyfile(REPORT,analysis/'automatic_report.md')
    pairs=list(csv.DictReader((OUT/'paired.csv').open()))
    sensitivity=list(csv.DictReader((OUT/'paired_sensitivity.csv').open()))
    primary=[r for r in pairs if r['baseline']=='v12_jacobian_extrapolate']
    assert len(primary)==2 and all(float(r['ci_low_pp'])>0 for r in primary)
    assert all(float(r['ci_low_pp'])>0 for r in sensitivity if r['baseline']=='v12_jacobian_extrapolate')
    a=json.loads((OUT/'prefix_audit.json').read_text())
    unique=sorted({tuple(e['case'][k] for k in ('suite','task','init')) for e in a['baseline_exceptions']})
    assert len(unique)==6
    body=REPORT.read_text();marker='\n|基准→候选|严格配对n|恢复/新增失败|task等权差值pp|探索95%区间pp|'
    assert marker in body
    text='''
补偿前后成功率明显提升：V14无上限63.5%→90.5%、间隔60%→89%；V12无上限39.5%→87.5%、间隔34.5%→89.5%。同规则的补偿前后严格配对为V14两项200例、V12无上限199例/间隔200例，探索区间均为正。

V12相对旧max2外推75%的改善较明确：无上限的严格task等权提升12.39pp（n194，探索95%区间7.22至17.56pp），间隔跳提升14.83pp（n195，区间8.72至21.83pp）；保留完整200例的敏感性方向也一致。V14相对旧max2外推88%的差异仍不确定，两项区间均跨零。

V12间隔89.5%为本轮最高点估计，无上限为87.5%；两者配对区间仍跨零。相对全局固定V12外推84.5%，间隔的完整200点差为+5pp，但严格task等权差+4.5pp的区间为-1.28至10.61pp，无上限相对固定的区间也跨零。因此尚不能确认新规则优于该固定组合。
'''
    body=body.replace(marker,'\n'+text+marker,1)
    body+='''
## 前缀异常的独立范围

21条基线异常记录对应6个独立case：Spatial task0的init19–23在四个新条件中重复计为20条，另有Goal task1/init22的V12无上限补偿一条。前者与上轮相同，首观测及梯度匹配、原生首Full动作有差异，根因尚未确认；并未放宽容差或重跑有效回合。

新增Goal例仅首Fast观测hash不同。与旧max2、旧无补偿及新间隔规则相比，其首Full raw与actions逐元素完全一致，前5步动作及前6条EEF位置/姿态也完全一致；具体观测差异原因未确认，记录见extra_prefix_exception_detail.json。该例保留完整描述结果，从相关严格配对排除。

因此，新规则互比为V14 n200、V12 n199；跨max2/余弦/固定的严格配对为n194或195；同规则补偿前后为n199或200。主表完整200成功率与严格task等权区间的样本/估计口径不同，不能混用。所有有效结果均保留，全部200敏感性见paired_sensitivity.csv。

本轮三卡服务已正常结束，800条全部完成，技术重试0。稳态周期均摊约120–122ms/请求，评分所在Full约262ms；补偿与新规则未消除一次Full-D10反传的主要成本。上述时延不含RPC排队与冷捕获，成本表和实际wall已分别列出。
'''
    REPORT.write_text(body)
    (analysis/'interpretation_complete.json').write_text(json.dumps(dict(time=time.time(),prefix_unique_cases=unique,valid_outcomes_retained=800,primary_v12_positive_strict_and_sensitivity=True),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(report=str(REPORT),new_episodes=800,unique_prefix_exception_cases=6,technical_retries=0),ensure_ascii=False))

if __name__=='__main__':main()
