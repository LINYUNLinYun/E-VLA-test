"""Post-run costs excluding cold capture requests, without modifying the policy."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from .common import ROOT,OUT
from preparations.idea1_v3_4.analyze import write_csv

def main():
    assert json.loads((OUT/'audit.json').read_text())['passed']
    requests=list(csv.DictReader((OUT/'requests.csv').open()));summary=[]
    for name in dict.fromkeys(r['condition'] for r in requests):
        rows=[r for r in requests if r['condition']==name];cold=[r for r in rows if float(r['selector_capture_ms'])>0];hot=[r for r in rows if float(r['selector_capture_ms'])==0]
        entry=dict(condition=name,requests=len(rows),hot_requests=len(hot),cold_requests=len(cold),capture_mean_ms=float(np.mean([float(r['selector_capture_ms']) for r in cold])))
        for metric in ['selector_hot_ms','policy_wall_ms','rpc_ms','vision_ms','vlm_ms','ae_ms']:
            values=[float(r[metric]) for r in hot];entry[metric+'_mean']=float(np.mean(values));entry[metric+'_p95']=float(np.percentile(values,95))
        groups={}
        for r in rows:groups.setdefault((r['suite'],r['task'],r['init']),[]).append(r)
        changed=[];replaced=[]
        for group in groups.values():
            group.sort(key=lambda r:int(r['replan']))
            for a,b in zip(group,group[1:]):
                sa,sb=set(json.loads(a['skip'])),set(json.loads(b['skip']));changed.append(sa!=sb);replaced.append(len(sa-sb))
        entry.update(mask_change_fraction=float(np.mean(changed)),mean_layers_replaced=float(np.mean(replaced)));summary.append(entry)
    write_csv(OUT/'steady_costs.csv',summary)
    (OUT/'steady_costs_audit.json').write_text(json.dumps(dict(passed=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),criterion='selector_capture_ms == 0',time=time.time()),indent=2)+'\n')
    lines=['','## 排除首次捕获请求的热态成本','','每个episode首次捕获的请求完整保留在主表和episode墙钟中；下表仅统计selector_capture_ms=0的请求，避免把“扣去捕获时间”与完整热态请求混用。','','|条件|热态请求数|选层均值/P95 ms|策略均值/P95 ms|RPC均值/P95 ms|组合变化率|每轮替换层数|','|---|---|---|---|---|---|---|']
    for r in summary:lines.append(f"|{r['condition']}|{r['hot_requests']}|{r['selector_hot_ms_mean']:.2f}/{r['selector_hot_ms_p95']:.2f}|{r['policy_wall_ms_mean']:.2f}/{r['policy_wall_ms_p95']:.2f}|{r['rpc_ms_mean']:.2f}/{r['rpc_ms_p95']:.2f}|{r['mask_change_fraction']:.1%}|{r['mean_layers_replaced']:.2f}|")
    lines+=['','组合变化率指四层集合中至少一层改变，并不意味着四层全部替换。此统计不能单独证明频繁换层导致失败。']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_online_time_weights_report.md'
    with report.open('a') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
