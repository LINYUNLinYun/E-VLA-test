"""CPU-only final evidence; release this task's lease after clean worker exit."""
import json
from pathlib import Path
import time
from .analyze import OUT,read,sha,rows,stats


def main():
    worker=read(OUT/'workers/gpu3/finished.json')
    assert worker['success'] and not worker['final_snapshot']['processes']
    audit=read(OUT/'execution_audit.json');assert audit['passed'] and audit['episodes']==160
    groups=rows();assert len(groups)==4 and all(len(v)==40 for v in groups.values())
    revision=read(OUT/'implementation_revision.json');checks={p:sha(p)==h for p,h in revision['source_sha256'].items()};assert all(checks.values())
    manifest=read(OUT/'manifest.json');assert all(sha(x['path'])==x['sha256'] for x in manifest['inputs'])
    assert read(OUT/'native_probe_gpu3.json')['passed']
    lease=read(OUT/'gpu_leases.json');lease.update(active=False,phase='complete',released=time.time());lease['workers']['3'].update(active=False,released=time.time(),finished_receipt=str(OUT/'workers/gpu3/finished.json'))
    (OUT/'gpu_leases.json').write_text(json.dumps(lease,indent=2))
    invalid=[str(p) for p in OUT.rglob('technical_error.json')]
    identity=read(OUT/'workers/gpu3/identity.json');start=identity['initial_snapshot']['time'];end=worker['time'];loaded=read(OUT/'workers/gpu3/loaded_protocol.json')
    pairs=read(OUT/'paired_results.json')['paired'];ss={k:stats(v) for k,v in groups.items()}
    text=['','## 结果解释','']
    for mode in ['periodic','verified']:
        a,b=f'p4_{mode}',f'p2_{mode}';p=pairs[a+'_vs_'+b];ratio=p['policy_wall_per_action_ratio']
        ci=p['policy_ratio_task_stratified_bootstrap_ci95'];uncertain=ci[0]<=1<=ci[1]
        text.append(f"{mode}: P4 {ss[a]['successes']}/40、P2 {ss[b]['successes']}/40，配对{p['recoveries']}恢复/{p['new_failures']}新失败；同卡完整policy ms/action比{ratio:.4f}，观察成本变化{100*(ratio-1):+.2f}%，配对task分层95%区间[{ci[0]:.4f},{ci[1]:.4f}]。"+('区间包含1，未显示稳定降本。' if uncertain else '区间未包含1，但仍受固定suite条件顺序等限制。')+'不能仅凭Full次数减少断言等幅加速，也不能以这40例宣布总体非劣。')
        text.append('')
    text+=['## 完成与资源','',f"160/160有效回合，技术异常attempt记录{len(invalid)}；全量{audit['requests']}请求审计通过，40组初态/启动Full签名一致，{len(checks)}项冻结源码/依赖hash相同。GPU3 worker从资源核验至正常退出{(end-start)/60:.2f}分钟，模型加载{loaded['load_s']:.2f}秒，吞吐{160*3600/(end-start):.1f}有效episode/小时；公用warm共{sum(s['warm_s'] for s in ss.values()):.2f} worker秒另列。末次C/G为空，本轮租约inactive，GPU0/1/2未使用。无自动扩样或额外GPU实验。",'']
    report=OUT/'report_zh.md';report.write_text(report.read_text().split('\n## 结果解释')[0]+'\n'.join(text))
    receipt=dict(status='complete',time=time.time(),valid_episodes=160,technical_attempt_records=invalid,source_checks=checks,
        runtime=dict(start_epoch=start,finished_epoch=end,elapsed_s=end-start,service_load_s=loaded['load_s'],effective_episodes_per_hour=160*3600/(end-start)),
        worker_finished=worker,manifest_sha256=sha(OUT/'manifest.json'),audit_sha256=sha(OUT/'execution_audit.json'),lease_sha256=sha(OUT/'gpu_leases.json'),
        postprocessor_sha256={str(p):sha(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['analyze.py','finalize.py']},
        limitations=['40 paired screening cases, no multiplicity correction or deployment noninferiority claim.','Globally balanced but suite-fixed condition order; small cost differences are not established stable speedups.','Shared ack retains inherited Kalman.observe in all arms.'])
    (OUT/'final_receipt.json').write_text(json.dumps(receipt,indent=2))
    names=['report_zh.md','manifest.json','implementation_revision.json','resource_assignment.json','gpu_leases.json','final_receipt.json','execution_audit.json','routing_summary.json','requests.csv','behavior_comparisons.json','paired_results.json','profiling_receipt.json','cpu_tests.log','native_probe_gpu3.json']
    (OUT/'evidence_index.json').write_text(json.dumps(dict(status='complete',time=time.time(),files={n:sha(OUT/n) for n in names}),indent=2))
    print(json.dumps(receipt['runtime'],indent=2))


if __name__=='__main__':main()
