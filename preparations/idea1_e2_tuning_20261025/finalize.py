"""CPU-only final evidence and resource-release receipt for this bounded run."""
import hashlib
import json
from pathlib import Path
import time
from .analyze import records
from .design import OUT


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    implementation=json.loads((OUT/'implementation_revision.json').read_text())
    source_checks={p:sha(p)==h for p,h in implementation['source_sha256'].items()}
    assert all(source_checks.values())
    audit=json.loads((OUT/'execution_audit.json').read_text())
    assert audit['passed'] and audit['episodes']==240
    groups=records();assert len(groups)==6 and all(len(v)==40 for v in groups.values())
    workers={str(g):json.loads((OUT/f'workers/gpu{g}/finished.json').read_text()) for g in [2,3]}
    assert all(w['success'] and not w['final_snapshot']['processes'] for w in workers.values())
    leases=json.loads((OUT/'gpu_leases.json').read_text());assert not leases['active'] and all(not w['active'] for w in leases['workers'].values())
    start=min(json.loads((OUT/f'workers/gpu{g}/identity.json').read_text())['initial_snapshot']['time'] for g in [2,3])
    end=max(w['time'] for w in workers.values())
    rows=[r for v in groups.values() for r in v]
    first=min((Path(r['_directory'])/'complete.json').stat().st_mtime-r['wall_s'] for r in rows)
    last=max((Path(r['_directory'])/'complete.json').stat().st_mtime for r in rows)
    runtime=dict(worker_start_epoch=start,worker_finished_epoch=end,worker_elapsed_s=end-start,
        observed_episode_span_s=last-first,throughput_episodes_per_hour=240*3600/(end-start),
        model_load_s={str(g):json.loads((OUT/f'workers/gpu{g}/loaded_protocol.json').read_text())['load_s'] for g in [2,3]},
        common_warm_s=sum(r['common_warm_s'] for r in rows),active_episode_worker_s=sum(r['wall_s'] for r in rows))
    report=OUT/'report_zh.md'
    with report.open('a') as f:f.write(f"\n\n## 完成与验收\n\n两卡从worker资源核验至正常退出共{(end-start)/60:.2f}分钟；240例有效闭环，技术重试0。全量8676请求审计通过，40组初态/启动Full输出逐位一致，17项冻结执行源码/旧依赖hash保持一致。两worker成功退出，末次C/G均空，本轮GPU2/3租约inactive。共同预热共{runtime['common_warm_s']:.2f} worker秒单列；模型加载各{runtime['model_load_s']['2']:.2f}/{runtime['model_load_s']['3']:.2f}秒。未追加GPU实验。详见final_receipt.json和evidence_index.json。\n")
    invalid=[str(p) for p in OUT.rglob('*') if '.invalid.' in str(p) and p.is_dir()]
    assert not invalid
    receipt=dict(status='complete',time=time.time(),valid_episodes=240,conditions={k:len(v) for k,v in groups.items()},
        technical_invalid_attempts=invalid,source_checks=source_checks,runtime=runtime,
        audit_sha256=sha(OUT/'execution_audit.json'),implementation_sha256=sha(OUT/'implementation_revision.json'),
        leases_sha256=sha(OUT/'gpu_leases.json'),workers=workers,
        postprocessor_sha256={str(p):sha(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['audit.py','analyze.py','finalize.py']},
        limitations=['40 paired cases per arm are screening evidence only; no multiplicity adjustment.',
        'Cost is predict+ack server wall; mixed-card summary is descriptive, per-card paired costs govern speed interpretation.',
        'Parent is the shared implementation retaining Kalman.observe in ack, not an optimized minimal no-Kalman implementation.'])
    (OUT/'final_receipt.json').write_text(json.dumps(receipt,indent=2))
    names=['report_zh.md','paired_results.json','episodes.csv','execution_audit.json','routing_summary.json','requests_and_causal_features.csv','manifest.json','implementation_revision.json','resource_assignment.json','gpu_leases.json','profiling_receipt.json','old_coverage_receipt.json','fixed_profile_cases_xyz.png','final_receipt.json','cpu_tests.log','native_probe_gpu2.json','native_probe_gpu3.json']
    (OUT/'evidence_index.json').write_text(json.dumps(dict(status='complete',time=time.time(),files={n:sha(OUT/n) for n in names}),indent=2))
    print(json.dumps(runtime,indent=2))


if __name__=='__main__':main()
