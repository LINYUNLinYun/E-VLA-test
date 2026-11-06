import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

ROOT = Path('/root/nfs/embodied/benchmark')
OUT = ROOT / 'results/idea1_v3_4_E2_selection_200'
REPORT = ROOT / 'docs/experiments/idea1/idea1_v3_4_E2_selection_200_report.md'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_rows(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def main():
    complete = read(OUT / 'complete.json')
    assert read(OUT / 'audit.json')['passed']
    manifest = read(OUT / 'manifest.json')
    episodes = csv_rows('episodes.csv')
    fresh = [r for r in episodes if r['reused'] == 'False']
    assert len(fresh) == 1440 and len(episodes) == 1760
    workers = []
    for wd in sorted((OUT / 'workers').glob('gpu*')):
        worker = dict(
            gpu=int(wd.name[3:]),
            startup=read(wd / 'startup.json'),
            final_state=read(wd / 'state.json'),
            exits=read(wd / 'exits.json'),
            identity=read(wd / 'identity.json'),
            compatibility_gate=read(wd / 'compatibility_gate.json')['passed'],
            isolation_gate=read(wd / 'isolation_gate.json')['passed'],
        )
        assert worker['final_state']['status'] == 'complete'
        assert worker['exits']['service'] == -15
        assert all(c == 0 for c in worker['exits']['evaluations'])
        workers.append(worker)
    start = min(w['startup']['time'] for w in workers)
    end = max(w['final_state']['time'] for w in workers)
    wall = end - start
    process = read(OUT / 'process.json')
    recovery = read(OUT / 'aggregation_recovery.json') if (OUT / 'aggregation_recovery.json').exists() else None
    controller_end = read(OUT/'failed.json')['time'] if recovery is not None else complete['time']
    groups = read(OUT / 'queue.json')['groups']
    assert len(groups) == 60 and all(g['state'] == 'complete' for g in groups)
    phases = {}
    for phase in ('random40', 'expand200'):
        gs = [g for g in groups if g['phase'] == phase]
        phase_start = min(g['started'] for g in gs)
        phase_end = max(g['finished'] for g in gs)
        phase_n = sum(1 for r in fresh if ('random' in r['condition']) == (phase == 'random40'))
        phases[phase] = dict(
            groups=len(gs), new_episodes=phase_n, start=phase_start,
            end=phase_end, wall_s=phase_end-phase_start,
            episodes_per_hour=phase_n * 3600 / (phase_end-phase_start),
        )
    attempts = [dict(path=str(f.relative_to(OUT)), **read(f))
                for f in (OUT / 'technical_attempts').rglob('*.json')]
    assert len(attempts) == 1440
    retries = [a for a in attempts if a['attempts'] > 1]
    errors = [str(p.relative_to(OUT)) for p in (OUT / 'workers').glob('gpu*/error_attempt*.json')]
    throughput = dict(
        new_episodes=len(fresh), reused_episodes=320,
        gpu_count=len(workers), environment_actors_per_gpu=[w['startup']['actors'] for w in workers],
        evaluation_start=start, evaluation_end=end, evaluation_wall_s=wall,
        controller_wall_s=controller_end - process['time'],
        execution_and_audit_wall_s=complete['time'] - process['time'],
        evaluation_episodes_per_hour=len(fresh)*3600/wall,
        new_executed_actions=sum(int(r['actions']) for r in fresh),
        new_predictions=sum(int(r['predictions']) for r in fresh),
        sum_overlapping_episode_wall_s=sum(float(r['wall_s']) for r in fresh),
        mean_new_episode_cold_capture_s=statistics.mean(float(r['cold_s']) for r in fresh),
        service_startup_s=[w['startup']['startup_s'] for w in workers],
        phases=phases, technical_retried_episodes=len(retries), worker_errors=len(errors),
        timing_scope='Real elapsed evaluation window across three GPUs and six environments; includes episode capture and environment startup, excludes initial service/model/preflight preparation. Controller wall includes preparation and original aggregation attempt. Execution-and-audit wall additionally includes offline aggregation recovery. Episode walls overlap and must not be added as elapsed runtime.',
    )
    throughput['new_executed_actions_per_s'] = throughput['new_executed_actions']/wall
    throughput['new_predictions_per_s'] = throughput['new_predictions']/wall
    write(OUT / 'throughput.json', throughput)
    write(OUT / 'execution_metadata.json', dict(
        completed_time=complete['time'], postprocessed_time=time.time(),
        manifest_hash=manifest['manifest_hash'], workers=workers,
        technical_attempts=len(attempts), retries=retries, worker_error_files=errors,
        aggregation_recovery=recovery,
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_data_were_not_rerun=True, foreign_gpu_untouched=True,
    ))
    costs = {p.stem: read(p) for p in (OUT / 'condition_audits').glob('*.json')}
    lines = [
        '', '## 实际执行与成本', '',
        f"本轮新完成1440回合，复用320回合；3张GPU、每卡2个独立环境，共6环境。实际评测窗口{wall/60:.2f}分钟，吞吐{throughput['evaluation_episodes_per_hour']:.1f}回合/小时；从控制器启动至最终审计完成{throughput['execution_and_audit_wall_s']/60:.2f}分钟。模型/服务与兼容检查准备各卡分别耗时" + '/'.join(f'{v:.2f}' for v in throughput['service_startup_s']) + '秒。评测窗口包含各回合冷启动与环境建立；并发回合wall之和不作为总耗时。',
        f"技术重试{len(retries)}回合、worker错误{len(errors)}项。全部已发起有效回合保留，官方初态及控制协议核对通过。", '',
        '|方法|预算|补偿|稳态完整周期均摊policy ms/请求|平均每回合selector冷捕获秒|',
        '|---|---|---|---|---|',
    ]
    for cfg in manifest['conditions']:
        a = costs[cfg['id']]
        lines.append(f"|{cfg['method']}|V{cfg['budget']}|{cfg['compensation']}|{a['cycle_policy_ms']:.2f}|{a['mean_cold_s']:.2f}|")
    lines += [
        '', '表中为各自自由轨迹上，排除首个完整周期之后，按完整Full/Fast/Fast周期计时的policy均值；每个扩样条件含40条既有单环境记录和160条本轮记录。policy计时不含RPC排队，两个环境并发时RPC延迟不能当作单机器人延迟。成功轨迹长度不同、输入不同，以上成本是部署量级描述，不构成相同输入的因果加速对照。首周期与冷捕获仍计入实际总吞吐。',
        '', '## 解释边界', '',
        '扩样使用原始一次正权Full-D10 Jacobian，每Full评分一次；本轮没有扩样补偿误差方向评分、2/3/4次投影或隔三周期评分方案。原40已参与此前探索，新160单列以便检查新场景是否延续趋势。',
        '', '四方法的公平直接比较使用相同40个case，见随机补偿报告。逐次随机加外推的成功率点估计为V14 36/40、V12 34/40，相对一次Jacobian的配对差值为+5pp和+10pp，但探索95%区间均跨零。单一预设随机根seed不足以证明随机种子总体更优，随机方法本轮也没有扩到200。',
        '', '全局固定V12组合[2,3,5,14,16,17]在所有case使用同一mask；每回合固定随机会按case抽不同mask，两者定义不同。200case参考为无补偿134/200、外推169/200，属于同case旧实验复用。',
    ]
    if recovery is not None:
        lines += ['', '离线汇总首次因旧全局固定记录缺少route.selected_skip字段而中断，原始错误保留在failed.json。review_v1/recover_summary.py仅对400条已固定引用，从result.config.skip读取层编号，并校验静态组合、D10/D2周期与identity.config一致；其余原冻结审计和配对检查均保留。独立适配后汇总通过，没有改动冻结实验源码或重跑评测。详见aggregation_recovery.json。']
    marker = '\n## 实际执行与成本\n'
    body = REPORT.read_text()
    assert marker not in body, 'Postprocessing already appended'
    REPORT.write_text(body + '\n'.join(lines) + '\n')
    print(json.dumps(throughput, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
