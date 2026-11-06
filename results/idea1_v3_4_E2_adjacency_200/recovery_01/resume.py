"""Resume two incomplete episodes; preserve frozen experiment and prior evidence."""
import json,os,shutil,time,traceback
from pathlib import Path
from preparations.idea1_v3_4_E2_adjacency_200 import controller,finish
from preparations.idea1_v3_4_E2_adjacency_200.common import OUT,release
from preparations.idea1_v3.io import atomic

ARCHIVE=OUT/'recovery_01'

def correct_execution_accounting():
    raw=json.loads((OUT/'throughput.json').read_text())
    atomic(ARCHIVE/'automatic_finish_throughput_before_recovery_accounting.json',raw)
    prior=[json.loads(p.read_text()) for p in (ARCHIVE/'workers_before_interruption').glob('gpu*/startup.json')]
    begin=min(r['time'] for r in prior)
    end=max(w['state']['time'] for w in raw['workers'])
    wall=end-begin;oldwall=raw['evaluation_wall_s']
    oldepisode_rate=raw['episodes_per_hour'];oldactions_rate=raw['actions_per_s'];oldrequests_rate=raw['requests_per_s']
    raw['evaluation_wall_s']=wall
    raw['episodes_per_hour']=800*3600/wall
    raw['actions_per_s']=oldactions_rate*oldwall/wall
    raw['requests_per_s']=oldrequests_rate*oldwall/wall
    raw['prior_worker_startups']=prior
    raw['evaluation_start']=begin;raw['evaluation_end']=end
    raw['timing_scope']='End to end from initial services ready to recovered final completion, including unobserved interruption/downtime and recovery startup; not uninterrupted active-evaluation throughput.'
    raw['recovery_window_wall_s']=oldwall
    raw['initial_controller_start']=json.loads((ARCHIVE/'process.json').read_text())['time']
    raw['recovery_controller_start']=raw.pop('controller_start')
    raw['interruption']=dict(preserved_valid_episodes=798,recovered_missing_episodes=2,unobserved_exit_reason=True,prior_worker_evidence='recovery_01/workers_before_interruption',interruption_time_not_known_exactly=True)
    atomic(OUT/'throughput.json',raw)
    report=Path(json.loads((OUT/'complete.json').read_text())['report'])
    body=report.read_text()
    oldtext=f'实际闭环窗口{oldwall/60:.2f}分钟，{oldepisode_rate:.1f}回合/小时、{oldactions_rate:.2f}执行动作/秒、{oldrequests_rate:.2f}模型请求/秒；窗口含各回合冷启动、环境、排队与落盘，模型服务准备单列throughput.json。'
    assert oldtext in body
    newtext=f'从最初模型服务可用到恢复后全部完成的端到端窗口{wall/60:.2f}分钟，{raw["episodes_per_hour"]:.1f}回合/小时、{raw["actions_per_s"]:.2f}执行动作/秒、{raw["requests_per_s"]:.2f}模型请求/秒；包含中断等待与恢复启动，不作为连续活跃评测吞吐。原始服务准备及恢复窗口单列throughput.json。'
    body=body.replace(oldtext,newtext,1)
    body+='\n运行在798/800处发生进程中断；退出原因没有捕获，不能归因为模型OOM或任务失败。原进程与worker已确认退出，旧队列、进度、日志、数值门和状态保存在recovery_01/。原798条有效结果未重跑；仅补两条缺失记录，其中一条已有未完成启动，使用第二次且最后一次技术尝试。恢复仅使用当前空闲卡，未改冻结实验源码、条件或种子。\n'
    report.write_text(body)
    atomic(OUT/'recovery_complete.json',dict(passed=True,time=time.time(),retained_episodes=798,recovered_episodes=2,accounting_corrected=True))

def main():
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    assert not (ARCHIVE/'prepared.json').exists()
    original=json.loads((OUT/'process.json').read_text())
    for record in [original,*[json.loads(p.read_text()) for p in (OUT/'workers').glob('gpu*/process.json')]]:
        try:os.kill(record['pid'],0)
        except ProcessLookupError:continue
        raise RuntimeError(f'Old process still alive: {record["pid"]}')
    status=json.loads((OUT/'status.json').read_text());assert status['new_completed']==798
    for name in ('process.json','status.json','resource_scan.json','queue.json','progress.json','controller.log'):
        shutil.copyfile(OUT/name,ARCHIVE/name)
    (OUT/'workers').rename(ARCHIVE/'workers_before_interruption')
    release(0)
    available=controller.available
    free=available();target=next((g for g in (2,3,0) if g in free),None)
    assert target is not None,'No free GPU for bounded recovery'
    controller.available=lambda:[g for g in available() if g==target]
    atomic(ARCHIVE/'prepared.json',dict(time=time.time(),target_gpu=target,preserved_episodes=798,missing_episodes=2,reason='Previous controller and workers disappeared without recorded exit; exit cause unknown.'))
    original_finish=finish.main
    def recovered_finish():
        original_finish();correct_execution_accounting()
    finish.main=recovered_finish
    controller.main()

if __name__=='__main__':
    try:main()
    except Exception:
        atomic(ARCHIVE/'failed.json',dict(error=traceback.format_exc(),time=time.time()));raise
