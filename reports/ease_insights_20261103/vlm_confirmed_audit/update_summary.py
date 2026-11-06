"""Update only the authorized VLM summary sections after the original 600 audit."""
import hashlib
import json
from pathlib import Path
import time
from preparations.ease_insights_20261103.vlm.common import atomic_json, sha

WORKSPACE=Path('/root/nfs/embodied/benchmark')
RESULTS=WORKSPACE/'results/ease_insights_20261103'
AUDIT=Path(__file__).resolve().parent
SUMMARY=AUDIT.parent/'README_zh.md'


def main():
    audit=json.loads((AUDIT/'audit.json').read_text())
    assert audit['status']=='complete' and audit['valid_episodes']==600
    old=SUMMARY.read_text()
    draft_start='## 固定 draft：'
    fast_start='## Fast path：'
    draft_before=old.split(draft_start,1)[1].split(fast_start,1)[0]
    prefix=old.split(fast_start,1)[0]
    prefix=prefix.replace('Full 对照正在补齐，工作尚未结束。','本轮 Full 对照及600例正式审阅已完成；新的局部搜索仍在推进，整体工作尚未结束。')
    summaries=audit['condition_summaries']
    full=summaries['full_d10']['successes'];jac=summaries['fixed_v12_jacobian']['successes'];mse=summaries['fixed_v12_action_mse']['successes']
    local_paths=[RESULTS/'local_supervisor/status.json',RESULTS/'vlm_local_search_v1/supervisor/status.json',
        RESULTS/'vlm_local_search_v1/local_supervisor/status.json']
    available=[p for p in local_paths if p.exists()]
    path=max(available,key=lambda p:p.stat().st_mtime) if available else None
    local=json.loads(path.read_text()) if path else None
    phase=None if local is None else local.get('phase')
    phases={'waiting_discovery':'等待局部发现阶段完成','freeze_selection':'正在冻结新候选',
        'prepare_confirmation':'正在准备新一轮独立确认','enqueue_confirmation':'正在将独立确认入队',
        'waiting_confirmation':'等待新一轮独立确认完成','analyze_confirmation':'正在分析新一轮独立确认',
        'report_ready':'新局部搜索报告已生成，仍需结合其结果审阅','no_candidate':'局部搜索未选出合格候选',
        'needs_attention':'局部搜索需要处理技术问题'}
    local_text='有限的同预算层组合局部搜索仍在准备；尚未发现local supervisor状态文件。' if path is None else (
        '有限的同预算层组合局部搜索：'+phases.get(phase,'当前状态为 '+str(phase))+'。'+
        '读取状态为 `'+str(phase)+'`，时间戳 '+str(local.get('time'))+'；详见 `'+str(path.relative_to(WORKSPACE))+'`。')
    lines=['## Fast path：可用路径存在，当前互补假设未复现','',
        f'VLM发现160例和独立确认600例均完整。Full成功{full}/200，固定选出的Jac/MSE路径分别成功{jac}/200与{mse}/200。两条路径保持相同的12/18保留层预算、Full-D10→Fast-D2→Fast-D2周期、H50/R5与残差补偿。它们的固定输入策略成本优势不等于闭环加速或质量非劣。','',
        '| 预定方向 | 独立成功数 | 差值 | Holm3 |','| --- | --- | ---: | ---: |']
    names={'primary_spatial':'Spatial：Jac > MSE','primary_long':'Long：MSE > Jac','supplementary_goal':'事先登记的补充 Goal：Jac > MSE'}
    for row in audit['three_fixed_contrasts']:
        lines.append(f"| {names[row['contrast_id']]} | {row['a_success']}/50 对{row['b_success']}/50 | {100*row['success_difference']:+.0f}pp | {row['inference']['holm_adjusted_p']:.6g} |")
    lines += ['',
        '三方向的case与task聚类区间均跨0，均未满足预先固定的稳定规则。原Spatial/Long主方向及预先登记的Goal/Long补充方向均未复现；不显著也不证明路径等价。[600例正式主/补审阅](vlm_confirmed_audit/README_zh.md)已完成，补充报告中的原主分析与[正式主报告](vlm_confirmation/analysis.md)逐项核对一致；原Holm2保留，三方向另给Holm3。此前vision pilot同样未复现预定互补方向，结果保留在[vision审阅](vision_pilot/scientific_verdict.md)。',
        '',
        'Full、Jac与MSE的质量结果全部保留原始成功/失败，没有因资源争用改写。policy/RPC/episode墙钟及固定输入探针成本都可能受外部负载影响；episode墙钟之和也不是并行makespan，因此不宣称独占GPU加速。完整成本、争用记录和逐task/init稳定性见正式报告。',
        '',
        '逐40任务及预定义语义类别的[后续探索](vlm_pair_interim/task_semantic_exploration.md)也未找到稳健的反向候选。新候选如使用这批数据筛选，必须标为新发现，并用新的初态独立验证；不得改写本轮预定方向。',
        '', '## 正在继续','',
        'Full确认此前曾在模型初始化阶段因同卡外部占用OOM，未产生有效episode。原失败记录保留；相同科学命令用新队列ID恢复，在GPU3通过连续60秒自然空闲检查后完成Full200/200。已完成的科学结果没有重跑。',
        '',
        '本轮600条已完成正式主分析、事先登记的补充分析和资源争用审阅；这只表示原确认矩阵完成，不表示整个研究目标完成。'+local_text,
        '',
        '局部搜索的发现、选择和后续确认属于另一个阶段，不能改写本轮阴性结论，也不能把排队或准备写成实验结果。原确认状态见 `results/ease_insights_20261103/pipeline/status.json`；局部新阶段以其supervisor状态及真实进程为准。无空闲卡时保持等待，不因资源不足宣告实验结束。','']
    updated=prefix+'\n'.join(lines)
    assert updated.split(draft_start,1)[1].split(fast_start,1)[0]==draft_before
    before_sha=sha(SUMMARY)
    SUMMARY.write_text(updated)
    atomic_json(AUDIT/'summary_update_receipt.json',dict(status='complete',updated_unix=time.time(),path=str(SUMMARY),
        before_sha256=before_sha,after_sha256=sha(SUMMARY),draft_section_unchanged=True,
        draft_section_sha256=hashlib.sha256(draft_before.encode()).hexdigest(),
        full_successes=full,local_supervisor_status_path=None if path is None else str(path),
        local_supervisor_status=local,whole_goal_complete_claim=False))
    print(json.dumps(dict(status='summary_updated',full_successes=full,local_phase=phase,whole_goal_complete_claim=False)),flush=True)


if __name__=='__main__':main()
