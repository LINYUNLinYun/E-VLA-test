"""Audit actual Full counts, independent mask scores, verifier and histories."""
import json
import time
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,PLAN,SUITES,VARIANTS,directory,ck,verify_frozen,write_csv
from .scoring import score_and_choose,refresh_due
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_3.rules import acceptance
from preparations.idea1_v3_4_E2_adjacency_200.common import legal


def audit_episode(cfg,output,result):
    output=Path(output)
    records=[json.loads(line) for line in (output/'infer.jsonl').read_text().splitlines()]
    counts=dict(full=0,draft=0,verify=0,selector=0,vjp=0,kalman=0)
    step=0;full_round=full_step=mask_step=None;history=[];mask=None
    draft_rounds=0;pending_gripper=False;last_gripper=None
    with np.load(output/'arrays.npz') as arrays:
        for i,record in enumerate(records):
            route=record['route'];timing=record['timing']
            assert record['round']==record['replan']==i and record['step']==step
            assert not route['cuda_graph'] and not route['kalman']
            assert route['period']==cfg['period'] and route['selection_stride']==cfg['selection_stride']
            assert route['aggregation']==cfg['aggregation'] and route['vjp_batch_size']==1
            assert route['draft_rounds_at_entry']==draft_rounds
            expected_age=None if full_round is None else dict(rounds=i-full_round,env_steps=step-full_step)
            assert route['full_age_at_entry']==expected_age
            reason='startup' if full_round is None else 'pending_gripper' if pending_gripper else 'periodic' if draft_rounds>=cfg['period']-1 else None
            attempt=reason is None
            assert route['draft_attempted']==route['verify_performed']==attempt
            proposal=5
            if attempt:
                assert route['draft_skip']==mask and route['verify_full_round']==full_round and route['verify_full_step']==full_step
                assert len(timing['draft_nodes'])==3
                counts['draft']+=1;counts['verify']+=1;draft_rounds+=1
                gap=None if len(history)<2 else history[-1]-history[-2]
                age=step-history[-1];ratio=0. if gap is None else age/gap
                assert route['draft_history_full_steps']==history and route['draft_residual_age']==age
                assert route['draft_full_gap']==gap and np.isclose(route['draft_extrapolation_ratio'],ratio)
                assert route['captured_layers']==list(range(1,18))
                length,detail=acceptance(arrays[f'draft_future_{i}'],arrays[f'reconstructed_future_{i}'],
                    cfg['tau'],5,last_gripper=last_gripper,protect=True)
                assert length==route['accepted_length']
                assert all(route[key]==detail[key] for key in ('spatial_length','verify_gripper','post_verify_gripper'))
                switch=detail['verify_gripper'] or detail['post_verify_gripper']
                if length==0:reason='reject_gripper' if switch else 'reject_spatial'
                else:proposal=length;pending_gripper=switch
            else:assert route['accepted_length'] is None
            full=reason is not None
            assert (route['path']=='full')==full and route['fallback']==(attempt and full)
            refresh=full and refresh_due(counts['full'],cfg['selection_stride'])
            if full:
                assert route['actual_full_index']==counts['full'] and route['full_reason']==reason
                assert len(timing['full_nodes'])==11
                if refresh:
                    gradient=arrays[f'gate_gradient_{i}'];score=arrays[f'layer_score_{i}']
                    mask,expected_score,combination_scores=score_and_choose(gradient,18-cfg['budget'],cfg['aggregation'])
                    np.testing.assert_array_equal(score,expected_score)
                    np.testing.assert_array_equal(arrays[f'combination_score_{i}'],combination_scores)
                    mask_step=step;counts['selector']+=1;counts['vjp']+=4
                else:
                    assert f'gate_gradient_{i}' not in arrays.files
                    assert timing['selector_total_ms']==0.
                counts['full']+=1;full_round=i;full_step=step;draft_rounds=0;pending_gripper=False
                # Every actual Full refreshes residuals, including non-selector Full.
                history=(history+[step])[-2:]
            assert legal(mask,18-cfg['budget'],'spaced')
            assert route['mask_refresh']==refresh
            assert route['steady_vjp_count']==(4 if refresh else 0)
            assert route['selector_forward_count']==int(refresh)
            assert route['actual_fulls_after']==counts['full'] and route['selected_skip']==mask
            assert route['mask_age_env_steps']==step-mask_step
            assert route['history_full_steps_after']==history and route['full_round_after']==full_round and route['full_step_after']==full_step
            assert route['draft_rounds_after']==draft_rounds and route['pending_gripper_after']==pending_gripper
            assert record['counts']==counts and timing['selector_capture_ms']==0.
            category='full_'+reason if full else 'fast_accept_5' if proposal==5 else 'fast_accept_partial'
            assert route['request_category']==category
            executed=record['executed_length'];assert 0<executed<=proposal<=5
            assert np.isfinite(arrays[f'raw_{i}']).all() and np.isfinite(arrays[f'actions_{i}']).all()
            last_gripper=float(arrays[f'raw_{i}'][executed-1,6]);step+=executed
    assert result['valid'] and result['counts']==counts and result['actions']==step and result['predictions']==len(records)
    expected_selectors=(counts['full']-1)//cfg['selection_stride']+1
    assert counts['selector']==expected_selectors and counts['vjp']==4*expected_selectors
    return dict(passed=True,requests=len(records),counts=counts,actual_actions=step,
        full_history_refreshed_each_full=True,selection_stride=cfg['selection_stride'])


def prefix(base,candidate):
    left=[json.loads(line) for line in (base/'infer.jsonl').read_text().splitlines()]
    right=[json.loads(line) for line in (candidate/'infer.jsonl').read_text().splitlines()]
    checks=dict(first_input=left[0]['input_hash']==right[0]['input_hash'],
        first_fast_input=len(left)<2 or len(right)<2 or left[1]['input_hash']==right[1]['input_hash'])
    with np.load(base/'arrays.npz') as a,np.load(candidate/'arrays.npz') as b:
        for key in ('raw_0','actions_0','gate_gradient_0'):
            tolerance=(1e-4,1e-3) if key=='gate_gradient_0' else (1e-5,1e-5)
            checks[key]=bool(np.allclose(a[key],b[key],atol=tolerance[0],rtol=tolerance[1]))
    return dict(passed=all(checks.values()),checks=checks)


def main():
    manifest=json.loads((OUT/'manifest.json').read_text());verify_frozen(manifest)
    summaries=[];requests=[];episodes=[];pair_rows=[];prefix_rows=[]
    for cfg in manifest['conditions']:
        condition_requests=[];condition_episodes=[]
        for case in manifest['cases']:
            output=directory(cfg['id'],case)
            result=json.loads((output/'result.json').read_text())
            assert (output/'complete.json').exists() and result['config']==cfg and result['manifest_hash']==manifest['manifest_hash']
            audit=audit_episode(cfg,output,result)
            atomic(OUT/'episode_audits'/cfg['id']/f'{ck(case)}.json',audit)
            condition_episodes.append(result)
            episodes.append(dict(condition=cfg['id'],variant=cfg['variant'],**case,success=result['success'],
                actions=result['actions'],requests=result['predictions'],policy_wall_s=result['policy_wall_s'],wall_s=result['wall_s']))
            for record in [json.loads(line) for line in (output/'infer.jsonl').read_text().splitlines()]:
                timing=record['timing'];route=record['route']
                condition_requests.append(dict(condition=cfg['id'],variant=cfg['variant'],**case,
                    replan=record['replan'],executed=record['executed_length'],full=route['path']=='full',
                    refresh=route['mask_refresh'],fallback=route['fallback'],category=route['request_category'],
                    policy_ms=timing['policy_wall_ms'],selector_ms=timing['selector_total_ms'],
                    scoring_ms=timing.get('selector_scoring_ms',0.),setup_ms=timing.get('selector_setup_ms',0.),
                    full_path_ms=timing.get('full_path_wall_ms',0.),rpc_ms=record['rpc_ms'],
                    aggregation=cfg['aggregation'],selection_stride=cfg['selection_stride']))
        requests+=condition_requests
        hot=[r for r in condition_requests if r['replan']>0]
        triggers=[r for r in hot if r['refresh']]
        plain=[r for r in hot if r['full'] and not r['refresh']]
        row=dict(condition=cfg['id'],base_id=cfg['base_id'],variant=cfg['variant'],budget=cfg['budget'],period=cfg['period'],
            n=len(condition_episodes),success=sum(int(r['success']) for r in condition_episodes),
            **{s:sum(int(r['success']) for r in condition_episodes if r['case']['suite']==s) for s in SUITES},
            requests=len(condition_requests),fulls=sum(r['full'] for r in condition_requests),
            selectors=sum(r['refresh'] for r in condition_requests),
            policy_ms_per_request=float(np.mean([r['policy_ms'] for r in condition_requests])),
            hot_policy_ms_per_request=float(np.mean([r['policy_ms'] for r in hot])),
            selector_ms_per_request=float(np.mean([r['selector_ms'] for r in hot])),
            policy_ms_per_action=sum(r['policy_ms'] for r in condition_requests)/sum(r['executed'] for r in condition_requests),
            trigger_full_mean_ms=float(np.mean([r['full_path_ms'] for r in triggers])) if triggers else None,
            trigger_request_mean_ms=float(np.mean([r['policy_ms'] for r in triggers])) if triggers else None,
            trigger_request_p95_ms=float(np.percentile([r['policy_ms'] for r in triggers],95)) if triggers else None,
            nonrefresh_full_mean_ms=float(np.mean([r['full_path_ms'] for r in plain])) if plain else None,
            scoring_mean_ms=float(np.mean([r['scoring_ms'] for r in triggers])) if triggers else None,
            total_policy_s=sum(r['policy_ms'] for r in condition_requests)/1000,
            episode_wall_s=sum(r['wall_s'] for r in condition_episodes),
            selector_setup_s=sum(r['setup_ms'] for r in condition_requests)/1000)
        summaries.append(row)
    for base_cfg in [c for c in manifest['conditions'] if c['variant']=='baseline']:
        for variant in ('stride2','stride3','combination'):
            cfg=next(c for c in manifest['conditions'] if c['base_id']==base_cfg['base_id'] and c['variant']==variant)
            values=[]
            for case in manifest['cases']:
                a=directory(base_cfg['id'],case);b=directory(cfg['id'],case)
                check=prefix(a,b);prefix_rows.append(dict(base=base_cfg['id'],candidate=cfg['id'],case=case,**check))
                ra=json.loads((a/'result.json').read_text());rb=json.loads((b/'result.json').read_text())
                values.append(dict(passed=check['passed'],baseline=ra['success'],candidate=rb['success']))
            for scope,selected in [('all8',values),('strict',[r for r in values if r['passed']])]:
                pair_rows.append(dict(base=base_cfg['id'],candidate=cfg['id'],scope=scope,n=len(selected),
                    baseline_success=sum(r['baseline'] for r in selected),candidate_success=sum(r['candidate'] for r in selected),
                    recovered=sum(not r['baseline'] and r['candidate'] for r in selected),
                    new_failures=sum(r['baseline'] and not r['candidate'] for r in selected)))
    write_csv(OUT/'summary.csv',summaries);write_csv(OUT/'requests.csv',requests)
    write_csv(OUT/'episodes.csv',episodes);write_csv(OUT/'paired.csv',pair_rows)
    atomic(OUT/'prefix_checks.json',dict(checks=prefix_rows,anomalies=sum(not r['passed'] for r in prefix_rows)))
    execution=json.loads((OUT/'execution.json').read_text());wall=execution['evaluation_finish']-execution['evaluation_start']
    throughput=dict(episodes=len(episodes),evaluation_wall_s=wall,episodes_per_hour=len(episodes)*3600/wall,
        actions_per_second=sum(r['actions'] for r in episodes)/wall,requests_per_second=len(requests)/wall,
        gpu=execution['gpu'],gpu_uuid=execution['gpu_uuid'],single_gpu=True)
    atomic(OUT/'throughput.json',throughput)
    benchmark=json.loads((OUT/'benchmark/complete.json').read_text())
    lines=['# 单卡选层开销：三项独立对照','',
        '原π0、四组投影seed340610、间隔跳层、V12/V14、周期3/5、Full-D10/Fast-D2、H50/R5、tau=.10及夹爪保护；无Kalman。',
        '所有CUDA Graph构建均由运行期守卫禁止。首次Full反推，stride2/3按实际Full索引0,2,4…/0,3,6…刷新；每次Full都更新外推残差。组合评分仍用串行四VJP、每Full刷新。',
        '', '4 suites × task0/2 × init19，每条件8例，16条件共128例。本轮是有界探索小批，不能据此确认总体成功率变化。', '',
        '|预算/周期|方法|成功|热态策略ms/请求|选层均摊ms/请求|触发选层Full ms|触发请求P95 ms|评分ms|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"|V{r['budget']}/P{r['period']}|{r['variant']}|{r['success']}/8|{r['hot_policy_ms_per_request']:.2f}|{r['selector_ms_per_request']:.2f}|{r['trigger_full_mean_ms']:.2f}|{r['trigger_request_p95_ms']:.2f}|{r['scoring_mean_ms']:.4f}|")
    lines += ['', '热态闭环描述统计去除每回合首请求；所有请求、冷准备和完整回合墙钟另保留于CSV。触发请求包括可能的被拒Fast及验证开销，因此与Full子路径不同。自由轨迹输入及Full组成不同，不能把均值差当作同输入因果加速。', '',
        f"同输入批量反传数值核验：{benchmark['equivalence']}。详见benchmark/checks.json、timings.csv、preparation.csv、memory.csv及gradients.npz；旧455ms含局部图，不作为本轮无图基线。", '',
        f"闭环墙钟{wall/60:.2f}分钟，吞吐{throughput['episodes_per_hour']:.1f}回合/小时；单张GPU{execution['gpu']}（{execution['gpu_uuid']}）。模型加载、独立微基准和选层首次准备分别记录。", '',
        '组合评分对每方向先累加所选层带符号梯度，再对四方向取RMS并遍历全部合法组合。它仍是局部一阶近似，未解决非线性交互，也不减少反传次数。', '',
        '[冻结方案](idea1_v3_4_selection_cost_plan.md)', '']
    (ROOT/'docs/experiments/idea1/idea1_v3_4_selection_cost_report.md').write_text('\n'.join(lines))
    atomic(OUT/'audit.json',dict(passed=True,episodes=128,graph_capture_count=0,source_hashes_checked=True,
        full_refresh_and_verifier_audited=True,prefix_anomalies=sum(not r['passed'] for r in prefix_rows)))
    atomic(OUT/'complete.json',dict(passed=True,episodes=128,time=time.time()))
    atomic(OUT/'status.json',dict(status='complete',completed=128,target=128,time=time.time()))


if __name__=='__main__':main()
