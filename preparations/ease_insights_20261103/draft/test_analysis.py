"""CPU regression: accounting must separate pure draft from verified acceptance."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from analyze import analyze
from common import atomic_json

class AccountingTest(unittest.TestCase):
    def test_paired_counts_and_technical_exclusion(self):
        with tempfile.TemporaryDirectory(prefix='ease-draft-analysis-') as tmp:
            root=Path(tmp)
            for mode,source in [('flash','libero_goal'),('flash','libero_spatial'),
                                ('draft','libero_goal'),('draft','libero_spatial'),('full','libero_goal')]:
                run=root/(mode+'_'+source)
                cases=[dict(suite='libero_goal',task_id=t,init_id=20) for t in range(3)]
                atomic_json(run/'manifest.json',dict(protocol={},mode=mode,source_suite=source))
                atomic_json(run/'cases.json',cases)
                requests=[]
                for task in range(3):
                    key='libero_goal_t%02d_i20'%task
                    result=run/'episodes'/key/'result.json'
                    if task==2:
                        atomic_json(result,dict(status='technical_failure')); continue
                    success=(mode=='full' or (task==0 and source=='libero_goal'))
                    atomic_json(result,dict(status='complete',success=success,wall_seconds=2.,mean_client_roundtrip_ms=5.,
                        episode=dict(infer_calls=2,executed_action_count=12)))
                    for step in range(2):
                        full=mode=='full' or (mode=='flash' and step==0)
                        draft=not full
                        verify=mode=='flash' and draft
                        requests.append(dict(suite='libero_goal',task_id=task,init_id=20,replan_id=step,
                            input_sha256='same-case-input',noise_sha256='same-case-noise',policy_wall_ms=5.,
                            route_calls=dict(full=int(full),draft=int(draft),verify=int(verify)),
                            timing=dict(total_ms=4.,accepted_prefix_len=6 if verify else 12,
                                ease_rejection_full=0,ease_periodic_full=0,ease_initial_full=int(full and step==0),
                                scheduled_full_fallback=0)))
                (run/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in requests))
            with contextlib.redirect_stdout(io.StringIO()):
                analyze([root],root/'analysis')
            summary=json.loads((root/'analysis/summary.json').read_text())
            for row in summary:
                self.assertEqual(row['n'],2)
                self.assertEqual(row['acceptance_rate'],.5 if row['mode']=='flash' else None)
                if row['mode']=='draft':
                    self.assertEqual(row['full_round_fraction'],0.)
            audit=json.loads((root/'analysis/audit.json').read_text())
            self.assertEqual(audit['complete_conditions'],10)
            self.assertEqual(len(audit['uncompleted']),5)
            self.assertEqual(audit['initial_pairing_mismatches'],[])
            pairs=json.loads((root/'analysis/paired_vs_matching_source.json').read_text())
            self.assertEqual(len(pairs),2)
            self.assertTrue(all(row['matched_only']==1 and row['foreign_only']==0 for row in pairs))


    def test_zero_action_rejection_cannot_fake_speedup(self):
        with tempfile.TemporaryDirectory(prefix='ease-zero-action-cost-') as tmp:
            root=Path(tmp)
            for mode,calls,policy_ms,rpc_ms in [('full',1,100.,110.),('flash',2,60.,65.)]:
                run=root/mode
                case=dict(suite='libero_goal',task_id=0,init_id=20)
                atomic_json(run/'manifest.json',dict(protocol={},mode=mode,source_suite='libero_goal'))
                atomic_json(run/'cases.json',[case])
                atomic_json(run/'episodes/libero_goal_t00_i20/result.json',dict(status='complete',success=True,
                    wall_seconds=.5 if mode=='full' else .7,mean_client_roundtrip_ms=rpc_ms,
                    client_roundtrip_sum_ms=rpc_ms*calls,episode=dict(infer_calls=calls,executed_action_count=12)))
                requests=[]
                for step in range(calls):
                    draft=mode=='flash' and step==1
                    requests.append(dict(suite='libero_goal',task_id=0,init_id=20,replan_id=step,
                        input_sha256='same-first-input',noise_sha256='same-first-noise',policy_wall_ms=policy_ms,
                        route_calls=dict(full=int(not draft),draft=int(draft),verify=int(draft)),
                        timing=dict(total_ms=.001,accepted_prefix_len=0 if draft else 12,
                                    scheduled_full_fallback=int(draft))))
                (run/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in requests))
            with contextlib.redirect_stdout(io.StringIO()):
                analyze([root],root/'analysis')
            rows={r['mode']:r for r in json.loads((root/'analysis/summary.json').read_text())}
            self.assertAlmostEqual(rows['full']['policy_ms_per_executed_action'],100/12)
            self.assertAlmostEqual(rows['flash']['policy_ms_per_executed_action'],120/12)
            self.assertEqual(rows['flash']['zero_action_draft_rounds'],1)
            speed=json.loads((root/'analysis/speed_vs_full.json').read_text())[0]
            self.assertAlmostEqual(speed['mean_policy_round_ratio_not_actual_speedup'],100/60)
            self.assertAlmostEqual(speed['policy_per_executed_action_speedup_vs_full'],100/120)
            self.assertAlmostEqual(speed['client_rpc_per_executed_action_speedup_vs_full'],110/130)
            self.assertLess(speed['policy_per_executed_action_speedup_vs_full'],1.)
            paired=json.loads((root/'analysis/paired_action_costs_vs_full.json').read_text())[0]
            self.assertAlmostEqual(paired['policy_per_executed_action_speedup_vs_full'],5/6)
            self.assertEqual((paired['full_success'],paired['condition_success']),(1,1))
            self.assertEqual((paired['full_wall_seconds'],paired['condition_wall_seconds']),(.5,.7))

if __name__=='__main__':
    unittest.main()
