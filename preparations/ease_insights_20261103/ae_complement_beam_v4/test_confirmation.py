"""Prospective checks for noise pseudoreplication, zero boundaries and complete-only inference."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .common import *
from . import analyze_confirmation as analysis
from . import statistics as stats
from .prepare_confirmation import ROOTS

AUDITOR=ROOT/'reports/ease_insights_20261103/ae_complement_beam_v4/confirmation_statistics_audit.py'
spec=importlib.util.spec_from_file_location('independent_cross_module_stats_test',AUDITOR)
independent=importlib.util.module_from_spec(spec);spec.loader.exec_module(independent)
DRIVER=ROOT/'reports/ease_insights_20261103/ae_complement_beam_v4/finish_confirmation.py'
spec=importlib.util.spec_from_file_location('cross_module_driver_test',DRIVER)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class ConfirmationInferenceContracts(unittest.TestCase):
    def compare(self,values):
        ids=[t for t in range(10) for _ in range(6)]
        result=stats.direction_inference(values,ids);other=independent.direction(values)
        self.assertEqual(result['primary_task_block']['task_difference_counts'],other['task_difference_counts'])
        self.assertEqual(result['primary_task_block']['p_one_sided'],other['p_one_sided'])
        for tag,ref in [('case','case_interval'),('task_cluster','task_interval')]:
            for bound in ('low','high'):
                self.assertEqual(result['confidence_intervals'][tag][bound],other[ref][bound])
        self.assertAlmostEqual(result['sensitivity_case_mcnemar_p'],other['sensitivity_case_p'])
        self.assertEqual(result['every_leave_one_task_out_positive'],other['loo_positive'])
        return result

    def test_all_ties_preserve_exact_zero_and1024_assignments(self):
        info=self.compare([0]*60)
        self.assertEqual(info['primary_task_block']['p_one_sided'],1.)
        self.assertEqual(info['primary_task_block']['assignments'],1024)
        self.assertEqual(info['confidence_intervals']['task_cluster']['low'],0.)
        self.assertFalse(info['task_cluster_lower_bound_positive'])

    def test_consistent_ten_task_effect(self):
        info=self.compare([1,0,0,0,0,0]*10)
        self.assertEqual(info['primary_task_block']['p_one_sided'],1/1024)
        self.assertGreater(info['confidence_intervals']['task_cluster']['low'],0.)
        self.assertTrue(info['every_leave_one_task_out_positive'])

    def test_twelve_positive_repeats_in_two_tasks_do_not_make_twelve_tasks(self):
        info=self.compare([1]*12+[0]*48)
        self.assertEqual(info['primary_task_block']['p_one_sided'],.25)
        self.assertLess(info['sensitivity_case_mcnemar_p'],.001)
        directions=[dict(inference=copy.deepcopy(info)) for _ in range(2)]
        self.assertEqual(stats.apply_verdict(directions,dict(vlm_c=True,ae14_d3=True)),'not_reproduced')

    def test_mixed_signed_task_sums_and_exact_zero_boundary(self):
        self.compare([1,1,0,0,-1,0,-1,-1,0,0,1,0]+[0]*48)
        info=self.compare([1,0,0,0,0,0]*3+[0]*42)
        self.assertEqual(info['confidence_intervals']['task_cluster']['low'],0.)

    def test_incorrect_cluster_design_rejected(self):
        for values,ids in [([0]*30,[t for t in range(10) for _ in range(3)]),([0]*60,list(range(60))),([2]*60,[t for t in range(10) for _ in range(6)])]:
            with self.assertRaises(ValueError):stats.direction_inference(values,ids)
        with self.assertRaises(ValueError):stats.task_signflip([7]+[0]*9)
        with self.assertRaises(ValueError):stats.holm([.01])

    def test_quality_or_cost_gate_cannot_be_overridden_by_significance(self):
        info=self.compare([1,0,0,0,0,0]*10)
        ds=lambda:[dict(inference=copy.deepcopy(info)) for _ in range(2)]
        self.assertEqual(stats.apply_verdict(ds(),dict(vlm_c=True,ae14_d3=True)),'complementarity_reproduced')
        self.assertEqual(stats.apply_verdict(ds(),dict(vlm_c=True,ae14_d3=False)),'not_reproduced')

    def test_no_partial_inference_before_complete_case_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory);(report/'confirmation_acceptance.json').write_text(json.dumps(dict(status='pass',new_cases=360,bundle_hash='b')))
            with patch.object(analysis,'REPORT',report),patch.object(analysis,'read_bundle',return_value=dict(bundle_hash='b')),patch.object(analysis,'read_manifest') as reader:
                with self.assertRaises(ValueError):analysis.load_complete()
                reader.assert_not_called()

    def test_streams_have_disjoint_output_roots_and360_cells_each(self):
        self.assertEqual(ROOTS,(FAMILY_ROOT/'confirmation',FAMILY_ROOT))
        base=json.loads((FAMILY_ROOT/'calibration/manifest.json').read_text())
        for repeat in (0,1):
            m=copy.deepcopy(base);m.update(phase='cross_module_confirm_v4',noise_repeat=repeat,confirmation_inits=[47,48,49],
                tasks=list(range(10)),case_table=[dict(suite=s,task=t) for s in SUITES for t in range(10)],
                cases_per_condition=120,total_confirmation_episodes=360,preregistered_hypotheses=copy.deepcopy(DIRECTIONS))
            validate_manifest(m)
            left,right=cases(m,'confirm',0,2),cases(m,'confirm',1,2)
            self.assertEqual((len(left),len(right)),(60,60))
            keys=lambda rows:{(s['suite'],s['task'],s['init']) for s in rows}
            self.assertFalse(keys(left)&keys(right));self.assertEqual(len(keys(left)|keys(right)),120)
            m['preregistered_hypotheses'].reverse()
            with self.assertRaises(ValueError):validate_manifest(m)

    def test_driver_stops_on_attention_without_analysis_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(driver,'REPORT',Path(directory)),patch.object(driver,'progress',return_value=dict(new_cases=0,streams=[dict(completed_shards=0),dict(completed_shards=0)],attention=True,all_complete=False)),patch.object(driver.subprocess,'run') as command:
                with self.assertRaises(RuntimeError):driver.run_locked(dict(bundle_hash='b'),'b','s',True)
                command.assert_not_called()

    def test_driver_pending_does_not_read_outcome_success_or_run_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(driver,'REPORT',Path(directory)),patch.object(driver,'progress',return_value=dict(new_cases=719,streams=[dict(completed_shards=6),dict(completed_shards=5)],attention=False,all_complete=False)),patch.object(driver.subprocess,'run') as command:
                self.assertEqual(driver.run_locked(dict(bundle_hash='b'),'b','s',False),dict(status='pending',new_cases=719))
                command.assert_not_called()


if __name__=='__main__':unittest.main()
