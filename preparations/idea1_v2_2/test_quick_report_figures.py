import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from common import SUITES,atomic_json,sha
import quick_report_figures as report


def fixture():
    conditions=[]; records={}; success=[]; latency=[]
    for mask in ('full','skip'):
        for steps in (10,2):
            name=f'{mask}_D{steps}'
            condition=dict(id=name,model='db-cogact',mask=mask,schedule=dict(kind='native',num_steps=steps,m=None))
            conditions.append(condition)
            for suite in SUITES:
                for task in range(10):
                    for init in (6,7,8):
                        key=f'{name}_{suite}_{task}_{init}'; passed=task%2==0
                        records[key]=dict(key=key,condition=name,model='db-cogact',suite=suite,task=task,init=init,
                            status='valid',success=passed,result=dict(success=passed,furthest_phase='success_only' if passed else 'grasp'))
            for suite in (*SUITES,'overall'):
                n=120 if suite=='overall' else 30
                success.append(dict(condition=name,suite=suite,successes=n//2,denominator=n,success_rate=.5))
            for component in report.COMPONENTS:
                latency.append(dict(condition=name,component=component,denominator=640,mean_ms=steps+1))
    scope=dict(main_inits=[6,7,8],main_keys=list(records),extra_keys=['extra'],valid_records=records)
    scope['valid_records']['extra']=dict(key='extra',success=False,result=dict(success=False,furthest_phase='extra_only'))
    return dict(conditions=conditions),scope,success,latency


class ReportingTests(unittest.TestCase):
    def test_exact_cohort_mapping_and_failure_excludes_success_extras(self):
        manifest,scope,success,latency=fixture(); data=report.build_data(manifest,scope,success,latency)
        self.assertEqual(data['main_episodes'],480); self.assertEqual(data['excluded_extras'],1)
        self.assertEqual(data['success_heatmaps'][0]['cells'][0][-1],dict(successes=60,denominator=120,percent=50))
        self.assertEqual(len(data['latency_comparisons']),4)
        self.assertEqual(sum(row['valid_failures'] for row in data['failure_furthest']),240)
        self.assertEqual({row['condition'] for row in data['failure_furthest']},{c['id'] for c in manifest['conditions']})
        self.assertTrue(all(row['valid_episodes']==120 for row in data['failure_furthest']))
        self.assertEqual({phase for row in data['failure_furthest'] for phase in row['furthest_counts']},{'grasp'})
        self.assertEqual({row['timing']['server_total_ms']['denominator'] for row in data['latency_comparisons']},{640})
        # A perfect u50/D2 condition must not inherit another schedule's failures.
        target=manifest['conditions'][0]['id']
        for row in scope['valid_records'].values():
            if row.get('condition')==target: row['success']=True; row['result']['success']=True
        for row in success:
            if row['condition']==target: row['successes']=row['denominator']; row['success_rate']=1.
        data=report.build_data(manifest,scope,success,latency)
        self.assertEqual(next(row['valid_failures'] for row in data['failure_furthest'] if row['condition']==target),0)
        self.assertEqual(next(row['valid_failures'] for row in data['failure_furthest'] if row['condition']!=target),60)

    def test_wrong_denominator_duplicate_or_wrong_rate_rejected(self):
        manifest,scope,success,latency=fixture()
        for change in ({'denominator':50},{'success_rate':.7}):
            broken=copy.deepcopy(success); broken[0].update(change)
            with self.assertRaises(ValueError): report.build_data(manifest,scope,broken,latency)
        with self.assertRaises(ValueError): report.build_data(manifest,scope,success+[success[0]],latency)
        broken=copy.deepcopy(latency); broken[0]['denominator']=160
        with self.assertRaises(ValueError): report.build_data(manifest,scope,success,broken)
        broken=copy.deepcopy(scope); broken['main_keys'][1]=broken['main_keys'][0]
        with self.assertRaises(ValueError): report.build_data(manifest,broken,success,latency)

    def test_render_three_new_figures_and_preserve_inputs(self):
        manifest,scope,success,latency=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); atomic_json(root/'condition_manifest.json',manifest); atomic_json(root/'quick_stage_scope.json',scope)
            for name,rows in [('success_by_condition_suite.csv',success),('latency_breakdown.csv',latency)]:
                with (root/name).open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            original={p.name:sha(p) for p in root.iterdir()}
            output=report.render(root)
            self.assertEqual(len(list(output.glob('*.png'))),3)
            receipt=json.loads((output/'reporting_receipt.json').read_text())
            self.assertFalse(receipt['independent_C01_C08_all_passed'])
            self.assertEqual(receipt['outputs']['plotted_data.json'],sha(output/'plotted_data.json'))
            self.assertTrue(all(sha(root/name)==value for name,value in original.items()))
            with self.assertRaises(FileExistsError): report.render(root)


if __name__=='__main__': unittest.main()
