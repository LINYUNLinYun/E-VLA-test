"""CPU contracts for strict D2 identities and paired, nonzero-only denominators."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import plus_d2_analysis as a


def result_fixture(job,m):
    initial=dict(passed=True,instruction_equal=True,initial_vector_sha256_equal=True,image_changed_channels_total=0,
        fields={key:dict(passed=True,exact_equal=True,max_abs=0,changed_channels=0) for key in a.KEYS})
    return dict(status='complete',success=True,mode='v14_d2',manifest_hash=m['manifest_hash'],case_hash=job['case_hash'],
        case=job['case'],group=job['group'],source_index=job['source_index'],H=50,R=5,D=2,gpu_uuid=m['required_gpu_uuid'],
        paired_results=job['paired_results'],initial_comparison=initial,requests=1,
        native_info=dict(initial_vector_sha256='vector',instruction='prompt'))


class D2SummaryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=a.read(a.DEFAULT_RUN/'manifest.json')
        cls.baseline={(j['group'],j['source_index']):{mode:dict(success=index%2==0,native_info=dict(initial_vector_sha256='vector',instruction='prompt'))
            for mode in a.MODES[:-1]} for index,j in enumerate(cls.manifest['jobs'])}

    def test_exact_target_and_suite_denominators(self):
        jobs=a.validate_structure(self.manifest);self.assertEqual(len(jobs),195)
        reference,paired,_=a.tables(self.manifest,self.baseline,{})
        denominator={r['value']:r['n'] for r in reference if r['group']=='perturbed350' and r['dimension']=='mother_suite' and r['condition']=='full_d10'}
        self.assertEqual(denominator,dict(zip(a.SUITES,(44,44,44,43))))
        for group,n in (('perturbed350',175),('original40',20)):
            self.assertEqual(next(r['n'] for r in reference if r['group']==group and r['dimension']=='overall'),n)
        self.assertTrue(all(r['n']==0 and r['success_rate'] is None and r['status']=='pending' for r in paired))

    def test_partial_intersection_is_common_to_all_four_conditions(self):
        job=self.manifest['jobs'][0];key=(job['group'],job['source_index'])
        reference,paired,changes=a.tables(self.manifest,self.baseline,{key:dict(success=False)})
        cells=[r for r in paired if r['group']=='perturbed350' and r['dimension']=='overall']
        self.assertEqual(len(cells),4);self.assertEqual({r['n'] for r in cells},{1})
        self.assertEqual({r['target_n'] for r in cells},{175})
        change=next(r for r in changes if r['group']=='perturbed350' and r['dimension']=='overall' and r['baseline']=='full_d10')
        self.assertEqual(change['baseline_only'],1);self.assertEqual(change['both_fail'],0)
        self.assertEqual(next(r['n'] for r in reference if r['group']=='perturbed350' and r['dimension']=='overall'),175)

    def test_complete_intersection_retains175_and20_denominators(self):
        complete={(j['group'],j['source_index']):dict(success=True) for j in self.manifest['jobs']}
        _,paired,changes=a.tables(self.manifest,self.baseline,complete)
        for group,n in (('perturbed350',175),('original40',20)):
            rows=[r for r in paired if r['group']==group and r['dimension']=='overall']
            self.assertEqual(len(rows),4)
            self.assertTrue(all(r['n']==n and r['pending_n']==0 and r['status']=='complete' for r in rows))
            for row in (r for r in changes if r['group']==group and r['dimension']=='overall'):
                self.assertEqual(sum(row[k] for k in ('both_success','baseline_only','D2_only','both_fail')),n)

    def test_wrong_case_configuration_and_initial_gate_are_rejected(self):
        job=self.manifest['jobs'][0];good=result_fixture(job,self.manifest)
        a.validate_result(good,job,self.manifest)
        for key,bad in (('D',10),('H',25),('R',1),('mode','v14_d10'),('case_hash','wrong'),('gpu_uuid','other')):
            result=copy.deepcopy(good);result[key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):a.validate_result(result,job,self.manifest)
        result=copy.deepcopy(good);result['initial_comparison']['fields']['robot0_eef_pos']['exact_equal']=False
        with self.assertRaises(ValueError):a.validate_result(result,job,self.manifest)
        m=copy.deepcopy(self.manifest);m['nodes']=[1.,0.]
        with self.assertRaises(ValueError):a.validate_structure(m)

    def test_technical_and_running_do_not_enter_denominator(self):
        job=self.manifest['jobs'][0]
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            for index,status in enumerate(('technical_failure','running')):
                directory=run/job['group']/('case%04d'%job['source_index'])/'v14_d2'/('attempt%03d'%index)
                directory.mkdir(parents=True);(directory/'result.json').write_text(json.dumps(dict(status=status,success=None)))
            complete,technical,other=a.collect(run,self.manifest,self.baseline)
        self.assertFalse(complete);self.assertEqual(len(technical),1);self.assertEqual(len(other),1)
        self.assertTrue(all(r['success_rate'] is None for r in a.tables(self.manifest,self.baseline,complete)[1]))

    def test_duplicate_complete_is_rejected_without_selecting_winner(self):
        job=self.manifest['jobs'][0];result=result_fixture(job,self.manifest)
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            for index in range(2):
                directory=run/job['group']/('case%04d'%job['source_index'])/'v14_d2'/('attempt%03d'%index)
                directory.mkdir(parents=True);(directory/'result.json').write_text(json.dumps(result))
                (directory/'initial_comparison.json').write_text(json.dumps(result['initial_comparison']))
            with patch.object(a,'bound'),patch.object(a,'validate_stream'):
                with self.assertRaisesRegex(ValueError,'Duplicate valid complete'):a.collect(run,self.manifest,self.baseline)

    def test_actual_d2_nodes_and_route_checked(self):
        response=dict(mode='v14_d2',manifest_hash=self.manifest['manifest_hash'],execute_length=5,
            timing=dict(actual_NFE=2,nodes=[1.,.5,0.]),observed_calls=dict(vision_encoder_calls=1,vlm_prefix_calls=1,
                ae_layer_calls=36,learned_head_calls=0,verify_calls=0,constructor_full_record_calls=0,ae_euler_updates=2))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'infer.jsonl';path.write_text(json.dumps(dict(response=response))+'\n')
            self.assertEqual(a.validate_stream(path,self.manifest,dict(requests=1)),1)
            response['timing']['nodes']=[1.,0.];path.write_text(json.dumps(dict(response=response))+'\n')
            with self.assertRaisesRegex(ValueError,'nodes/NFE'):a.validate_stream(path,self.manifest,dict(requests=1))


if __name__=='__main__':unittest.main()
