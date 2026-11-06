"""Independent CPU contracts for fixed run2 provenance, shard identity and denominators."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import plus_d2_quality_v3_analysis as a


def result_fixture(job,m,ordinal):
    initial=dict(passed=True,instruction_equal=True,initial_vector_sha256_equal=True,image_changed_channels_total=0,
        fields={key:dict(passed=True,exact_equal=True,max_abs=0,changed_channels=0) for key in a.common.KEYS})
    card=ordinal%2
    return dict(status='complete',success=True,mode='v14_d2',manifest_hash=m['manifest_hash'],case_hash=job['case_hash'],
        case=job['case'],group=job['group'],source_index=job['source_index'],H=50,R=5,D=2,gpu_uuid=a.ASSIGNED[str(card)],
        paired_results=job['paired_results'],initial_comparison=initial,requests=1,
        native_info=dict(initial_vector_sha256='vector',instruction='prompt',binding=dict(gpu=card,gpu_uuid=a.ASSIGNED[str(card)])))


class QualityV3SummaryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=a.read(a.DEFAULT_RUN/'manifest.json')
        cls.order=a.read(a.resolve(cls.m['legacy_history_manifest']['path']))['legacy_mode_order']
        cls.baseline={(j['group'],j['source_index']):{mode:dict(success=True,native_info=dict(initial_vector_sha256='vector',instruction='prompt')) for mode in a.common.MODES[:-1]} for j in cls.m['jobs']}

    def test_targets_and_pending_rates(self):
        a.validate_structure(self.m)
        ref,paired,_=a.common.tables(self.m,self.baseline,{})
        counts={r['value']:r['n'] for r in ref if r['group']=='perturbed350' and r['dimension']=='mother_suite' and r['condition']=='full_d10'}
        self.assertEqual(counts,dict(zip(a.common.SUITES,(44,44,44,43))))
        self.assertTrue(all(r['n']==0 and r['success_rate'] is None and r['status']=='pending' for r in paired))
        self.assertEqual([r['n'] for r in ref if r['dimension']=='overall' and r['condition']=='full_d10'],[175,20])

    def test_shards_use_global_ordinal_not_source_index(self):
        ordinal=next(i for i,j in enumerate(self.m['jobs']) if j['group']=='original40' and j['source_index']==0)
        self.assertEqual(ordinal,175);job=self.m['jobs'][ordinal]
        result=result_fixture(job,self.m,ordinal);a.validate_result(result,job,self.m,ordinal)
        result['gpu_uuid']=a.ASSIGNED['0'];result['native_info']['binding']=dict(gpu=0,gpu_uuid=a.ASSIGNED['0'])
        with self.assertRaises(ValueError):a.validate_result(result,job,self.m,ordinal)

    def test_wrong_config_case_video_and_assignment_rejected(self):
        job=self.m['jobs'][0];good=result_fixture(job,self.m,0)
        for key,value in (('D',10),('case_hash','wrong'),('video',dict(path='video.mp4')),('R',1)):
            result=copy.deepcopy(good);result[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):a.validate_result(result,job,self.m,0)
        changed=copy.deepcopy(self.m);changed['assigned_gpus']['0']=a.ASSIGNED['1']
        with self.assertRaises(ValueError):a.validate_structure(changed)

    def test_runtime_gate_requires_all16_raw_and_decoded_bytewise(self):
        gate=a.read(a.DEFAULT_RUN/'runtime_gpu0/strict_gate/gate.json')
        a.validate_gate(gate,self.m,0,self.order)
        bad=copy.deepcopy(gate);bad['D10'][15]['repeats_vs_history'][1]['raw']['bitwise_equal']=False
        with self.assertRaisesRegex(ValueError,'not bytewise'):a.validate_gate(bad,self.m,0,self.order)
        bad=copy.deepcopy(gate);bad['D2'][1]['costs']['nodes']=[1.,0.]
        with self.assertRaisesRegex(ValueError,'nodes'):a.validate_gate(bad,self.m,0,self.order)

    def test_technical_excluded_and_duplicate_complete_rejected(self):
        job=self.m['jobs'][0];good=result_fixture(job,self.m,0)
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);mode=run/job['group']/('case%04d'%job['source_index'])/'v14_d2'
            directory=mode/'attempt000';directory.mkdir(parents=True)
            (directory/'result.json').write_text(json.dumps(dict(status='technical_failure',success=None,error='preserved')))
            complete,technical,_,_=a.collect(run,self.m,self.baseline,self.order)
            self.assertFalse(complete);self.assertEqual(len(technical),1)
            for index in (1,2):
                directory=mode/('attempt%03d'%index);directory.mkdir()
                (directory/'result.json').write_text(json.dumps(good));(directory/'initial_comparison.json').write_text(json.dumps(good['initial_comparison']))
            with patch.object(a,'validate_gate'),patch.object(a.common,'bound'),patch.object(a.common,'validate_stream'):
                gatepath=run/'runtime_gpu0/strict_gate/gate.json';gatepath.parent.mkdir(parents=True);gatepath.write_text(json.dumps(dict(arrays=[])))
                with self.assertRaisesRegex(ValueError,'Duplicate valid complete'):a.collect(run,self.m,self.baseline,self.order)

    def test_partial_uses_same_intersection_and_change_counts(self):
        job=self.m['jobs'][0];key=(job['group'],job['source_index'])
        ref,paired,changes=a.common.tables(self.m,self.baseline,{key:dict(success=False)})
        group=[r for r in paired if r['group']=='perturbed350' and r['dimension']=='overall']
        self.assertEqual(len(group),4);self.assertTrue(all(r['n']==1 and r['target_n']==175 for r in group))
        change=next(r for r in changes if r['group']=='perturbed350' and r['dimension']=='overall')
        self.assertEqual(change['baseline_only'],1);self.assertEqual(change['D2_only'],0)
        self.assertEqual(next(r['n'] for r in ref if r['group']=='perturbed350' and r['dimension']=='overall'),175)


if __name__=='__main__':unittest.main()
