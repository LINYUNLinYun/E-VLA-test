"""CPU-only contracts for immutable odd39 and explicit GPU0 continuation merging."""
import copy
import unittest
from . import d2_gpu0_continuation_analysis as a


def result_fixture(job,m):
    initial=dict(passed=True,instruction_equal=True,initial_vector_sha256_equal=True,image_changed_channels_total=0,
        fields={k:dict(passed=True,exact_equal=True,max_abs=0,changed_channels=0) for k in a.common.KEYS})
    return dict(status='complete',success=True,mode='v14_d2',manifest_hash=m['manifest_hash'],case_hash=job['case_hash'],case=job['case'],
        group=job['group'],source_index=job['source_index'],H=50,R=5,D=2,gpu_uuid=a.quality.ASSIGNED['0'],paired_results=job['paired_results'],
        initial_comparison=initial,native_info=dict(binding=dict(gpu=0,gpu_uuid=a.quality.ASSIGNED['0'])))


class ContinuationAnalysisContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=a.read(a.DEFAULT_RUN/'manifest.json');cls.parent=a.read(a.common.resolve(cls.m['parent_quality_manifest']['path']))
        cls.ordinals,cls.keys=a.validate_partition(cls.m,cls.parent)
        cls.frozen={a.key(row):dict(path=row['path'],sha256=row['sha256']) for row in cls.m['parent_gpu1_reused_complete']}
        cls.parent_complete={k:dict(path=r['path'],result_sha256=r['sha256'],success=True) for k,r in cls.frozen.items()}

    def test_exact98_39_58_partition(self):
        self.assertEqual(len(self.ordinals),195);self.assertEqual(len(self.keys),39);self.assertEqual(len(self.m['jobs']),58)
        self.assertEqual(sum(j['group']=='perturbed350' for j in self.m['jobs']),48)
        self.assertEqual(sum(j['group']=='original40' for j in self.m['jobs']),10)
        self.assertTrue(all(self.ordinals[a.key(j)]%2==1 for j in self.m['jobs']))

    def test_original_source0_odd_key_runs_on_gpu0_now(self):
        job=next(j for j in self.m['jobs'] if j['group']=='original40' and j['source_index']==0)
        self.assertEqual(self.ordinals[a.key(job)],175)
        result=result_fixture(job,self.m);a.validate_new_result(result,job,self.m)
        result['gpu_uuid']=a.quality.ASSIGNED['1']
        with self.assertRaises(ValueError):a.validate_new_result(result,job,self.m)

    def test_frozen39_sha_and_no_gpu1_expansion(self):
        merged,counts=a.merge_parent(self.parent_complete,self.frozen,self.ordinals)
        self.assertEqual(counts['frozen_parent_gpu1_odd'],39)
        changed=copy.deepcopy(self.parent_complete);next(iter(changed.values()))['result_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'SHA/path'):a.merge_parent(changed,self.frozen,self.ordinals)
        expanded=copy.deepcopy(self.parent_complete);expanded[a.key(self.m['jobs'][0])]=dict(path='extra',result_sha256='extra')
        with self.assertRaisesRegex(ValueError,'outside frozen39'):a.merge_parent(expanded,self.frozen,self.ordinals)

    def test_duplicate_valid_rejected_but_technical_retry_is_legal(self):
        retry=a.key(self.m['jobs'][0]);self.assertEqual(retry,('perturbed350',154))
        merged,_=a.merge_parent(self.parent_complete,self.frozen,self.ordinals)
        # Old case154 technical attempt never enters valid-complete map.
        result=a.merge_unique(merged,{retry:dict(success=False,namespace='continuation_gpu0_odd')})
        self.assertEqual(len(result),40);self.assertFalse(result[retry]['success'])
        with self.assertRaisesRegex(ValueError,'Duplicate valid complete'):a.merge_unique(result,{retry:dict(success=True)})

    def test_partial_and_final195_denominators(self):
        baseline={a.key(j):{mode:dict(success=True) for mode in a.common.MODES[:-1]} for j in self.parent['jobs']}
        partial={k:dict(success=True) for k in self.keys}
        _,table,_=a.common.tables(self.parent,baseline,partial)
        self.assertEqual(next(r['n'] for r in table if r['group']=='perturbed350' and r['dimension']=='overall'),39)
        full={a.key(j):dict(success=True) for j in self.parent['jobs']}
        _,table,_=a.common.tables(self.parent,baseline,full)
        for group,n in (('perturbed350',175),('original40',20)):
            rows=[r for r in table if r['group']==group and r['dimension']=='overall']
            self.assertEqual(len(rows),4);self.assertTrue(all(r['n']==n and r['pending_n']==0 for r in rows))

    def test_wrong_config_case_and_mutated_partition_rejected(self):
        job=self.m['jobs'][0];result=result_fixture(job,self.m);result['D']=10
        with self.assertRaises(ValueError):a.validate_new_result(result,job,self.m)
        result=result_fixture(job,self.m);result['case_hash']='wrong'
        with self.assertRaises(ValueError):a.validate_new_result(result,job,self.m)
        m=copy.deepcopy(self.m);m['jobs'][0]['case_hash']='wrong'
        with self.assertRaises(ValueError):a.validate_partition(m,self.parent)


if __name__=='__main__':unittest.main()
