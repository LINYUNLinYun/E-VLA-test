import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from common import SUITES, planned, sha, atomic_json
import quick_stage_scope as scope
import runtime_analysis_quick as analysis


class QuickTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=dict(conditions=[dict(id=f'c{i}',model='m',extension=False,schedule={},mask='full') for i in range(54)],
                          protocol_hashes={'m':'protocol'},manifest_hash='canonical')
        cls.plan=list(planned(cls.manifest))
        cls.envelope=dict(data=cls.manifest,file_sha256='file',path='/tmp/condition_manifest.json')
        cls.entries=[dict(row,status='valid',token=f'token{n}',directory=f'/tmp/e{n}',validated_sha256=f'v{n}',success=n%2==0)
                     for n,row in enumerate(cls.plan) if row['init'] in scope.INITS]
        cls.entries.extend(dict(row,status='valid',token=f'extra{n}',directory=f'/tmp/x{n}',validated_sha256=f'x{n}',success=False)
                           for n,row in enumerate([r for r in cls.plan if r['init']==9][:2]))

    def selected(self,entries):
        return scope.select(self.envelope,json.dumps(dict(entries=entries)).encode())

    def test_scope_exact_main_failed_valid_and_extras_preserved(self):
        result=self.selected(self.entries)
        self.assertEqual(len(result['main']),6480); self.assertEqual(len(result['extras']),2)
        self.assertFalse(result['original_full_complete'])
        self.assertTrue(any(not row['success'] for row in result['main'].values()))
        self.assertEqual({row['init'] for row in result['main'].values()},{6,7,8})
        self.assertEqual(sum(not row['success'] for row in result['extras'].values()),2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); atomic_json(root/'condition_manifest.json',self.manifest)
            atomic_json(root/'episode_ledger.json',dict(entries=self.entries)); atomic_json(root/'drain.json',{})
            from retry_history_audit import manifest_identity
            envelope=manifest_identity(root/'condition_manifest.json')
            selected=scope.select(envelope,(root/'episode_ledger.json').read_bytes())
            value=dict(kind='user_authorized_quick_stage',main_inits=[6,7,8],main_expected=6480,
                source_sha256=sha(scope.__file__),manifest_sha256=sha(root/'condition_manifest.json'),
                ledger_snapshot_sha256=sha(root/'episode_ledger.json'),drain_receipt=str(root/'drain.json'),drain_receipt_sha256=sha(root/'drain.json'),
                main_keys=sorted(selected['main']),extra_keys=sorted(selected['extras']),
                valid_records={**selected['main'],**selected['extras']},retry_history=selected['history'])
            atomic_json(root/'quick_stage_scope.json',value); scope.load_scope(root)
            removed={**value,'extra_keys':[],'valid_records':selected['main']}
            atomic_json(root/'quick_stage_scope.json',removed)
            with self.assertRaises(ValueError): scope.load_scope(root)
            atomic_json(root/'quick_stage_scope.json',{**value,'source_sha256':'changed'})
            with self.assertRaises(ValueError): scope.load_scope(root)
            # An arbitrary receipt cannot stand in for the independently saved drain.
            atomic_json(root/'saved.json',dict(entries=self.entries))
            with self.assertRaises(ValueError): scope.build(root,root/'saved.json',root/'drain.json')

    def test_missing_duplicate_wrong_init_reserved_and_history_rejected(self):
        variants=[self.entries[1:],self.entries+[dict(self.entries[0],token='duplicate')],
                  [dict(self.entries[0],init=9)]+self.entries[1:],
                  [dict(self.entries[0],status='reserved')]+self.entries[1:]]
        for entries in variants:
            with self.assertRaises(ValueError): self.selected(entries)
        old=dict(self.entries[0],token='old',status='invalid')
        with self.assertRaises(ValueError): self.selected([old]+self.entries)

    def test_bootstrap_three_init_known_mean_and_old_five_init_rejected(self):
        values={(s,t,i):.25 for s in SUITES for t in range(10) for i in (6,7,8)}
        self.assertEqual(analysis.bootstrap(values),(25.,25.))
        self.assertEqual(analysis.bootstrap({**values,**{(s,t,i):1 for s in SUITES for t in range(10) for i in (9,10)}}),(None,None))
        values.pop(next(iter(values)))
        self.assertEqual(analysis.bootstrap(values),(None,None))

    def test_metadata_only_checks_and_explicit_nonfull_labels(self):
        condition=dict(model='db-cogact',skip=[],schedule=dict(kind='ddim_custom',num_steps=2,m=.5))
        protocol=dict(H=2,v22_audited_schedules={json.dumps(condition['schedule'],sort_keys=True):dict(nodes=[998,499,'clean'])})
        record=dict(seed=7,actions=[[0]*7]*2,input_hash='input',noise_hash='noise',
            category='measurement',audit=dict(NFE=2,schedule=condition['schedule'],skip=[],nodes=[998,499,'clean'],
                timestep_map=[499,998],timesteps=[998,499],alpha_clean=1.,instrumentation_mode='minimal'),
            timing=dict(measurement_mode='minimal_action_generation'))
        with patch('numpy.load',side_effect=AssertionError('Must not open raw')):
            scope.check_prediction_metadata(record,condition,protocol,7)
        with self.assertRaises(ValueError): scope.check_prediction_metadata(record,condition,protocol,8)
        with self.assertRaises(ValueError): scope.check_prediction_metadata({**record,'actions':[[0]*7]},condition,protocol,7)
        full={**record,'audit':{**record['audit'],'instrumentation_mode':'full','actual_NFE':2,'actual_timesteps':[998,499]}}
        scope.check_prediction_metadata(full,condition,protocol,7)
        for field,value in [('actual_NFE',1),('actual_timesteps',[998]),('timesteps',[998,498])]:
            with self.assertRaises(ValueError): scope.check_prediction_metadata({**full,'audit':{**full['audit'],field:value}},condition,protocol,7)
        result=self.selected(self.entries)
        self.assertFalse(result['original_full_complete']); self.assertEqual(result['original_planned'],10800)
        # The new wrapper cannot masquerade as the signed full-completion path.
        import quick_final_adapter
        source=Path(quick_final_adapter.__file__).read_text()
        self.assertIn("kind='quick_idea_validation_stage_not_full_C_seal'",source)
        self.assertIn('independent_C01_C08_all_passed=False',source)

    def test_media_main_coverage_uses_canonical_receipt_not_index_count(self):
        import quick_final_adapter as final
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); evidence=root/'media'; output=evidence/'batches/token/index.json'
            atomic_json(evidence/'source_manifest.json',{}); atomic_json(evidence/'source_acceptance.json',{})
            row=dict(self.entries[0],result=dict(artifacts={'result.json':'result','video.mp4':'video'},actions=9,frames=10))
            bound={k:row[k] for k in ('key','condition','model','suite','task','init','protocol_hash','schedule','mask','token','validated_sha256')}
            bound.update(manifest_sha256='manifest',canonical_manifest_hash='canonical',directory=row['directory'],
                result_sha256='result',video_sha256='video',actions=9,expected_frames=10)
            saved=dict(passed=True,before={},after={},binding=bound,decoded_frames=10,
                source_manifest_sha256=sha(evidence/'source_manifest.json'),source_acceptance_sha256=sha(evidence/'source_acceptance.json'))
            receipt=evidence/'receipts'/f"{row['key']}.json"; atomic_json(receipt,saved)
            index=dict(kind='independent_media_coverage',partial=True,covered_count=10800,
                source_manifest_sha256=sha(evidence/'source_manifest.json'),source_acceptance_sha256=sha(evidence/'source_acceptance.json'),
                receipts={row['key']:dict(path=str(receipt),sha256=sha(receipt))})
            atomic_json(output,index)
            small_scope=dict(main_keys=[row['key']],valid_records={row['key']:row},manifest_sha256='manifest',manifest_hash='canonical')
            with patch('independent_media_audit.verify_source',return_value={}):
                proof=final.media_coverage(small_scope,root,[output])
                self.assertTrue(proof['main_media_complete']); self.assertFalse(proof['original_full_media_complete'])
                self.assertEqual(proof['main_covered'],1)
                empty=final.media_coverage(small_scope,root,[])
                self.assertEqual(empty['main_missing_keys'],[row['key']])
                atomic_json(receipt,{**saved,'binding':{**bound,'token':'other'}})
                index['receipts'][row['key']]['sha256']=sha(receipt); atomic_json(output,index)
                with self.assertRaises(ValueError): final.media_coverage(small_scope,root,[output])


if __name__=='__main__': unittest.main()
