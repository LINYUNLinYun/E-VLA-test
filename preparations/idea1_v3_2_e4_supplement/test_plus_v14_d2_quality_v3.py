from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from . import plus_v14_d2_quality_v3 as v3

BASE='results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_20261023T0429Z/manifest.json'
PROBE='results/idea1_v3_2_e4_supplement/d2_cross_device_bitwise_20261023_run1/manifest.json'
HISTORY='results/idea1_v3_2_e4_supplement/d2_legacy_history_20261023_run1/manifest.json'
CALLS=dict(vision_encoder_calls=1,vlm_prefix_calls=1,ae_layer_calls=36,learned_head_calls=0,verify_calls=0,constructor_full_record_calls=0)


class QualityV3(unittest.TestCase):
    def register(self,root):return v3.register(BASE,PROBE,HISTORY,root,time.time()+3600)

    def test_manifest_all195_preserved_gpu0_novideo_new_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'quality';m=self.register(root);base=json.loads(Path(BASE).read_text())
            self.assertEqual(len(m['jobs']),195);self.assertEqual(m['condition_count'],195)
            self.assertTrue(all(not j['video'] for j in m['jobs']));self.assertEqual(m['assigned_gpus']['0'],v3.UUID)
            self.assertEqual([(j['case_hash'],j['source_index']) for j in m['jobs']],[(j['case_hash'],j['source_index']) for j in base['jobs']])
            self.assertEqual(v3.verify(root/'manifest.json')['manifest_hash'],m['manifest_hash'])
            self.assertGreater(m['deadline_unix'],time.time());self.assertEqual(m['repeat_count'],2)
            first,second=v3.shard_jobs(m,0),v3.shard_jobs(m,1)
            self.assertEqual((len(first),len(second)),(98,97))
            keys=lambda rows:{(j['group'],j['source_index']) for j in rows}
            self.assertFalse(keys(first)&keys(second));self.assertEqual(keys(first)|keys(second),keys(m['jobs']))

    def test_gate_order_legacy32historical_then_d2_raw32_bitwise(self):
        with tempfile.TemporaryDirectory() as temporary:
            m=self.register(Path(temporary)/'quality');events=[];responses={}
            for ref in m['references']:
                with Path(ref['infer']['path']).open() as stream:old=json.loads(stream.readline())
                with np.load(ref['observation']['path']) as saved:instruction=str(saved['instruction'])
                responses[(ref['mode'],instruction,old['seed'])]=old
            class Production:
                def __init__(self,*args,**kwargs):pass
                def legacy_preflight(self,bank):events.append('legacy6');return dict(passed=True,checks=[dict(mode=mode) for mode in v3.ORDER])
                def predict(self,mode,raw,instruction,seed,category):
                    events.append(mode);self.asserted=category
                    old=responses[(mode,instruction,seed)]
                    return dict(raw_actions=old['response']['raw_actions'],actions=old['actions'],observed_calls=dict(ae_layer_calls=180))
            class D2:
                def __init__(self,*args,**kwargs):self.protocol={};self.gate_passed=False
                def predict(self,mode,raw,instruction,seed,category):
                    events.append('D2wrapper');assert category=='preflight'
                    return dict(raw_actions=np.zeros((50,32)),actions=np.zeros((50,7)),observed_calls=CALLS)
            def prepare(backend,raw,*args,**kwargs):
                events.append('D2native');self.assertEqual(raw['state8'].dtype,np.float32);return {},{}
            def sample(*args,**kwargs):return np.zeros((50,7)),np.zeros((50,32)),dict(actual_NFE=2,nodes=[1.,.5,0.])
            @contextmanager
            def observed():yield Counter(CALLS)
            with patch.object(v3,'TransferPolicy',Production),patch.object(v3,'V14D2Policy',D2),patch.object(v3.RawTritonBackend,'prepare',prepare),patch.object(v3.RawTritonBackend,'sample',sample),patch.object(v3,'route_observations',observed):
                policy,gate=v3.strict_gate(object(),m,Path(temporary)/'gate')
            expected=['legacy6']+[ref['mode'] for ref in m['references'] for _ in range(2)]+['D2native','D2wrapper']*2
            self.assertEqual(events,expected);self.assertTrue(gate['passed']);self.assertTrue(policy.gate_passed)
            self.assertEqual(gate['quality_requests'],0);self.assertEqual(len(gate['D10']),16)

    def test_subthreshold_padding_difference_is_rejected(self):
        reference=np.zeros((50,32),np.float32);actual=reference.copy();actual[49,31]=1e-8
        pair=dict(raw=v3.diagnostic.comparison(actual,reference),decoded=v3.diagnostic.comparison(np.zeros((50,7)),np.zeros((50,7))))
        self.assertFalse(v3.all_equal([pair]));self.assertLess(pair['raw']['max_abs'],1e-5)


if __name__=='__main__':unittest.main()
