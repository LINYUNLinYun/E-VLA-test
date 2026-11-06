from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import numpy as np
from . import d2_legacy_history_probe as probe

PARENT=Path('results/idea1_v3_2_e4_supplement/d2_cross_device_bitwise_20261023_run1/manifest.json')


class HistoryBranch(unittest.TestCase):
    def test_register_exact_parent_bank_deadline_and_sources(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(probe.time,'time',return_value=probe.DEADLINE-100):
            path=Path(temporary)/'registered';m=probe.register(PARENT,path,'results/idea1_v3_1/input_bank.pkl')
            self.assertEqual(m['deadline_unix'],1792742812);self.assertEqual(m['legacy_bank']['sha256'],probe.BANK_SHA)
            self.assertEqual(len(m['references']),2);self.assertEqual(m['quality_requests'],0)
            self.assertEqual(m['required_gpu_uuid'],probe.v1.CARDS[0]);self.assertEqual(probe.verify(path/'manifest.json')['manifest_hash'],m['manifest_hash'])
            self.assertIn(str(Path(probe.__file__).resolve()),m['source_files'])

    def test_exact_model_call_order_and_capture_last(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(probe.time,'time',return_value=probe.DEADLINE-100):
            root=Path(temporary);m=probe.register(PARENT,root/'registered','results/idea1_v3_1/input_bank.pkl');output=root/'output';output.mkdir();events=[]
            originals={}
            for ref in m['references']:
                with Path(ref['infer']['path']).open() as stream:originals[ref['mode']]=json.loads(stream.readline())
            class Policy:
                counters={'quality':{},'preflight':{}}
                def legacy_preflight(self,bank):
                    events.append('legacy_preflight')
                    return dict(passed=True,checks=[dict(mode=mode) for mode in probe.ORDER],bank=bank)
                def predict(self,mode,raw,instruction,seed,category):
                    events.append(mode);self.assert_category=category
                    with np.load(next(r for r in m['references'] if r['mode']==mode)['current_stable']['path']) as source:
                        return dict(raw_actions=source['raw'].tolist(),actions=source['decoded'].tolist(),input_hash=originals[mode]['response']['input_hash'],
                            noise_hash=originals[mode]['response']['noise_hash'],observed_calls=dict(ae_layer_calls=180),timing=dict(actual_NFE=10))
            @contextmanager
            def captures():
                events.append('capture_enter');yield {},[np.zeros((50,32),np.float32) for _ in range(10)]
            policy=Policy()
            with patch.object(probe.v1,'capture_buffers',captures):report=probe.execute(policy,m,output)
            self.assertEqual(events,['legacy_preflight','full_d10','full_d10','v14_d10','v14_d10','capture_enter','full_d10','capture_enter','v14_d10'])
            self.assertEqual(report['quality_requests'],0);self.assertEqual(policy.assert_category,'preflight')
            self.assertTrue(all(c['versus']['v1_stable']['raw']['bitwise_equal'] for c in report['uncaptured']))
            self.assertFalse(report['uncaptured'][0]['versus']['historical']['raw']['bitwise_equal'])
            self.assertTrue(all(c['versus_first_uncaptured']['raw']['bitwise_equal'] for c in report['captures']))
            self.assertFalse((output/'continue.json').exists())

    def test_expired_registration_no_gpu_calls(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(probe.time,'time',return_value=probe.DEADLINE+1),self.assertRaises(TimeoutError):
            probe.register(PARENT,Path(temporary)/'blocked','results/idea1_v3_1/input_bank.pkl')


if __name__=='__main__':unittest.main()
