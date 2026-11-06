import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from . import plus_v14_d2_gpu3_v2 as v2
from .native_envs import digest


class StrictGPU3(unittest.TestCase):
    def test_gpu3_uuid_only_busy_memory_utilization_gates(self):
        def sample(processes='',memory=1,utilization=0,uuid=v2.UUID,index=3):
            with patch.object(v2.subprocess,'check_output',side_effect=[processes,'%s,%s,%s,%s'%(index,uuid,memory,utilization)]) as tool:
                result=v2.sample_gpu3()
                self.assertTrue(all(call.args[0][2]==v2.UUID for call in tool.call_args_list))
                return result
        self.assertTrue(sample()['idle']);self.assertFalse(sample('369809, gpu, 100')['idle'])
        self.assertFalse(sample(memory=17)['idle']);self.assertFalse(sample(utilization=1)['idle'])
        with self.assertRaises(RuntimeError):sample(uuid='GPU-other')

    def test_strict_threshold_padding_is_never_exempt(self):
        actual=np.zeros((50,32),np.float32);expected=actual.copy();actual[4,31]=.03125
        result=v2.detailed_comparison(actual,expected)
        self.assertFalse(result['passed']);self.assertEqual(result['argmax'],[4,31])
        self.assertEqual(result['active_first7_max_abs'],0.);self.assertEqual(result['padding_after7_max_abs'],.03125)
        self.assertEqual(result['per_channel_max_abs'][31],.03125)
        self.assertTrue(v2.detailed_comparison(np.zeros((50,7)),np.zeros((50,7)))['bitwise_equal'])

    def test_manifest_preserves195_cases_all_no_video(self):
        old_path=Path('results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_20261023T0429Z/manifest.json')
        with tempfile.TemporaryDirectory() as temporary:
            new=v2.register(old_path,Path(temporary)/'v2',time.time()+3600)
            old=json.loads(old_path.read_text())
            self.assertEqual(len(new['jobs']),195);self.assertTrue(all(not j['video'] for j in new['jobs']))
            self.assertEqual([(j['source_index'],j['case_hash']) for j in old['jobs']],[(j['source_index'],j['case_hash']) for j in new['jobs']])
            self.assertEqual(new['required_gpu_uuid'],v2.UUID);self.assertEqual(new['required_gpu_index'],3)
            self.assertEqual(new['startup_gate']['historical_D10_raw_max_abs'],1e-5)
            self.assertEqual(new['manifest_hash'],digest({k:v for k,v in new.items() if k!='manifest_hash'}))
            self.assertEqual(old,json.loads(old_path.read_text()))


if __name__=='__main__':unittest.main()
