import json
from pathlib import Path
import unittest
import numpy as np
from .design import ROOT
from .features import decide,feature_after_observe
from .policy import Policy
from preparations.idea1_v3_3.test_policy import Backend
from preparations.idea1_v3_3.rules_c1 import gate


class Tests(unittest.TestCase):
    def test_old_c5_real80_no_residual_history(self):
        count=0;cases=0
        for p in (ROOT/'results/idea1_v3_3/E2_a30_b0.1_c5/episodes').glob('*/task*/init*/result.json'):
            if '.invalid.' in str(p) or not p.with_name('complete.json').exists():continue
            r=json.loads(p.read_text())
            if r['case']['init'] not in (6,7):continue
            cases+=1;tele=[json.loads(l) for l in p.with_name('telemetry.jsonl').read_text().splitlines()]
            for line in p.with_name('infer.jsonl').read_text().splitlines():
                v=json.loads(line)
                if 'distance' not in v['route']:continue
                a=[s['action'] for s in tele[1:v['step']+1]];q=[s['eef_quat'] for s in tele[1:v['step']+1]]
                actual=decide(dict(kind='abc',angle=30,change=.1,pose=5),a,q,[])
                self.assertEqual(actual[:2],gate(a,q,30,.1,5));self.assertEqual(actual[1],v['route']['gate_blocked']);count+=1
        self.assertEqual(cases,80);self.assertGreater(count,1900)

    def test_only_metric_collects_and_filter_state_is_bitwise_equal(self):
        old=Policy(Backend());new=Policy(Backend())
        for p,kind in [(old,'abc'),(new,'metric')]:
            p.configure(dict(model='pi0',experiment='baseline',kind=kind))
            r=p.predict({},'x','libero_goal',7)
            ack=p.acknowledge(r['actions'][:5],np.tile([0,0,0,1],(5,1)))
            self.assertGreaterEqual(ack['ack_policy_wall_ms'],0)
        self.assertEqual(old.feature_history,[]);self.assertEqual(len(new.feature_history),5)
        self.assertTrue(np.array_equal(old.kalman.x,new.kalman.x));self.assertTrue(np.array_equal(old.kalman.P,new.kalman.P))
        a=old.kalman.forecast(50);b=new.kalman.forecast(50)
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)))

    def test_metric_boundary_and_floor(self):
        a=np.zeros((5,7));q=np.tile([0,0,0,1],(5,1));h=[dict(forecast_error_rms=.15)]*5
        c=dict(kind='metric',metric='forecast_error_rms_max5',threshold=.15,pose=5)
        self.assertTrue(decide(c,a,q,h)[0]);self.assertFalse(decide(c,a,q,[dict(forecast_error_rms=np.nextafter(.15,np.inf))]*5)[0])
        c.update(metric='forecast_error_relative_max5',threshold=.5)
        self.assertTrue(decide(c,a,q,[dict(forecast_error_rms=.01)]*5)[0])
        self.assertFalse(decide(c,np.zeros((5,3)),q,h)[0])


if __name__=='__main__':unittest.main()
