import types
import unittest
import numpy as np
from preparations.idea1_v4_1_warmstart.schedule import mix
from .common import conditions
from .policy import Policy,shift_prediction


class Raw:
    def float(self):return self
    def cpu(self):return self
    def numpy(self):return np.zeros((50,32),np.float32)


class Coordinates:
    def __init__(self,transform,state):
        self.offset=np.r_[np.repeat(float(state[0]),6),0.].astype(np.float32)
        self.inverse=np.diag(np.linspace(1.,2.,7,dtype=np.float32))
    def encode_commands(self,commands,raw):
        raw[:,:7]=(commands-self.offset)@self.inverse;return raw


class Backend:
    def __init__(self):self.calls=[];self.adapter=types.SimpleNamespace(backend=types.SimpleNamespace(output_transform=None))
    def prepare_residual(self,obs,prompt,skip=(),**kwargs):return dict(residuals=obs,transformed_state=np.full(7,obs)),{}
    def sample(self,*args):raise AssertionError('Full AE forbidden')


def output():
    action=np.arange(50*7,dtype=np.float32).reshape(50,7)/10
    action[:,6]=np.where(np.arange(50)<9,-1.,1.)
    return action,Raw(),{}


def gaussian(b,ctx,seed,steps):b.calls.append(('gaussian',steps,1.));return output()
def initial(b,ctx,value,steps,start):b.calls.append(('initial',steps,start));return output()


class FuturePriorTests(unittest.TestCase):
    def test_shift_uses_actual_execution_and_last_prediction_tail(self):
        action=output()[0]
        for r in (1,2,5):
            shifted=shift_prediction(action,r)
            self.assertTrue(np.array_equal(shifted[:-r],action[r:]))
            self.assertTrue(np.array_equal(shifted[-r:],np.repeat(action[-1:],r,axis=0)))
        for r in (0,6):
            with self.assertRaises(ValueError):shift_prediction(action,r)

    def test_partial_ack_current_coordinates_one_ae_and_reset(self):
        for cfg in conditions():
            b=Backend();p=Policy(b,gaussian,initial,Coordinates);p.configure(cfg)
            previous=None;routes=[];lengths=(2,5,1,5)
            for i in range(4):
                result=p.predict(i,'p','s',40+i);routes.append(result['route'])
                if i:
                    prior=result['arrays']['prior_commands']
                    shifted=shift_prediction(previous['actions'],lengths[i-1])
                    self.assertTrue(np.array_equal(prior[:,6],shifted[:,6]))
                    if cfg['initial']=='shifted_chunk':self.assertTrue(np.array_equal(prior,shifted))
                    else:
                        self.assertEqual(p.kalman.filter.count,sum(lengths[:i]))
                        self.assertTrue(np.array_equal(prior[:,:6],p.kalman.last_forecast[:,:6]))
                    coord=Coordinates(None,[i]);encoded=coord.encode_commands(prior,np.zeros((50,32),np.float32))
                    self.assertTrue(np.array_equal(encoded[:5],result['arrays']['prior_encoded_prefix']))
                    self.assertTrue(np.array_equal(mix(encoded,40+i,.1,1.)[:5],result['arrays']['prior_initial_prefix']))
                    self.assertEqual(result['route']['shift_executed_length'],lengths[i-1])
                    self.assertEqual(result['route']['prior_source_round'],i-1)
                with self.assertRaises(RuntimeError):p.predict(i,'p','s',40+i)
                with self.assertRaises(ValueError):p.acknowledge(np.ones((5,7),np.float32)*999)
                p.acknowledge(result['actions'][:lengths[i]]);previous=result
            self.assertEqual(b.calls,[('gaussian',1,1.)]+[('initial',1,.1)]*3)
            self.assertEqual([r['residual_refresh'] for r in routes],[True,False,False,True])
            self.assertEqual(p.counts['full'],0);self.assertEqual(p.counts['verify'],0)
            p.configure(cfg);self.assertIsNone(p.future);self.assertIsNone(p.pending);self.assertEqual(p.history,[])
            result=p.predict(0,'p','s',40);self.assertEqual(result['route']['initial_state'],'gaussian')


if __name__=='__main__':unittest.main()
