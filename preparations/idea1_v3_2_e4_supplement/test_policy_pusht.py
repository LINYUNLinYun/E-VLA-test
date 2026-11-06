"""CPU contract tests, not a native-model load or benchmark success claim."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from .policy_pusht import (NativePushTPolicy, denormalize_xy, normalize_xy,
                          read_xy_stats, strict_key_gate, tensor_header,
                          validate_config)


class PushTContractTests(unittest.TestCase):
    def test_released_horizon_and_xy_contract(self):
        config=dict(type='pi0',chunk_size=50,n_action_steps=50,num_inference_steps=10,
                    max_state_dim=32,max_action_dim=32,paligemma_variant='gemma_2b',
                    action_expert_variant='gemma_300m',tokenizer_max_length=48,
                    input_features={'observation.state':{'shape':[2]},'observation.image':{'shape':[3,96,96]}},
                    output_features={'action':{'shape':[2]}},
                    normalization_mapping={'VISUAL':'IDENTITY','STATE':'MEAN_STD','ACTION':'MEAN_STD'})
        self.assertEqual(validate_config(config)['chunk_size'],50)
        config['n_action_steps']=5
        with self.assertRaises(ValueError):validate_config(config)

    def test_normalization_has_native_pixel_units(self):
        stats={'mean':[200.,300.],'std':[100.,50.]}
        self.assertEqual(normalize_xy([200.,300.],stats),[0.,0.])
        self.assertEqual(denormalize_xy([1.,-1.],stats),[300.,250.])
        with self.assertRaises(ValueError):normalize_xy([1.,2.,3.],stats)
        with self.assertRaises(ValueError):denormalize_xy([float('nan'),0.],stats)

    def test_strict_keys_and_shapes(self):
        header={'a':{'shape':[2,3]}}
        self.assertEqual(strict_key_gate(header,{'a':(2,3)})['missing'],[])
        for expected in [{'a':(3,2)},{'b':(2,3)},{'a':(2,3),'b':(1,)}]:
            with self.assertRaises(ValueError):strict_key_gate(header,expected)

    def test_tied_alias_requires_proven_native_identity(self):
        header={'a':{'shape':[2,3]},'__metadata__':{'b':'a'}}
        expected={'a':(2,3),'b':(2,3)}
        with self.assertRaises(ValueError):strict_key_gate(header,expected)
        self.assertEqual(strict_key_gate(header,expected,{'b':'a'})['missing'],[])

    def test_small_real_safetensors_layout(self):
        with tempfile.TemporaryDirectory() as d:
            header={};values=[]
            for feature in ('observation.state','action'):
                for name in ('mean','std'):
                    start=4*len(values);values.extend([1.,2.])
                    header[feature+'.'+name]={'dtype':'F32','shape':[2],'data_offsets':[start,4*len(values)]}
            body=json.dumps(header).encode();p=Path(d)/'stats.safetensors'
            p.write_bytes(struct.pack('<Q',len(body))+body+struct.pack('<8f',*values))
            self.assertEqual(read_xy_stats(p)['action']['std'],[1.,2.])
            p.write_bytes(struct.pack('<Q',5*1024*1024))
            with self.assertRaises(ValueError):tensor_header(p)

    def test_no_tokenizer_no_model_or_network(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(PermissionError):NativePushTPolicy.load_cpu(d,Path(d)/'missing')

    def test_cpu_adapter_orientation_horizon_reset_and_seed(self):
        import torch
        class Model:
            resets=0
            def reset(self):self.resets+=1
            def predict_action_chunk(self,batch):
                self.batch=batch
                return torch.zeros((1,50,2))
        class Tokenizer:
            def __call__(self,text,**kwargs):
                self.text,self.kwargs=text,kwargs
                return {'input_ids':torch.ones((1,48),dtype=torch.long),'attention_mask':torch.ones((1,48),dtype=torch.long)}
        model=Model();tok=Tokenizer();stats={'mean':[200.,300.],'std':[100.,50.]}
        policy=NativePushTPolicy(model,tok,{'observation.state':stats,'action':stats})
        image=torch.zeros((96,96,3),dtype=torch.uint8);image[0,0,0]=255
        raw={'image':image,'state2':[200.,300.]};a=policy.predict(raw,'Push block',42)
        self.assertEqual(a['actions'],[[200.,300.]]*50);self.assertEqual(a['execute'],50)
        self.assertEqual(tok.text,['Push block\n']);self.assertEqual(tok.kwargs['max_length'],48)
        self.assertEqual(float(model.batch['observation.image'][0,0,0,0]),1.)
        self.assertEqual(float(model.batch['observation.image'][0,0,-1,-1]),0.)
        self.assertEqual(model.batch['observation.state'].tolist(),[[0.,0.]])
        policy.reset();self.assertEqual(model.resets,2)
        b=policy.predict(raw,'Push block\n',42);self.assertEqual(a,b)


if __name__=='__main__':unittest.main()
