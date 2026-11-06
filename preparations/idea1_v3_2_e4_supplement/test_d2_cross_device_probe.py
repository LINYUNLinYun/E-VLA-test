import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import d2_cross_device_probe as probe


class Probe(unittest.TestCase):
    def test_full_idle_guard_and_deadline_without_gpu(self):
        guard=dict(device='0, %s, NVIDIA A100, 1 MiB, 0 %%'%probe.CARDS[0],preexisting_compute_processes='')
        self.assertTrue(probe.validate_idle_guard(guard,0)['passed'])
        for device,processes in [('0, %s, A100, 17 MiB, 0 %%'%probe.CARDS[0],''),
                ('0, %s, A100, 1 MiB, 1 %%'%probe.CARDS[0],''),
                ('1, %s, A100, 1 MiB, 0 %%'%probe.CARDS[0],''),
                (guard['device'],'123, uuid, 20 MiB')]:
            with self.assertRaises(RuntimeError):probe.validate_idle_guard(dict(device=device,preexisting_compute_processes=processes),0)
        with self.assertRaises(TimeoutError):probe.deadline_check(time.time()-1)
        self.assertIn('SPEC_TRITON_INPUT_PREPARE_FAST',probe.ENV_ALLOWLIST)

    def test_outcome_blind_minimum_per_group_suite(self):
        jobs=[dict(group=group,source_index=index,case=dict(mother_suite=suite),success=bool(index%2))
              for group in ('perturbed350','original40') for suite in ('a','b','c','d') for index in (8,2,5)]
        selected=probe.select_jobs(jobs)
        self.assertEqual(len(selected),8);self.assertTrue(all(j['source_index']==2 for j in selected))
        for j in jobs:j['success']=not j['success']
        self.assertEqual(selected,probe.select_jobs(jobs))

    def test_bitwise_full_output_including_signedzero_padding_future(self):
        expected=np.zeros((50,32),np.float32)
        for index,value in [((49,31),1e-8),((0,0),-0.)]:
            actual=expected.copy();actual[index]=value
            result=probe.comparison(actual,expected);self.assertFalse(result['bitwise_equal'])
        actual=expected.copy();actual[49,31]=.03125
        result=probe.comparison(actual,expected)
        self.assertEqual(result['argmax'],[49,31]);self.assertEqual(result['active_first7_max_abs'],0.)

    def test_capture_only_actual_written_buffers_and_restore(self):
        modules={name:ModuleType(name) for name in ('scripts','scripts.spec','scripts.spec.triton','scripts.spec.triton.pi0_infer')}
        for name,module in modules.items():module.__path__=[]
        kernels=modules['scripts.spec.triton.pi0_infer'];modules['scripts.spec.triton'].pi0_infer=kernels
        def vision(weights,buffers,num_views):buffers['vision_x'].fill_(3)
        def vlm(weights,buffers,seq,hidden_trace=None,**kwargs):
            buffers['encoder_x'][:512].fill_(7)
            if hidden_trace is not None:hidden_trace.append(buffers['encoder_x'].clone())
            buffers['encoder_x'].fill_(9)
        def update(x,weight,bias,out,factor):out.add_(1)
        kernels.vision_encoder=vision;kernels.transformer_encoder=vlm;kernels.rms_matmul_k_1024_32_bias_res=update
        buffers={key:torch.zeros(shape) for key,shape in dict(observation_images_normalized=(2,3),observation_state_normalized=(1,32),vision_x=(2,3),encoder_x=(514,2),encoder_K=(18,514,2),encoder_V=(18,514,2)).items()}
        with patch.dict(sys.modules,modules):
            with probe.capture_buffers() as (captured,euler):
                kernels.vision_encoder({},buffers,2);kernels.transformer_encoder({},buffers,514)
                out=torch.zeros((50,32))
                for _ in range(10):kernels.rms_matmul_k_1024_32_bias_res(None,None,None,out,None)
            self.assertTrue(torch.all(captured['vision_final_block_output']==3))
            self.assertTrue(torch.all(captured['projected_prefix_H0']==7));self.assertTrue(torch.all(buffers['encoder_x']==9))
            self.assertEqual(len(euler),10);self.assertTrue(torch.all(euler[-1]==10));self.assertTrue(torch.all(captured['initial_noise_BF16']==0))
            self.assertIs(kernels.vision_encoder,vision);self.assertIs(kernels.transformer_encoder,vlm);self.assertIs(kernels.rms_matmul_k_1024_32_bias_res,update)

    def test_register_fixed16_references_without_model_or_gpu(self):
        parent='results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_20261023T0429Z/manifest.json'
        with tempfile.TemporaryDirectory() as temporary:
            m=probe.register(parent,Path(temporary)/'diagnostic',time.time()+3500)
            self.assertEqual(len(m['references']),16);self.assertEqual(sum(r['capture'] for r in m['references']),2)
            self.assertEqual(m['repeat_count'],2);self.assertEqual(m['quality_requests'],0)
            self.assertEqual(m['production_path'].count('float32'),1)
            self.assertTrue(m['no_quality_marker']);self.assertEqual(set(m['cards']),{0,1})


if __name__=='__main__':unittest.main()
