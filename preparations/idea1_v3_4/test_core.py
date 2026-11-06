import unittest
import numpy as np
from .backend import shifted_tail,prefix_encoder
from .manifest import cases,conditions


class CoreTests(unittest.TestCase):
    def test_case_budget_and_paired_init(self):
        cs=conditions()
        self.assertEqual(sum(c['n'] for c in cs if c['experiment']=='E1'),920)
        for c in cs:
            rows=cases(c['n']); self.assertEqual(len(rows),c['n'])
            self.assertEqual(len({(x['suite'],x['task'],x['init']) for x in rows}),c['n'])
            self.assertEqual([x for x in rows if x['init']==14],cases(40))
            for shard in range(3):
                self.assertEqual(sum(x['task']%3==shard for x in rows),c['n']//10*(4 if shard==0 else 3))

    def test_actual_execution_shifts_remaining_without_wrap(self):
        raw=np.repeat(np.arange(50)[:,None],32,axis=1)
        a=shifted_tail(raw,3)
        np.testing.assert_array_equal(a[:47],raw[3:])
        np.testing.assert_array_equal(a[-3:],np.repeat(raw[-1:],3,axis=0))
        self.assertFalse(np.shares_memory(raw,a))
        with self.assertRaises(ValueError): shifted_tail(raw,50)

    def test_prefix_truly_stops_projections_and_unused_hidden(self):
        class Kernels:
            def __init__(self): self.projections=[]; self.hidden=0
            def layer_norm_matmul_n256_1152_2048_bias(self,*args): pass
            def rms_matmul_n_2048_2560_qkv_rope(self,*args): self.projections.append(('qkv',args[1]))
            def rms_matmul_n_2048_2560_kv_rope(self,*args): self.projections.append(('kv',args[1]))
            def AttnSingleKey(self,*args): return None
            def matmul_n_2048_2048_res(self,*args): self.hidden+=1
            def rms_matmul_n_2048_16384_gate(self,*args): pass
            def matmul_n_16384_2048_res(self,*args): pass
        w={key:np.arange(18) for key in ('encoder_attn_qkv_w','encoder_attn_o_w','encoder_ffn_gate_w','encoder_ffn_up_w','encoder_ffn_down_w')}
        w.update({key:None for key in ('vision_final_norm_w','vision_final_norm_b','encoder_multi_modal_projector_w','encoder_multi_modal_projector_b')})
        b={key:None for key in ('vision_x','encoder_x','vision_x_norm','encoder_rope_weights','encoder_Q','encoder_x_norm','encoder_hidden')}
        b.update(encoder_K=np.zeros((18,8)),encoder_V=np.zeros((18,8)))
        k=Kernels(); kept=frozenset(set(range(18))-{2,3,8,10})
        prefix_encoder(k,w,b,8,16,kept)
        self.assertEqual([x[1] for x in k.projections],list(range(15)))
        self.assertEqual(k.hidden,10)
        self.assertEqual([x[1] for x in k.projections if x[0]=='kv'],[2,3,8,10])

if __name__=='__main__': unittest.main()
