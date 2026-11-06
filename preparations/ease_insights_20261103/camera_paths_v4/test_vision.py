"""CPU dispatch checks with a nontrivial fake block;no CUDA or real model."""
from types import SimpleNamespace
import unittest
import torch
from preparations.ease_insights_20261103.camera_paths_v3.vision_views import SHAPES,SCRATCH_PER_VIEW
from .common import FULL,SPARSE,SPARSE_SKIP,ALLOCATIONS,conditions,validate_condition
from .vision import checked_layers,run_vision,selected_view

torch.set_num_threads(1)


class Kernel:
    def __init__(self,callback):self.callback=callback
    def __getitem__(self,grid):
        def invoke(**kwargs):return self.callback(grid,kwargs)
        return invoke


class Native:
    def __init__(self):
        self.visits=[];self.project_grids=[];self.conv_views=[]
        self.matmul_512x1152x1152_twopart_bias_res=Kernel(self.project)
        self.combine_1536_1152_twopart=Kernel(lambda grid,args:None)
    def project(self,grid,args):
        self.project_grids.append((grid,args['seq_len']));args['out_ptr'].add_(args['weight_ptr']+1)
    def conv2d_embed_n256_1152_res(self,images,w,b,p,out):
        self.conv_views.append((int(images.shape[0]),tuple(float(x) for x in images[:,0,0,0])))
        out.copy_(images[:,0,0,0].reshape(-1,1,1).expand_as(out))
    def layer_norm_QKV_matmul_n256_1152_3456_bias(self,x,nw,nb,qw,qb,qkv,norm):
        self.visits.append((int(x.shape[0]),nw+1))
    def AttnMultiKey(self,qkv):return torch.zeros(qkv.shape[0],256,1152)
    def matmul_n256_1152_1152_bias_res(self,x,w,b,res,out,scratch):
        if x.shape[0]!=2:raise RuntimeError('Generic B1 projection must never be used')
        out.add_(w+1)
    def layer_norm_matmul_n256_1152_4304_bias_gelu(self,*args):pass
    def matmul_n256_4304_1152_bias_res(self,x,w,b,res,out,scratch):out.add_(2*(w+1))
    def original(self,w,b,n):
        from .vision import restore_oracle
        restore_oracle(self,w,b,(FULL,FULL))


def fixtures():
    b={k:torch.empty(shape) for k,shape in SHAPES.items()};b['vision_x_split_k_buf']=torch.empty(2*SCRATCH_PER_VIEW)
    b['observation_images_normalized'][0].fill_(1);b['observation_images_normalized'][1].fill_(7);b['vision_x'].fill_(-999)
    keys=('vision_pre_attn_norm_w','vision_pre_attn_norm_b','vision_attn_qkv_w','vision_attn_qkv_b','vision_attn_o_w','vision_attn_o_b',
        'vision_pre_ffn_norm_w','vision_pre_ffn_norm_b','vision_ffn_up_w','vision_ffn_up_b','vision_ffn_down_w','vision_ffn_down_b')
    w={k:list(range(27)) for k in keys};w.update(vision_patch_embedding_w=None,vision_patch_embedding_b=None,vision_position_embedding=None)
    return w,b


class VisionAllocation(unittest.TestCase):
    def test_equal_budget_original_layer_identities(self):
        self.assertEqual(len(SPARSE),18);self.assertEqual(len(SPARSE_SKIP),9)
        for layers in ALLOCATIONS.values():self.assertEqual(sum(len(v) for v in layers),45)
        for config in conditions():validate_condition(config)
    def test_native_dense_uses_real_batch2_and54_image_layers(self):
        n=Native();w,b=fixtures();audit=run_vision(n.original,n,w,b,dict(kind='native'))
        self.assertEqual(audit['actual_kernel_input_batches'],[2]);self.assertEqual(audit['image_layer_evaluations'],54)
        self.assertEqual(n.visits,[(2,i) for i in FULL]);self.assertEqual(n.conv_views,[(2,(1.,7.))])
    def test_identity_two_b1_matches_fake_original_b2(self):
        n=Native();w,b=fixtures();run_vision(n.original,n,w,b,dict(kind='native'));expected=b['vision_x'].clone();n.visits.clear()
        audit=run_vision(n.original,n,w,b,dict(kind='identity_b1'))
        self.assertTrue(torch.equal(b['vision_x'],expected));self.assertEqual(n.visits,[(1,i) for _ in (0,1) for i in FULL])
        self.assertEqual(audit['image_layer_evaluations'],54);self.assertEqual(audit['actual_kernel_input_batches'],[1,1])
    def test_both_allocations_match_independent_restore_oracle(self):
        for layers in ALLOCATIONS.values():
            n=Native();w,b=fixtures();ids={k:id(v) for k,v in b.items()};audit=run_vision(n.original,n,w,b,dict(kind='allocated',layers=layers));actual=b['vision_x'].clone()
            self.assertEqual(n.visits,[(1,i) for view in layers for i in view]);self.assertEqual(audit['image_layer_evaluations'],45)
            self.assertTrue(all(id(b[k])==value for k,value in ids.items()));self.assertEqual(n.conv_views,[(1,(1.,)),(1,(7.,))])
            oracle=run_vision(n.original,n,w,b,dict(kind='oracle',layers=layers));self.assertTrue(torch.equal(actual,b['vision_x']))
            self.assertEqual(oracle['image_layer_evaluations'],54);self.assertTrue(oracle['numerical_reference_only'])
            self.assertFalse(audit['numerical_reference_only']);self.assertTrue(audit['fresh_images_consumed'])
    def test_fresh_second_input_does_not_reuse_previous_features(self):
        n=Native();w,b=fixtures();layers=ALLOCATIONS['agent_deep_d2'];run_vision(n.original,n,w,b,dict(kind='allocated',layers=layers));before=b['vision_x'].clone()
        b['observation_images_normalized'][0].add_(3);b['observation_images_normalized'][1].add_(11)
        run_vision(n.original,n,w,b,dict(kind='allocated',layers=layers))
        self.assertTrue(torch.equal(b['vision_x'][0],before[0]+3));self.assertTrue(torch.equal(b['vision_x'][1],before[1]+11))
    def test_real_b1_uses_released_two_part_projection_order(self):
        n=Native();w,b=fixtures();run_vision(n.original,n,w,b,dict(kind='allocated',layers=ALLOCATIONS['agent_deep_d2']))
        self.assertEqual(n.project_grids,[((128,),256)]*45)
    def test_unplanned_or_renumbered_layer_sets_rejected(self):
        for value in [[],[0,1],[1,1],[2,1],[28],[True]]:
            with self.assertRaises(ValueError):checked_layers(value)
        n=Native();w,b=fixtures()
        with self.assertRaises(ValueError):run_vision(n.original,n,w,b,dict(kind='allocated',layers=(FULL,FULL)))
        with self.assertRaises(ValueError):run_vision(n.original,n,w,b,dict(kind='allocated',layers=(FULL,tuple(range(1,19)))))
        self.assertFalse(n.visits)
    def test_batch2_cannot_silently_enter_b1_path(self):
        n=Native();w,b=fixtures()
        with self.assertRaises(ValueError):selected_view(n,w,b,FULL)
    def test_reference_is_explicit_and_has_no_vision_work(self):
        n=Native();w,b=fixtures();raw=torch.ones_like(b['vision_x']);audit=run_vision(n.original,n,w,b,dict(kind='reference',raw=raw))
        self.assertTrue(torch.equal(b['vision_x'],raw));self.assertEqual(audit['image_layer_evaluations'],0);self.assertFalse(audit['fresh_images_consumed'])
        self.assertTrue(audit['numerical_reference_only']);self.assertFalse(n.visits)


if __name__=='__main__':unittest.main()
