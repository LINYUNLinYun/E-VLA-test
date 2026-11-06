"""Check camera/token masks and residual definitions independently of timing."""
import unittest

import numpy as np

from .pooling import pool


class PoolingTests(unittest.TestCase):
    def test_padding_never_changes_features_or_statistics(self):
        token=np.arange(8,dtype=np.float32)[None,None,:,None]
        hidden=np.repeat(np.arange(19,dtype=np.float32)[:,None,None,None]+token,4,axis=-1)
        attention=hidden[:-1]+2
        prefix=np.array([[True,True,True,True,False,False,True,False]])
        images={name:dict(hidden=np.zeros((28,1,2,4),np.float32),projected=np.zeros((1,2,4),np.float32))
                for name in ('base','wrist','dummy')}
        result=dict(vision=images,prefix_mask=prefix,vlm=dict(hidden=hidden,
                    post_attention=attention,attn_update=np.full_like(attention,2),
                    ffn_update=np.full_like(attention,-1)))
        first,stats=pool(result)
        for key,array in result['vlm'].items():
            array[:,:,~prefix[0],:]=123456.
        second,other_stats=pool(result)
        self.assertEqual(set(first),set(second))
        for key in first:np.testing.assert_array_equal(first[key],second[key])
        for key in stats:np.testing.assert_array_equal(stats[key],other_stats[key])
        self.assertFalse(any('/dummy/' in key for key in first))
        self.assertEqual(float(first['vlm/base/hidden'][0,0]),.5)
        self.assertEqual(float(first['vlm/wrist/hidden'][0,0]),2.5)
        self.assertEqual(float(first['vlm/vision/hidden'][0,0]),1.5)
        self.assertEqual(float(first['vlm/language/hidden'][0,0]),6.)
        self.assertEqual(float(first['vlm/vision/ffn_update'][0,0]),-1.)
        self.assertAlmostEqual(float(stats['vlm/vision/block_relative_norm'][0]),4/np.sqrt(56))


if __name__=='__main__':unittest.main()
