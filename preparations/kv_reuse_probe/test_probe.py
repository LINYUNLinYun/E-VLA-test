import unittest
import numpy as np
from .probe import adjacent_pairs, mixed_context, kv_metrics, action_metrics


class ProbeTest(unittest.TestCase):
    def test_pair_is_strict_previous_and_gap5(self):
        rows = [dict(request=r, step=s) for r,s in [(4,20),(0,0),(1,5),(3,15),(5,26)]]
        pairs = adjacent_pairs(rows, 'episodeA')
        self.assertEqual([(p['previous']['request'],p['current']['request']) for p in pairs], [(0,1),(3,4)])
        self.assertTrue(all(p['episode']=='episodeA' for p in pairs))
        self.assertEqual(adjacent_pairs([dict(request=0,step=0)], 'episodeB'), [])
        with self.assertRaises(ValueError):
            adjacent_pairs([dict(request=0,step=0)]*2, 'duplicate')

    def test_mixing_inclusive_layer_without_mutation(self):
        current = dict(K=np.ones((18,3,2)), V=np.ones((18,3,2))*2, seq=3, state='current')
        old = dict(K=np.ones((18,3,2))*3, V=np.ones((18,3,2))*4, seq=3, state='old')
        mixed = mixed_context(current, old, 10)
        np.testing.assert_array_equal(mixed['K'][:9], 1)
        np.testing.assert_array_equal(mixed['K'][9:], 3)
        self.assertEqual(mixed['state'], 'current')
        mixed['K'][0,0,0] = 99
        np.testing.assert_array_equal(current['K'], 1)
        np.testing.assert_array_equal(old['K'], 3)
        np.testing.assert_array_equal(mixed_context(current, old, 1)['V'], 4)
        np.testing.assert_array_equal(mixed_context(current, old, 19)['K'], 1)
        with self.assertRaises(ValueError):
            mixed_context(current, dict(old,seq=2), 10)

    def test_metrics_detect_scale_and_action_grouping(self):
        m = kv_metrics(np.ones(6), np.ones(6)*2)
        self.assertAlmostEqual(m['cosine'],1)
        self.assertAlmostEqual(m['relative_l2'],1)
        self.assertAlmostEqual(m['rms_ratio'],2)
        a, b = np.ones((50,7)), np.ones((50,7))
        b[:5,:3] += 2
        b[0,6] = -1
        rows = action_metrics(a,b)
        self.assertEqual(next(r for r in rows if r['horizon']==5 and r['component']=='translation')['rmse'], 2)
        self.assertEqual(next(r for r in rows if r['horizon']==5 and r['component']=='gripper')['sign_mismatch_fraction'], .2)


if __name__ == '__main__':
    unittest.main()
