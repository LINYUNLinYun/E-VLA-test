"""Reject stale buffer lengths and shifted RoPE before any future GPU trial."""
import unittest
from .layout_contract import plan,validate_buffers


class LayoutTest(unittest.TestCase):
    def test_identity_preserves_all_positions_and_original_shapes(self):
        p=plan('identity',528)
        self.assertEqual(p.keep_indices,tuple(range(528)))
        self.assertEqual(p.packed_prefix,528)
        self.assertEqual(p.kv_shape,(18,579,256))
        self.assertEqual(p.decoder_absolute_positions,p.decoder_storage_positions)

    def test_patterns_match_camera_budget_and_keep_all_language(self):
        for pattern in ['center_columns','checkerboard']:
            p=plan(pattern,528)
            self.assertEqual(p.packed_prefix,272)
            self.assertEqual(sum(i<256 for i in p.keep_indices),128)
            self.assertEqual(sum(256<=i<512 for i in p.keep_indices),128)
            self.assertEqual(tuple(i for i in p.keep_indices if i>=512),tuple(range(512,528)))
            self.assertEqual(p.kv_shape,(18,323,256))
            self.assertEqual(p.attention_shape,(408,323))

    def test_patterns_have_different_actual_edges_at_equal_budget(self):
        a,b=plan('center_columns',528),plan('checkerboard',528)
        self.assertEqual(a.kv_shape,b.kv_shape)
        self.assertNotEqual(a.keep_indices,b.keep_indices)

    def test_decoder_positions_stay_original_while_storage_moves(self):
        p=plan('center_columns',528)
        self.assertEqual(p.decoder_absolute_positions,tuple(range(528,579)))
        self.assertEqual(p.decoder_storage_positions,tuple(range(272,323)))
        self.assertTrue(validate_buffers(p,p.kv_shape,p.kv_shape,p.attention_shape,True,p.decoder_absolute_positions))
        with self.assertRaises(ValueError):validate_buffers(p,p.kv_shape,p.kv_shape,p.attention_shape,True,p.decoder_storage_positions)

    def test_only_changing_prefix_scalar_with_old_buffers_is_rejected(self):
        original=plan('identity',528);packed=plan('checkerboard',528)
        with self.assertRaises(ValueError):validate_buffers(packed,original.kv_shape,original.kv_shape,original.attention_shape,True,packed.decoder_absolute_positions)

    def test_strided_or_mismatched_attention_buffers_are_rejected(self):
        p=plan('checkerboard',528)
        with self.assertRaises(ValueError):validate_buffers(p,p.kv_shape,p.kv_shape,p.attention_shape,False,p.decoder_absolute_positions)
        with self.assertRaises(ValueError):validate_buffers(p,p.kv_shape,p.kv_shape,(408,579),True,p.decoder_absolute_positions)

    def test_native_softmax_bound_and_unsupported_shapes(self):
        self.assertEqual(plan('identity',973).kv_shape[1],1024)
        for prefix in [511,974,True,528.0]:
            with self.assertRaises(ValueError):plan('center_columns',prefix)
        with self.assertRaises(ValueError):plan('center_columns',528,horizon=49)
        with self.assertRaises(ValueError):plan('unknown',528)

    def test_optional_fp32_logits_follow_the_same_packed_shape(self):
        p=plan('checkerboard',528)
        self.assertTrue(validate_buffers(p,p.kv_shape,p.kv_shape,p.attention_shape,True,p.decoder_absolute_positions,p.attention_shape,True))
        with self.assertRaises(ValueError):validate_buffers(p,p.kv_shape,p.kv_shape,p.attention_shape,True,p.decoder_absolute_positions,(408,579),True)


if __name__=='__main__':unittest.main()
