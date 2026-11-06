import unittest
from unittest.mock import patch
from .policy import reuse_this_round
from .backend import Backend,OriginalBackend
from preparations.idea1_v3.policy import V14


class TestFollowup(unittest.TestCase):
    def test_independent_and_combined_round_selection(self):
        self.assertEqual([reuse_this_round({},r) for r in (0,1)],[True,True])
        self.assertEqual([reuse_this_round({'first_only':True},r) for r in (0,1)],[True,False])

    def test_second_round_executes_current_v14_without_old_snapshot(self):
        b=Backend.__new__(Backend); b.reuse_allowed=False
        with patch.object(OriginalBackend,'prepare',return_value=({},{})) as fresh,patch.object(OriginalBackend,'prepare_hybrid') as old:
            c,t=b.prepare_hybrid('obs','prompt',object(),16)
            fresh.assert_called_once_with('obs','prompt',V14); old.assert_not_called()
            self.assertEqual(t['old_kv_layers'],0)

    def test_layer17_uses_existing_exact_truncation(self):
        b=Backend.__new__(Backend); b.reuse_allowed=True; full=object()
        with patch.object(OriginalBackend,'prepare_hybrid',return_value='ok') as old:
            self.assertEqual(b.prepare_hybrid('obs','prompt',full,17),'ok')
            old.assert_called_once_with('obs','prompt',full,17)
