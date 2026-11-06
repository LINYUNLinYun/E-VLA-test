import copy
import unittest
from unittest.mock import patch
from performance_window_audit import check_slots, verify_time, verify_rates, unique_rows


class SlotTests(unittest.TestCase):
    def test_comparison_self_reported_times_rejected(self):
        receipts = [dict(read_completed_utc='2026-10-18T11:00:00+00:00'), dict(read_completed_utc='2026-10-18T11:10:00+00:00')]
        data = dict(baseline=dict(snapshot_read_completed_utc=receipts[0]['read_completed_utc']),
            endpoint=dict(snapshot_read_completed_utc=receipts[1]['read_completed_utc']), elapsed_s=600)
        self.assertEqual(verify_time(data, receipts), 600)
        data['baseline']['snapshot_read_completed_utc'] = '2026-10-18T12:00:00+00:00'
        data['endpoint']['snapshot_read_completed_utc'] = '2026-10-18T12:10:00+00:00'
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            verify_time(data, receipts)

    def test_rate_eta_and_duplicate_mutations_rejected(self):
        count = dict(expected_episodes=2, full_scope_valid_at_endpoint=1, full_scope_reserved_at_endpoint=0,
            remaining_not_yet_valid=1, new_valid=1, new_actions=10, new_predictions=2,
            observed_valid_per_hour=6, observed_actions_per_second=10/600, observed_predictions_per_second=2/600,
            projected_hours_remaining_at_observed_rate=1/6)
        aggregate = {k:v for k,v in count.items() if k.startswith(('new_', 'observed_'))}
        aggregate.update(max_projected_remaining_hours=1/6, bottleneck_model_by_projected_remaining_hours='pi0')
        data = dict(per_model={'pi0': count}, aggregate=aggregate)
        endpoint = dict(entries=[dict(status='valid', model='pi0')])
        with patch('performance_window_audit.planned', return_value=[dict(model='pi0')]*2):
            verify_rates(data, {}, endpoint, 600)
            for field in ('observed_valid_per_hour', 'projected_hours_remaining_at_observed_rate'):
                bad = copy.deepcopy(data)
                bad['per_model']['pi0'][field] += 1
                with self.assertRaisesRegex(ValueError, 'rate or remaining ETA'):
                    verify_rates(bad, {}, endpoint, 600)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            unique_rows([dict(key='a', token='b')]*2)

    def fixture(self):
        rows = [dict(slot='0', client_pid=62326, key='a', token='t0', actions=10, predictions=2, frames=11),
                dict(slot='1', client_pid=62327, key='b', token='t1', actions=20, predictions=4, frames=21)]
        counts = dict(client_pids={'0': 62326, '1': 62327}, new_valid=2, new_actions=30, new_predictions=6, new_frames=32,
            slots={str(i): dict(pid=row['client_pid'], new_valid=1, actions=row['actions'], predictions=row['predictions'],
                frames=row['frames'], new_keys=[row['key']], new_tokens=[row['token']]) for i, row in enumerate(rows)})
        return rows, counts, dict(client_pids={'0': 62326, '1': 62327})

    def test_correct_slot_sums(self):
        rows, counts, session = self.fixture()
        check_slots('pi0', counts, rows, session)

    def test_pid_slot_transposition_rejected(self):
        rows, counts, session = self.fixture()
        counts['slots'] = {str(value['pid']): dict(value, pid=int(slot)) for slot, value in counts['slots'].items()}
        with self.assertRaisesRegex(ValueError, 'Slot keys'):
            check_slots('pi0', counts, rows, session)

    def test_swapped_pid_and_wrong_sum_rejected(self):
        rows, counts, session = self.fixture()
        bad = copy.deepcopy(counts)
        bad['slots']['0']['pid'] = 62327
        with self.assertRaisesRegex(ValueError, 'Slot/PID'):
            check_slots('pi0', bad, rows, session)
        counts['new_predictions'] += 1
        with self.assertRaisesRegex(ValueError, 'slot sum'):
            check_slots('pi0', counts, rows, session)


if __name__ == '__main__':
    unittest.main()
