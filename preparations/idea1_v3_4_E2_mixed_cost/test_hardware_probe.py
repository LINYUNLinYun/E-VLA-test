"""CPU-only schedule, route, aggregation, occupancy, and request-cap checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import hardware_probe as hp


class HardwareProbeTests(unittest.TestCase):
    def test_frozen_4976_schedule_and_balanced_order(self):
        self.assertEqual(hp.planned_requests(), 4976)
        self.assertLess(hp.planned_requests(), hp.MAX_REQUESTS)
        for sample in range(8):
            for layer in hp.LAYERS:
                orders = [hp.route_order(sample, layer, repeat) for repeat in range(4)]
                self.assertEqual(orders.count(hp.ROUTES), 2)
                self.assertEqual(orders.count(tuple(reversed(hp.ROUTES))), 2)
                self.assertTrue(all(set(order) == set(hp.ROUTES) for order in orders))

    def test_route_kwargs_separate_A_F_and_compensation(self):
        history, older = object(), object()
        self.assertEqual(hp.route_kwargs('keep', (8,)), {})
        a = hp.route_kwargs('A+', (8,), history, older)
        self.assertEqual(a['skip_attn'], (8,))
        self.assertEqual(a['skip_ffn'], ())
        self.assertIs(a['history'], history)
        self.assertIs(a['older'], older)
        self.assertEqual(a['ratio'], .8)
        f = hp.route_kwargs('F0', (8,))
        self.assertEqual(f, dict(skip_attn=(), skip_ffn=(8,)))
        both = hp.route_kwargs('AF+', (8,), history, older)
        self.assertEqual(both['skip_attn'], both['skip_ffn'])
        with self.assertRaises(ValueError):
            hp.route_kwargs('AF+', (8,))

    def test_negative_readings_preserved_and_profile_semantics(self):
        pairs, profiles = [], []
        for sample in range(8):
            routes = {name: [] for name in ('keep', 'A+', 'F+', 'AF+')}
            for layer in hp.LAYERS:
                for stage, ms in (('qkv', .2), ('attn', .3), ('ffn', 1.3)):
                    routes['keep'].append(dict(layer=layer, stage=stage, ms=ms))
                routes['A+'].extend([dict(layer=layer, stage='kv', ms=.1), dict(layer=layer, stage='patch_attn', ms=.05)])
                routes['F+'].append(dict(layer=layer, stage='patch_ffn', ms=.04))
                routes['AF+'].extend([dict(layer=layer, stage='kv', ms=.11), dict(layer=layer, stage='patch_attn', ms=.051), dict(layer=layer, stage='patch_ffn', ms=.041)])
                for repeat in range(4):
                    for kind, net, gross, overhead in (('A', .35, .4, .05), ('F', 1.25, 1.3, .05), ('AF', 1.7, 1.8, .1)):
                        if layer == 17 and kind == 'A':
                            net = -.1
                        row = dict(layer=layer, kind=kind, sample=sample, repeat=repeat,
                            net_saved_ms=net, gross_saved_ms=gross, patch_overhead_ms=overhead, request_net_saved_ms=net)
                        if kind == 'AF':
                            row['af_nonadditivity_ms'] = .1
                        pairs.append(row)
            profiles.append(dict(sample=sample, routes=routes))
        rows, distribution = hp.aggregate_costs(pairs, profiles)
        self.assertEqual(len(rows), 17)
        self.assertAlmostEqual(rows[0]['attn_saved_ms'], .35)
        self.assertAlmostEqual(rows[-1]['attn_saved_ms'], -.1)
        self.assertFalse(rows[-1]['attn_eligible'])
        self.assertTrue(rows[0]['attn_eligible'])
        self.assertAlmostEqual(rows[0]['profile']['attention_complete_ms']['median_ms'], .5)
        self.assertAlmostEqual(rows[0]['profile']['attention_removable_before_patch_ms']['median_ms'], .4)
        self.assertAlmostEqual(rows[0]['profile']['af_profile_net_saved_ms']['median_ms'], 1.598)
        self.assertAlmostEqual(distribution['complete_attention_over_ffn_ratio']['median'], .5 / 1.3)
        self.assertEqual(distribution['attn_saved_ms']['minimum'], -.1)

    def test_persistent_request_cap_includes_attempted_calls(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            path = Path(directory) / 'requests.jsonl'
            with patch.object(hp, 'MAX_REQUESTS', 3):
                ledger = hp.Ledger(path)
                for index in range(3):
                    self.assertEqual(ledger.consume(phase='warmup'), index + 1)
                with self.assertRaises(RuntimeError):
                    ledger.consume(phase='hot')
                ledger.close()
            self.assertEqual(len(path.read_text().splitlines()), 3)
            with self.assertRaises(FileExistsError):
                hp.Ledger(path)

    def test_gpu_occupancy_refuses_others_and_non_4090(self):
        props = 'NVIDIA GeForce RTX 4090, GPU-test, 24564, 580.95\n'
        with patch.object(hp.subprocess, 'check_output', side_effect=[props, '']):
            self.assertEqual(hp.gpu_snapshot(3)['uuid'], 'GPU-test')
        with patch.object(hp.subprocess, 'check_output', side_effect=[props, '123, other-job, 1000']):
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                hp.gpu_snapshot(3)
        with patch.object(hp.subprocess, 'check_output', side_effect=['NVIDIA A100, GPU-test, 80000, 580.95\n', '']):
            with self.assertRaisesRegex(RuntimeError, 'RTX 4090'):
                hp.gpu_snapshot(3)


if __name__ == '__main__':
    unittest.main()
