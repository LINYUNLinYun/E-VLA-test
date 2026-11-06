"""Independent acceptance checks. These do not grant real-model launch approval."""
import csv
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from common import Ledger, SUITES, atomic_json, conditions, digest, planned, seed
from common import gpu_lease
from runtime_analysis import analyze, native_control
from schedules import ddim_indices, flow_schedule, normalize_schedule


def reserve_valid_failure(directory, row, queue):
    ledger = Ledger(directory)
    token = ledger.reserve(row)
    if token:
        ledger.update(token, status='valid', success=False)
    queue.put(bool(token))


def independently_recompute_real_parity(attempt, model):
    """Read-only evidence audit; never loads a model or grants acceptance."""
    import numpy as np
    root = Path(attempt)
    native = root/'native'/model
    checks = [json.loads(x) for x in (native/'parity_checks.jsonl').read_text().splitlines()]
    predictions = [json.loads(x) for x in (native/'prediction_ledger.jsonl').read_text().splitlines()]
    references = [json.loads(x) for x in (native/'native_reference_ledger.jsonl').read_text().splitlines()]
    ordered = sorted(predictions + references, key=lambda row: row['call'])
    if len({row['call'] for row in ordered}) != len(ordered):
        raise AssertionError('Duplicate real prediction call identity')
    offset = 0
    results = []
    def compare(label, left, right):
        with np.load(left['prediction_path'], allow_pickle=False) as z:
            a = z['actions'].copy()
        with np.load(right['prediction_path'], allow_pickle=False) as z:
            b = z['actions'].copy()
        with np.load(left['noise_path'], allow_pickle=False) as z:
            na = z['initial_noise'].copy()
        with np.load(right['noise_path'], allow_pickle=False) as z:
            nb = z['initial_noise'].copy()
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.array_equal(na, nb), label + ': initial noise differs'
        maximum = float(np.max(np.abs(a-b)))
        assert np.allclose(a, b, rtol=1e-4, atol=1e-5), (label, maximum)
        return label, maximum
    for check in checks:
        count = check['actual_predictions']
        group = ordered[offset:offset+count]
        assert len(group) == count
        offset += count
        schedules = check['coverage']['custom_schedules']
        size = len(schedules)
        comparisons = []
        for index in range(size):
            cached, fresh = group[2*index:2*index+2]
            assert cached['audit']['schedule'] == schedules[index] == fresh['audit']['schedule']
            assert cached['audit']['actual_NFE'] == fresh['audit']['actual_NFE'] == schedules[index]['num_steps']
            comparisons.append(compare('cache_vs_independent', cached, fresh))
        before, reduced = group[2*size:2*size+2]
        for index in range(size):
            switched, minimal = group[2*size+2+2*index:2*size+4+2*index]
            comparisons.append(compare('roundtrip_custom_vs_initial', switched, group[2*index]))
            comparisons.append(compare('minimal_vs_instrumented', minimal, switched))
        after, native_ref, reduced_ref = group[4*size+2:4*size+5]
        comparisons.append(compare('native_roundtrip', before, after))
        for candidate, reference in [(before, native_ref), (reduced, reduced_ref)]:
            steps = reference['num_steps']
            assert candidate['audit']['actual_NFE'] == reference['actual_NFE'] == steps
            comparisons.append(compare('upstream_native_D'+str(steps), candidate, reference))
        if model != 'db-cogact':
            uniform = next(group[2*i] for i, schedule in enumerate(schedules) if schedule.get('m') == .5)
            comparisons.append(compare('u50_vs_native_D2', uniform, reduced))
        if check['skip'] and model != 'pi0':
            comparisons.append(compare('reference_vs_fast_hidden_bypass', group[-1], reduced))
        recorded = check['comparisons']
        assert len(comparisons) == len(recorded)
        for (label, maximum), row in zip(comparisons, recorded):
            assert row['comparison'] == label and row['passed'] is True
            assert row['same_noise'] is True and row['rtol'] == 1e-4 and row['atol'] == 1e-5
            assert np.isclose(maximum, row['max_abs'], rtol=0, atol=1e-12)
        results.append(dict(input_hash=check['input_hash'], skip=check['skip'],
                            actual_predictions=count, comparisons=len(comparisons),
                            max_abs=max(value for _, value in comparisons)))
    assert offset == len(ordered)
    return results


def independently_audit_formal_episode(attempt, entry, decode_video=True):
    """Recompute a committed episode's scientific and media invariants."""
    import numpy as np
    from common import sha
    from runtime_data import audited_protocol
    from runtime_client import check_prediction, legacy
    root = Path(attempt)
    manifest = json.loads((root/'condition_manifest.json').read_text())
    expected = {row['key']: row for row in planned(manifest)}[entry['key']]
    assert all(entry[key] == value for key, value in expected.items())
    directory = Path(entry['directory'])
    validated = directory/'validated.json'
    result = json.loads(validated.read_text())
    assert entry['status'] == 'valid' and sha(validated) == entry['validated_sha256']
    assert result['valid'] and result['kind'] == 'formal'
    assert not result.get('not_formal_success_estimate', False)
    assert result['episode_key'] == entry['key'] and result['success'] == entry['success']
    for name, expected_hash in result['artifacts'].items():
        assert sha(directory/name) == expected_hash, name
    identity = json.loads((directory/'identity.json').read_text())
    condition = next(row for row in manifest['conditions'] if row['id'] == entry['condition'])
    assert identity['skip'] == condition['skip'] and identity['kind'] == 'formal'
    init = next(row for row in manifest['official_initial_states']
                if row['suite'] == entry['suite'] and row['task'] == entry['task'])
    assert identity['init_vector_hash'] == init['indices'][str(entry['init'])]
    assert identity['init_file_sha256'] == init['sha256']
    protocol = audited_protocol(root, entry['model'])
    service_protocol = json.loads((directory/'native_protocol.json').read_text())
    native_r = protocol.get('R_by_suite', protocol['R'])
    native_r = native_r[entry['suite']] if isinstance(native_r, dict) else native_r
    assert result['H'] == protocol['H'] and result['R'] == native_r
    limit = manifest['native_action_limits']['limits'][entry['suite']]
    assert 0 < result['actions'] <= limit
    assert result['termination'] == ('success' if result['success'] else 'action_limit')
    if not result['success']:
        assert result['actions'] == limit
    states = [json.loads(x) for x in (directory/'telemetry.jsonl').read_text().splitlines()]
    inference = [json.loads(x) for x in (directory/'infer.jsonl').read_text().splitlines()]
    assert len(states) == result['actions'] + 1 == result['frames']
    assert len(inference) == result['predictions']
    assert [x['step'] for x in states] == list(range(len(states)))
    assert np.isfinite([x['eef_pos'] for x in states]).all()
    assert np.allclose(np.diff([x['sim_time_s'] for x in states]), .05, rtol=0, atol=1e-6)
    assert not any(x['success'] for x in states[1:-1])
    assert states[-1]['success'] == result['success']
    offset = 0
    for index, row in enumerate(inference):
        check_prediction(row, condition, protocol, seed(entry['suite'], entry['task'], entry['init'], index))
        expected_cache_key = legacy.value_digest(dict(model=entry['model'],
            protocol=legacy.value_digest(service_protocol), skip=tuple(condition['skip']),
            input=row['input_hash'], dtype='bfloat16'))
        assert row['audit']['cache_key'] == expected_cache_key
        assert row['step'] == offset and row['replan'] == index
        assert 0 < row['executed_length'] <= min(native_r, limit-offset)
        if index < len(inference)-1:
            assert row['executed_length'] == native_r
        assert sha(row['input_path']) == row['input_file_sha256']
        assert sha(row['prediction_path']) == row['prediction_sha256']
        with np.load(row['input_path'], allow_pickle=False) as z:
            obs = {key: z[key] for key in z.files if key not in ('instruction', 'seed')}
            assert legacy.input_digest(obs, str(z['instruction'].item())) == row['input_hash']
        executed = np.asarray(row['chunk_actions'])[:row['executed_length']].copy()
        if entry['model'] == 'db-cogact':
            executed[:, -1] = np.where(executed[:, -1] < 0, -1, 1)
        actual = np.asarray([x['action'] for x in states[offset+1:offset+len(executed)+1]])
        assert np.array_equal(executed, actual)
        offset += len(executed)
    assert offset == result['actions']
    environment = json.loads((directory/'environment_protocol.json').read_text())
    assert environment['control_frequency_hz'] == 20
    assert environment['loaded_client_sha256'] == sha(ROOT/'preparations/idea1_v2_1/runtime_client.py')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(directory/'video.mp4')]))
    assert int(probe['streams'][0]['nb_read_frames']) == len(states)
    if decode_video:
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(directory/'video.mp4'), '-f', 'null', '-'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return dict(key=entry['key'], condition=entry['condition'], suite=entry['suite'], task=entry['task'],
                init=entry['init'], success=result['success'], actions=result['actions'], frames=len(states),
                predictions=len(inference), H=result['H'], R=native_r, video_fully_decoded=decode_video,
                validated_sha256=sha(validated), video_sha256=sha(directory/'video.mp4'))


class IndependentAcceptance(unittest.TestCase):
    def test_medium_acceptance_binds_manifest_revision_and_model(self):
        from runtime_data import verify_acceptance
        with tempfile.TemporaryDirectory(prefix='v22-independent-acceptance-') as directory:
            root = Path(directory)
            manifest = dict(manifest_hash='manifest-a', code_hashes={'runtime.py': 'revision-a'})
            acceptance = dict(schema=1, reviewer_model='gpt-6.1-sol', reasoning_effort='medium',
                              manifest_hash='manifest-a', runtime_code_hashes=manifest['code_hashes'],
                              models={'pi0': dict(accepted=True, parity_identity='parity-a', evidence=['real-gate.json'])})
            atomic_json(root/'condition_manifest.json', manifest)
            atomic_json(root/'runtime_acceptance.json', acceptance)
            self.assertTrue(verify_acceptance(root, 'pi0', {'identity': 'parity-a'})['accepted'])
            for field, value in [('reasoning_effort', 'high'), ('manifest_hash', 'manifest-b'),
                                 ('runtime_code_hashes', {'runtime.py': 'revision-b'})]:
                atomic_json(root/'runtime_acceptance.json', dict(acceptance, **{field: value}))
                with self.assertRaises(ValueError):
                    verify_acceptance(root, 'pi0', {'identity': 'parity-a'})
            atomic_json(root/'runtime_acceptance.json', acceptance)
            for model, parity in [('pi0', 'stale-parity'), ('db-cogact', 'parity-a')]:
                with self.assertRaises(ValueError):
                    verify_acceptance(root, model, {'identity': parity})

    def test_inherited_attempt_counts_reserved_calls_and_deduplicates_sources(self):
        from runtime_data import inherited_diagnostics
        with tempfile.TemporaryDirectory(prefix='v22-independent-prior-') as directory:
            root = Path(directory)
            manifest = dict(prior_diagnostics={})
            manifest['manifest_hash'] = digest(manifest)
            atomic_json(root/'condition_manifest.json', manifest)
            atomic_json(root/'parity/pi0.json', dict(passed=False, reserved_calls=40))
            for name, count in [('prediction_ledger.jsonl', 35), ('native_reference_ledger.jsonl', 2)]:
                path = root/'native/pi0'/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}\n' * count)
            atomic_json(root/'readiness/pi0/diagnostic_budget.json', dict(smoke_reserved=4, smoke_actual=3))
            base = inherited_diagnostics({})['pi0']
            config = dict(prior_attempts=[str(root), str(root)])
            result = inherited_diagnostics(config)['pi0']
            self.assertEqual(result['actual_predictions'] - base['actual_predictions'], 40)
            self.assertEqual(result['conservative_predictions'] - base['conservative_predictions'], 44)
            self.assertEqual(len(result['sources']), len({x['path'] for x in result['sources']}))
            with self.assertRaises(ValueError):
                inherited_diagnostics(dict(prior_attempts=[dict(path=str(root), manifest_sha256='wrong')]))

    def test_pi0_native_d10_uses_frozen_checkpoint_fusions_and_independent_snapshots(self):
        import torch
        from pi0_adapter import Pi0Adapter
        # Deliberately omit source MLP tensors: native D10 must never re-fuse
        # BF16 weights or regenerate its released checkpoint time biases.
        weights = {
            'decoder_action_fused_time_biases': torch.arange(36, dtype=torch.float32).reshape(12, 3).to(torch.bfloat16),
            'decoder_action_fused_out_proj_w': torch.tensor([[.123, -.789]], dtype=torch.bfloat16),
            'decoder_action_fused_out_proj_b': torch.tensor([.271], dtype=torch.bfloat16),
        }
        adapter = Pi0Adapter.__new__(Pi0Adapter)
        adapter.torch = torch
        adapter.schedules = {}
        runtime = SimpleNamespace(weights=weights)
        nodes = flow_schedule('pi0', None, 10)
        biases, output_weights, output_biases, times = adapter.schedule(runtime, nodes)
        self.assertTrue(torch.equal(biases, weights['decoder_action_fused_time_biases'][:10]))
        self.assertEqual(times.tolist(), torch.tensor(nodes['timesteps'], dtype=torch.float32).tolist())
        for item in output_weights:
            self.assertTrue(torch.equal(item, weights['decoder_action_fused_out_proj_w']))
            self.assertNotEqual(item.data_ptr(), weights['decoder_action_fused_out_proj_w'].data_ptr())
        for item in output_biases:
            self.assertTrue(torch.equal(item, weights['decoder_action_fused_out_proj_b']))
        self.assertEqual(len({item.data_ptr() for item in output_weights}), 10)
        before = biases.clone()
        weights['decoder_action_fused_time_biases'].zero_()
        self.assertTrue(torch.equal(biases, before))

    def test_matrix_scientific_identity_and_db_control_separation(self):
        for extension, expected in ((False, 38), (True, 54)):
            matrix = conditions(extension)
            manifest = dict(conditions=matrix, protocol_hashes={m: digest(m) for m in ('pi0', 'xiaomi-robotic0', 'db-cogact')})
            keys = list(planned(manifest))
            self.assertEqual(len(keys), expected * 200)
            self.assertEqual(len({x['key'] for x in keys}), expected * 200)
            for model, count in [('pi0', 16), ('xiaomi-robotic0', 18), ('db-cogact', 20 if extension else 4)]:
                self.assertEqual(sum(x['model'] == model for x in matrix), count)
            for mask in ('full', 'skip9_adjacent_pairs'):
                control = native_control(matrix, 'db-cogact', mask)
                self.assertEqual(control['schedule'], dict(kind='native', m=None, num_steps=10))
            if extension:
                for mask in ('full', 'skip9_adjacent_pairs'):
                    selected = [x for x in matrix if x['model'] == 'db-cogact' and x['mask'] == mask]
                    native = next(x for x in selected if x['schedule'] == dict(kind='native', m=None, num_steps=2))
                    uniform = next(x for x in selected if x['schedule'] == dict(kind='ddim_custom', m=.5, num_steps=2))
                    self.assertNotEqual(native['id'], uniform['id'])

    def test_euler_against_independent_closed_form_and_constant_field(self):
        for model, direction in [('pi0', -1), ('xiaomi-robotic0', 1)]:
            for middle in (.5, .3, .7, .1, .25, .9, .75, .6):
                s = flow_schedule(model, middle)
                self.assertEqual(s['nodes'][0], 1 if direction == -1 else 0)
                self.assertEqual(s['nodes'][-1], 0 if direction == -1 else 1)
                x, constant = 2., 2.
                calls = []
                for t, dt in zip(s['timesteps'], s['dt']):
                    calls.append((x, t))
                    x += dt * (3 * x + 2 * t)
                    constant += dt * 7
                a = direction * (1 - middle)
                b = direction * middle
                t0 = 1 if direction == -1 else 0
                t1 = middle if direction == -1 else 1 - middle
                expected = (1 + 3*b) * ((1 + 3*a)*2 + 2*a*t0) + 2*b*t1
                self.assertAlmostEqual(x, expected)
                self.assertAlmostEqual(constant, 2 + 7*direction)
                self.assertEqual(len(calls), 2)
                self.assertNotEqual(calls[1][0], calls[0][0])

    def test_nonfinite_and_incomplete_nodes_rejected(self):
        for value in (float('nan'), float('inf'), -.1, 0, 1, 1.1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_schedule('pi0', dict(kind='flow', m=value, num_steps=2))

    def test_real_upstream_ddim_updates_match_independent_alpha_equation(self):
        import numpy as np
        import torch
        path = ROOT/'policies/db-cogact/dexbotic/model/cogact/action_model/diffusion.py'
        spec = importlib.util.spec_from_file_location('independent_native_diffusion', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kwargs = dict(noise_schedule='squaredcos_cap_v2', diffusion_steps=100, sigma_small=True, learn_sigma=False)
        base = module.create_diffusion('', **kwargs)
        native = module.create_diffusion('ddim10', **kwargs)
        self.assertEqual(max(native.timestep_map), 90)
        for middle in (.5, .3, .7, .1, .25, .9, .75, .6):
            high, mid = ddim_indices(90, middle)
            sampler = module.SpacedDiffusion(use_timesteps={high, mid}, betas=base.betas, model_mean_type=native.model_mean_type, model_var_type=native.model_var_type, loss_type=native.loss_type)
            actual = []
            def denoiser(x, t, **unused):
                actual.append((int(t[0]), x.detach().clone()))
                return .1*x + t.reshape(-1, 1, 1)*.001
            initial = torch.full((1, 2, 7), .7)
            output = sampler.ddim_sample_loop(denoiser, initial.shape, noise=initial, clip_denoised=False, progress=False, device='cpu', eta=0.)
            self.assertEqual([t for t, _ in actual], [high, mid])
            self.assertFalse(torch.equal(actual[0][1], actual[1][1]))
            expected = initial.numpy().astype(np.float64)
            for current, following in ((high, mid), (mid, None)):
                alpha = base.alphas_cumprod[current]
                next_alpha = 1. if following is None else base.alphas_cumprod[following]
                epsilon = .1*expected + current*.001
                predicted_clean = (expected-np.sqrt(1-alpha)*epsilon)/np.sqrt(alpha)
                expected = np.sqrt(next_alpha)*predicted_clean + np.sqrt(1-next_alpha)*epsilon
            np.testing.assert_allclose(output.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_e1_interaction_uses_uniform_control_instead_of_native_high_d(self):
        matrix = conditions(True)
        manifest = dict(conditions=matrix, protocol_hashes={m: digest(m) for m in ('pi0', 'xiaomi-robotic0', 'db-cogact')}, observations={m: [] for m in ('pi0', 'xiaomi-robotic0', 'db-cogact')})
        with tempfile.TemporaryDirectory(prefix='v22-independent-analysis-') as directory:
            root = Path(directory)
            atomic_json(root/'condition_manifest.json', manifest)
            atomic_json(root/'episode_ledger.json', dict(entries=[]))
            analyze(root)
            with (root/'paired_comparisons.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for model, mask in [('xiaomi-robotic0', 'skip12_short_runs'), ('db-cogact', 'skip9_adjacent_pairs')]:
                variant = f'{model}_{mask}_m70'
                expected = {(variant, 1), (f'{model}_{mask}_u50', -1), (f'{model}_full_m70', -1), (f'{model}_full_u50', 1)}
                comparisons = [{tuple(term) for term in json.loads(row['terms'])} for row in rows if 'interaction' in row['comparison'] and variant in row['comparison']]
                self.assertIn(expected, comparisons, f'{model}: E1 must hold D=2 and compare against u50 in each mask')

    def test_seed_has_independent_canonical_definition_across_processes(self):
        expected = int(hashlib.sha256(json.dumps(['idea1-v22', 1701, 'libero_goal', 4, 9, 3], sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:8], 16)
        self.assertEqual(seed('libero_goal', 4, 9, 3), expected)
        script = "import sys;sys.path.insert(0,sys.argv[1]);from common import seed;print(seed('libero_goal',4,9,3))"
        for hashseed in ('17', '938'):
            result = subprocess.check_output([sys.executable, '-c', script, str(HERE)], env=dict(os.environ, PYTHONHASHSEED=hashseed), text=True)
            self.assertEqual(int(result.strip()), expected)

    def test_concurrent_processes_cannot_replace_valid_failure(self):
        context = multiprocessing.get_context('fork')
        with tempfile.TemporaryDirectory(prefix='v22-independent-ledger-') as directory:
            matrix = dict(conditions=conditions(), protocol_hashes={m: digest(m) for m in ('pi0', 'xiaomi-robotic0', 'db-cogact')})
            row = next(planned(matrix))
            queue = context.Queue()
            processes = [context.Process(target=reserve_valid_failure, args=(directory, row, queue)) for _ in range(8)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(sum(queue.get(timeout=2) for _ in processes), 1)
            entries = Ledger(directory).read()['entries']
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]['status'], 'valid')
            self.assertIs(entries[0]['success'], False)
            self.assertIsNone(Ledger(directory).reserve(row))

    def test_gpu_lease_is_mutual_exclusion_across_independent_processes(self):
        # No GPU is queried or used: only the exact production lock mechanism
        # is exercised under a temporary workspace root.
        script = """import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import common
common.ROOT = Path(sys.argv[2])
try:
    with common.gpu_lease(999, allow_inherited=False):
        print('acquired')
except RuntimeError:
    print('blocked')
"""
        with tempfile.TemporaryDirectory(prefix='v22-independent-gpu-lock-') as directory:
            env = dict(os.environ)
            env.pop('V22_GPU_LEASE_FD', None)
            with mock.patch('common.ROOT', Path(directory)), gpu_lease(999, allow_inherited=False):
                output = subprocess.check_output([sys.executable, '-c', script, str(HERE), directory], env=env, text=True)
                self.assertEqual(output.strip(), 'blocked')
            output = subprocess.check_output([sys.executable, '-c', script, str(HERE), directory], env=env, text=True)
            self.assertEqual(output.strip(), 'acquired')


if __name__ == '__main__':
    unittest.main()
