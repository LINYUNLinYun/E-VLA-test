"""Real sequential-observation prefix KV comparison and AE injection.

No GPU imports occur until selection has been persisted. Cutoff L10 means
L1..L9 current V14 and L10..L18 previous Full. Prefix KV only is exchanged;
current state, prompt, noise and every AE block remain current and unchanged.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V14 = (3, 4, 9, 11)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def adjacent_pairs(rows, episode):
    """Never bridge a missing request, episode boundary, future or non-R5 gap."""
    ordered = sorted(rows, key=lambda r: int(r['request']))
    if len({int(r['request']) for r in ordered}) != len(ordered):
        raise ValueError('Duplicate requests: ' + episode)
    pairs = []
    for old, current in zip(ordered, ordered[1:]):
        if int(current['request']) != int(old['request']) + 1:
            continue
        if int(current['step']) - int(old['step']) != 5:
            continue
        if old.get('execute_length', 5) != 5:
            continue
        pairs.append(dict(episode=episode, previous=old, current=current))
    return pairs


def select_pairs(source, per_episode=4, maxpairs=None):
    source = Path(source).resolve()
    selected = []
    for infer in sorted(source.glob('*/task*/init*/infer.jsonl')):
        result = json.loads(infer.with_name('result.json').read_text())
        if not result.get('valid') or result.get('mode') != 'fresh_serial' or result.get('interval') != 1:
            raise ValueError('Requires valid fresh_serial interval1 episode: ' + str(infer))
        if any(result.get(key) != value for key,value in dict(H=50,R=5,control_hz=20).items()):
            raise ValueError('Requires recorded H50/R5/control20: ' + str(infer))
        rows = [json.loads(line) for line in infer.read_text().splitlines() if line.strip()]
        pairs = adjacent_pairs(rows, str(infer.parent.relative_to(source)))
        if not pairs:
            raise ValueError('No consecutive R5 pair: ' + str(infer))
        # Uses observation request indices only, never outcome/action errors.
        indices = np.unique(np.rint(np.linspace(0, len(pairs)-1, min(per_episode, len(pairs)))).astype(int))
        for index in indices:
            pair = pairs[index]
            item = dict(episode=pair['episode'], infer_path=str(infer), infer_sha256=sha(infer))
            for side in ('previous', 'current'):
                row = pair[side]
                path = Path(row['input_path'])
                path = path if path.is_absolute() else ROOT / path
                if sha(path) != row['input_file_sha256']:
                    raise ValueError('Input digest mismatch: ' + str(path))
                item[side] = {k: row[k] for k in ('request', 'step', 'seed', 'input_hash')}
                item[side].update(path=str(path), sha256=sha(path))
            selected.append(item)
    if maxpairs is not None:
        selected = selected[:maxpairs]
    if not selected:
        raise ValueError('No pairs selected')
    return selected


def load_observation(side):
    if sha(side['path']) != side['sha256']:
        raise ValueError('Frozen input changed')
    with np.load(side['path'], allow_pickle=False) as saved:
        instruction = str(saved['instruction'].item())
        seed = int(saved['seed'].item())
        obs = {k: saved[k].copy() for k in saved.files if k not in ('instruction', 'seed')}
    if seed != side['seed']:
        raise ValueError('Seed differs from inference record')
    from preparations.idea1_v2_1.runtime_client import input_digest
    if input_digest(obs, instruction) != side['input_hash']:
        raise ValueError('Observation identity differs from inference record')
    return obs, instruction


def mixed_context(current, previous, cutoff):
    """Clone snapshots; never alias or mutate original contexts (NumPy or Torch)."""
    if not 1 <= cutoff <= 19:
        raise ValueError('cutoff is 1-based in [1,19]')
    if current['seq'] != previous['seq']:
        raise ValueError('Different prefix lengths cannot be mixed')
    result = dict(current)
    for key in ('K', 'V'):
        a, b = current[key], previous[key]
        if a.shape != b.shape or a.shape[0] != 18:
            raise ValueError('Expected matching 18-layer snapshots')
        clone = a.clone() if hasattr(a, 'clone') else a.copy()
        clone[cutoff-1:] = b[cutoff-1:]
        result[key] = clone
    return result


def kv_metrics(reference, candidate):
    a = np.asarray(reference, np.float64).ravel()
    b = np.asarray(candidate, np.float64).ravel()
    if a.size != b.size or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid KV arrays')
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if an == 0 or bn == 0:
        raise ValueError('Undefined zero-norm KV metric')
    return dict(cosine=float(a @ b / (an * bn)), relative_l2=float(np.linalg.norm(b-a)/an), rms_ratio=float(bn/an))


def action_metrics(reference, candidate):
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != (50,7) or b.shape != a.shape or not np.isfinite([a,b]).all():
        raise ValueError('Expected finite decoded H50 action7')
    rows = []
    for horizon in (5,50):
        delta = b[:horizon] - a[:horizon]
        for name, sl in (('translation', slice(0,3)), ('rotation', slice(3,6)), ('gripper', slice(6,7))):
            error = delta[:, sl]
            row = dict(horizon=horizon, component=name, rmse=float(np.sqrt(np.mean(error**2))),
                       mean_l2=float(np.linalg.norm(error, axis=-1).mean()), max_abs=float(np.abs(error).max()), sign_mismatch_fraction='', clipped_controller_translation_mm_proxy='')
            if name == 'translation':
                clipped = np.clip(b[:horizon,:3],-1,1) - np.clip(a[:horizon,:3],-1,1)
                row['clipped_controller_translation_mm_proxy'] = float(np.linalg.norm(clipped,axis=-1).mean()*50)
            if name == 'gripper':
                row['sign_mismatch_fraction'] = float(np.mean((b[:horizon,6] >= 0) != (a[:horizon,6] >= 0)))
            rows.append(row)
    return rows


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args, pairs, manifest):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from preparations.idea1_v3_2.residual import require_idle_gpu
    manifest['assigned_gpu_preload_receipt'] = require_idle_gpu()
    from preparations.idea1_v3.policy import TritonBackend
    backend = TritonBackend()
    torch = backend.torch
    manifest.update(gpu=args.gpu, device=torch.cuda.get_device_name(0), torch=torch.__version__, backend_protocol=backend.protocol)
    metrics, action_rows, sanity = [], [], []
    for pair_id, pair in enumerate(pairs):
        old_obs, old_prompt = load_observation(pair['previous'])
        obs, prompt = load_observation(pair['current'])
        if old_prompt != prompt:
            raise ValueError('Prompt changed within pair')
        seed = pair['current']['seed']
        with torch.inference_mode():
            old, _ = backend.prepare(old_obs, old_prompt, ())
            full, _ = backend.prepare(obs, prompt, ())
            draft, _ = backend.prepare(obs, prompt, V14)
            if old['seq'] != full['seq'] or full['seq'] != draft['seq']:
                raise ValueError('Prefix sequence lengths differ')
            equal = all(torch.equal(full[key][:2], draft[key][:2]) for key in ('K','V'))
            sanity.append(dict(pair_id=pair_id, first_two_layers_bitwise_equal=equal))
            if not equal:
                raise ValueError('V14 L1/L2 differs from same-input Full')
            for key in ('K','V'):
                reference = full[key].float().cpu().numpy()
                for name, context in (('current_v14', draft), ('previous_full', old)):
                    candidate = context[key].float().cpu().numpy()
                    for layer in range(18):
                        for region, sl in (('all', slice(None)), ('vision', slice(0,512)), ('language', slice(512,None))):
                            if reference[layer, sl].size:
                                metrics.append(dict(pair_id=pair_id, episode=pair['episode'], current_request=pair['current']['request'],
                                    candidate=name, kv=key, layer=layer+1, region=region, **kv_metrics(reference[layer,sl], candidate[layer,sl])))
            baseline, baseline_raw, _ = backend.sample(full, seed, args.denoising)
            if pair_id == 0:
                repeat, repeat_raw, _ = backend.sample(full, seed, args.denoising)
                repeat_equal = bool(np.array_equal(baseline, repeat) and torch.equal(baseline_raw,repeat_raw))
                manifest['same_seed_full_repeat'] = dict(bitwise_equal=repeat_equal, decoded_max_abs=float(np.abs(baseline-repeat).max()),
                    raw_max_abs=float((baseline_raw.float()-repeat_raw.float()).abs().max().cpu()))
                if not repeat_equal:
                    raise ValueError('Same-seed Full repeat differs')
            contexts = [('full_current', full), ('v14_current', draft)]
            contexts += [(f'v14_then_old_from_L{cutoff}', mixed_context(draft, old, cutoff)) for cutoff in args.cutoffs]
            contexts += [('all_old', mixed_context(draft, old, 1))]
            action_arrays = dict(full_current=baseline)
            for name, context in contexts:
                if name == 'full_current':
                    actions, raw = baseline, baseline_raw
                else:
                    actions, raw, _ = backend.sample(context, seed, args.denoising)
                raw_np = raw.float().cpu().numpy()
                if raw_np.shape != (50,32) or not np.isfinite(raw_np).all():
                    raise ValueError('Invalid raw action horizon/dim or nonfinite')
                action_arrays[name] = actions
                action_arrays[name+'_raw'] = raw_np
                for row in action_metrics(baseline, actions):
                    action_rows.append(dict(pair_id=pair_id, episode=pair['episode'], current_request=pair['current']['request'],
                        candidate=name, seed=seed, noise_sha256=hashlib.sha256(np.random.default_rng(seed).standard_normal((50,32), dtype=np.float32).tobytes()).hexdigest(), **row))
            np.savez_compressed(Path(args.output)/f'actions_pair{pair_id:03d}.npz', **action_arrays)
        write_csv(Path(args.output)/'layer_metrics.csv', metrics)
        write_csv(Path(args.output)/'action_metrics.csv', action_rows)
        manifest.update(completed_pairs=pair_id+1, sanity=sanity)
        dump(Path(args.output)/'manifest.json', manifest)
        print(json.dumps(dict(completed_pairs=pair_id+1, total_pairs=len(pairs), episode=pair['episode'])), flush=True)
    manifest.update(status='complete')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', type=int)
    p.add_argument('--per-episode', type=int, default=4)
    p.add_argument('--maxpairs', type=int)
    p.add_argument('--cutoffs', type=int, nargs='+', default=[10,13,16])
    p.add_argument('--denoising', type=int, choices=(2,10), default=2)
    p.add_argument('--select-only', action='store_true')
    args = p.parse_args()
    if args.per_episode < 1 or (args.maxpairs is not None and args.maxpairs < 1):
        p.error('positive sample counts required')
    if any(c < 2 or c > 18 for c in args.cutoffs):
        p.error('hybrid cutoffs must be in [2,18]')
    if not args.select_only and args.gpu is None:
        p.error('--gpu required for GPU execution')
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    pairs = select_pairs(args.source, args.per_episode, args.maxpairs)
    selection_path = dest/'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != pairs:
        raise ValueError('Existing frozen selection differs; use a fresh output')
    dump(selection_path, pairs)
    if args.select_only:
        print(json.dumps(dict(selected_pairs=len(pairs), selection=str(selection_path))))
        return
    if (dest/'manifest.json').exists():
        raise ValueError('Output already has GPU run manifest; use a fresh output')
    manifest = dict(status='running', started_unix=time.time(), argv=sys.argv, python=sys.executable, selection_sha256=sha(selection_path),
        source_sha256={str(path):sha(path) for path in (Path(__file__), ROOT/'preparations/idea1_v3/policy.py',
            ROOT/'preparations/idea1_v2_2/pi0_adapter.py', ROOT/'policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py')},
        num_pairs=len(pairs), denoising=args.denoising, skip_1based=list(V14), cutoffs=args.cutoffs, execute=5, horizon=50,
        region_contract='vision tokens [0,512), language [512,T); all unpadded native Triton prefix tokens',
        cutoff_contract='L cutoff is inclusive first previous-Full layer; earlier layers current V14. This exchanges each AE layer prefix KV directly, without recomputing VLM hidden states.',
        full_final_layer='Triton L18 materializes KV only: final prefix hidden output is unused by AE. This does not establish that L18 KV is unimportant.',
        action_contract='Decoded native LIBERO action coordinates: translation/rotation coordinate errors, not Cartesian pose or SO(3) geodesic error; gripper raw and sign error. Clipped translation meanL2*50mm is only a controller delta proxy. All paths use current state and same current noise.',
        scope='Shadow mechanism probe on recorded real Full trajectory; no rollout, no speed measurement or performance claim.')
    dump(dest/'manifest.json', manifest)
    try:
        run(args, pairs, manifest)
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest.update(finished_unix=time.time())
        dump(dest/'manifest.json', manifest)


if __name__ == '__main__':
    main()
