"""Read-only source validation. No imports of model or environment packages."""
from collections import defaultdict
import json
import numpy as np
from .core import SUITES, WEIGHT, rows, sha

def load(source):
    groups = defaultdict(list)
    files = {}
    keys = set()
    for suite in SUITES:
        path = source/'shadow_supplement_v14/round2_20261011/V14_A03'/suite/'predictions.jsonl'
        files[str(path)] = sha(path)
        for record in rows(path):
            key = (suite, int(record['task_id']), int(record['episode_idx']), int(record['replan_id']))
            if key in keys:
                raise ValueError('Duplicate prediction key %r' % (key,))
            keys.add(key)
            if record['checkpoint_sha256'] != WEIGHT or record['skip_layers'] != [3,4,9,11]:
                raise ValueError('Unexpected prediction protocol')
            for field in ('full_actions','variant_actions'):
                array = np.asarray(record[field],dtype=np.float64)
                if array.shape != (50,7) or not np.isfinite(array).all():
                    raise ValueError('Invalid %s at %r' % (field,key))
            groups[key[:3]].append(record)
    if len(groups) != 200 or len(keys) != 6979:
        raise ValueError('Expected 200 episodes/6979 points, found %d/%d' % (len(groups),len(keys)))
    counts = {'calibration':0,'evaluation':0}
    for key, predictions in groups.items():
        predictions.sort(key=lambda r:r['environment_action_step'])
        infer_path = source/(__import__('pathlib').Path(predictions[0]['full_infer_path']).relative_to(source))
        telemetry_path = infer_path.with_name('telemetry.jsonl')
        infer = rows(infer_path)
        telemetry = rows(telemetry_path)
        files[str(infer_path)],files[str(telemetry_path)] = sha(infer_path),sha(telemetry_path)
        if [r['step'] for r in telemetry] != list(range(len(telemetry))):
            raise ValueError('Incomplete state timeline')
        if not np.allclose(np.diff([r['sim_time_s'] for r in telemetry]),.05,rtol=0,atol=1e-8):
            raise ValueError('Control period is not .05s')
        actions = np.asarray([r['action'] for r in telemetry[1:]],dtype=np.float64)
        by_id = {r['infer_id']:r for r in infer}
        if len(by_id) != len(infer) or len(infer) != len(predictions):
            raise ValueError('Inference coverage mismatch')
        for record in predictions:
            t = int(record['environment_action_step'])
            meta = by_id[record['replan_id']]
            if t != meta['frame_idx'] or meta['experiment11']['noise_sha256'] != record['noise_sha256']:
                raise ValueError('Prediction/time/noise mismatch')
            n = min(5,len(actions)-t)
            if n <= 0 or not np.allclose(actions[t:t+n],np.asarray(record['full_actions'])[:n],rtol=1e-4,atol=1e-5):
                raise ValueError('Executed prefix differs from source chunk')
            record['_actions'] = actions
            record['_telemetry'] = telemetry
            record['_prefix'] = n
            counts['calibration' if key[1]==0 else 'evaluation'] += 1
            path=__import__('pathlib').Path(record['full_input_path'])
            if not path.is_file() or source not in path.parents:raise ValueError('Missing/unexpected source input path')
            files[str(path)]=sha(path)
            with np.load(path,allow_pickle=False) as cached:
                from .core import noise
                actual=__import__('hashlib').sha256(cached['noise'].tobytes()).hexdigest()
                if actual!=record['noise_sha256'] or not np.array_equal(cached['noise'],noise(key[2],record['replan_id'])):
                    raise ValueError('Cached noise does not match frozen noise rule')
                input_meta=json.loads(str(cached['metadata_json'].item()))
                digest=__import__('hashlib').sha256()
                for name in ('agent_image','wrist_image','robot_state','noise'):
                    digest.update(np.ascontiguousarray(cached[name]).tobytes())
                digest.update(str(cached['prompt'].item()).encode('utf-8'))
                digest.update(key[0].encode('utf-8'))
                if digest.hexdigest()!=record['full_input_sha256'] or input_meta.get('input_sha256')!=record['full_input_sha256']:
                    raise ValueError('Input identity mismatch')
    if counts != {'calibration':634,'evaluation':6345}:
        raise ValueError('Frozen split mismatch %r' % counts)
    # Source manifests/episode provenance are independently hashed.
    baseline_counts = {}
    for condition in ('Full18_D3_R5_FORMAL','V14_A03_D3_R5_FORMAL'):
        total = 0
        successes = 0
        for suite in SUITES:
            root = source/'closed_loop'/condition/suite/'run'
            for name in ('manifest.json','episode_log.json','validated.json'):
                files[str(root/name)] = sha(root/name)
            episodes = json.loads((root/'episode_log.json').read_text())
            expected = {(t,i) for t in range(10) for i in range(5)}
            actual = {(r['task_id'],r['episode_idx']) for r in episodes}
            if actual != expected or len(episodes)!=50 or any(r.get('failure_reason') for r in episodes):
                raise ValueError('Invalid baseline coverage %s/%s' % (condition,suite))
            total += len(episodes)
            successes += sum(bool(r['success']) for r in episodes)
            for row in episodes:
                for name in ('infer_path','trace_path','telemetry_path'):
                    files[str(root/row[name])]=sha(root/row[name])
        baseline_counts[condition] = {'episodes':total,'successes':successes}
    for name in ('closed_loop_config_formal.json','candidate_manifest.json'):
        files[str(source/'protocol'/name)]=sha(source/'protocol'/name)
    return groups, {'files':files,'points':len(keys),'episodes':len(groups),'split':counts,'baselines':baseline_counts}
