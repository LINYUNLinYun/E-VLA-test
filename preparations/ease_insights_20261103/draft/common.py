"""Frozen identifiers and filesystem-only preparation for draft generalization."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / 'policies/realtime-vla-flash'
CHECKPOINTS = ROOT / 'shared/checkpoints/realtime-vla-flash/converted'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
ARTIFACTS = dict(zip(SUITES, ('draft_spatial', 'draft_object', 'draft', 'draft_10')))
PROTOCOL = dict(H=50, R=12, settling=10, environment_seed=7, noise_seed=7,
                denoise_steps=10, tau_radius=0.15, periodic_full_every_n_draft_rounds=2,
                config='pi0_libero', resize_size=224, environment_resolution=256,
                max_steps=dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520),
                noise_rule='SHA256(ease-insights-v1|7|suite|task|init|replan), first 8 little-endian bytes; numpy default_rng standard_normal float32 cast',
                normalization='fixed converted/base/assets; shared pi0_libero action/state transformations')

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)

def case_key(case):
    return '%s_t%02d_i%02d' % (case['suite'], case['task_id'], case['init_id'])

def noise_seed(suite, task_id, init_id, replan_id):
    value = 'ease-insights-v1|7|%s|%d|%d|%d' % (suite, task_id, init_id, replan_id)
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')

def normalize_cases(path=None, target_suite='all', init_ids=(20,)):
    targets = SUITES if target_suite == 'all' else tuple(target_suite.split(','))
    if not set(targets).issubset(SUITES):
        raise ValueError('Unknown target suite')
    if path:
        data = json.loads(Path(path).read_text())
        rows = data.get('cases', []) if isinstance(data, dict) else data
        if not isinstance(rows, list) or not rows:
            raise ValueError('Cases file must be nonempty list or {cases: [...]}')
    else:
        rows = [dict(suite=s, task_id=t, init_id=i) for s in targets for t in range(10) for i in init_ids]
    cases = []
    for row in rows:
        suite = row.get('suite', row.get('target_suite', row.get('task_suite_name')))
        if suite not in targets:
            continue
        task = int(row.get('task_id', row.get('task', -1)))
        init = int(row.get('init_id', row.get('episode_idx', row.get('init', -1))))
        if not 0 <= task < 10 or not 0 <= init < 50:
            raise ValueError('Invalid official task/init index: %r' % row)
        cases.append(dict(suite=suite, task_id=task, init_id=init))
    if not cases or len({case_key(c) for c in cases}) != len(cases):
        raise ValueError('Empty or duplicate case matrix')
    return sorted(cases, key=lambda c: (SUITES.index(c['suite']), c['task_id'], c['init_id']))

def stage_artifacts(output, source_suite):
    """Private mutable prompt manifest; only checkpoint/embedding weights are links."""
    destination = Path(output) / 'artifacts'
    destination.mkdir(parents=True, exist_ok=True)
    embed_dir = destination / 'language_embeds'
    embed_dir.mkdir(exist_ok=True)
    weight = CHECKPOINTS / ARTIFACTS[source_suite] / 'draft_triton.pkl'
    link = destination / 'draft_triton.pkl'
    if link.exists():
        if link.resolve() != weight.resolve():
            raise ValueError('Existing staged draft belongs to a different source')
    else:
        link.symlink_to(weight)
    prompts = {}
    for artifact in ARTIFACTS.values():
        source = CHECKPOINTS / artifact
        for prompt, entry in json.loads((source / 'manifest.json').read_text())['prompts'].items():
            original = source / entry['embed_path']
            local = embed_dir / original.name
            if not local.exists():
                local.symlink_to(original)
            item = dict(entry, embed_path=str(local.relative_to(destination)))
            if prompt in prompts and prompts[prompt]['prompt_len'] != item['prompt_len']:
                raise ValueError('Inconsistent prompt token lengths')
            prompts[prompt] = item
    atomic_json(destination / 'manifest.json', dict(prompts=prompts))
    return destination
