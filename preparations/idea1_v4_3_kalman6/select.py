import json
import time
from collections import defaultdict
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v4_0.finish import load_episode
from .common import OUT, read, grid, config, directory, key, anchors, unique


def replay_selection():
    grouped = defaultdict(list)
    for bank in read()['replay_bank']:
        d = OUT/'replay'/key(bank['case'])
        assert (d/'complete.json').exists()
        for row in json.loads((d/'metrics.json').read_text()):
            grouped[row['id']].append(row)
    summary = []
    for name, rows in grouped.items():
        result = dict(id=name,samples=len(rows))
        for field in ('mse','spatial_rmse','xyz_rmse','rotation_rmse','gripper_mismatch','ae_ms'):
            result[field] = float(np.mean([r[field] for r in rows]))
        accepts = [r for r in rows if 'accepted_length' in r]
        if accepts:
            result['rejection_rate'] = float(np.mean([r['accepted_length']==0 for r in accepts]))
            result['accepted_length'] = float(np.mean([r['accepted_length'] for r in accepts]))
        result['p90_mse'] = float(np.quantile([r['mse'] for r in rows],.9))
        summary.append(result)
    by_id = {r['id']:r for r in summary}
    selected = []
    for nfe in (1,2):
        ranked = sorted([c for c in grid() if c['fast_steps']==nfe],
            key=lambda c:(by_id[c['id']]['mse'],by_id[c['id']]['p90_mse'],c['id']))
        selected.extend(ranked[:2])
    result = dict(candidates=unique(anchors()+selected),screen_top_two_per_nfe=selected,summary=summary,
        rule=read()['selection']['replay'],manifest_hash=read()['manifest_hash'],time=time.time())
    atomic(OUT/'replay_selection.json',result)
    return result


def champions():
    manifest = read()
    candidates = json.loads((OUT/'replay_selection.json').read_text())['candidates']
    ranked = []
    for candidate in candidates:
        cfg = config(candidate)
        episodes = [load_episode(cfg,c,directory(cfg['id'],c))[3] for c in manifest['tune_cases']]
        ranked.append(dict(candidate=candidate,success=sum(r['success'] for r in episodes),
            n=len(episodes),warm_tail_ms_per_action=sum(r['warm_tail_ms'] for r in episodes)/sum(r['warm_tail_actions'] for r in episodes)))
    chosen = []
    for nfe in (1,2):
        group = [r for r in ranked if r['candidate']['fast_steps']==nfe]
        group.sort(key=lambda r:(-r['success'],r['warm_tail_ms_per_action'],r['candidate']['id']))
        chosen.append(group[0]['candidate'])
    result = dict(candidates=chosen,tuning=ranked,rule=manifest['selection']['closed_loop'],
        manifest_hash=manifest['manifest_hash'],time=time.time())
    atomic(OUT/'champions.json',result)
    return result
