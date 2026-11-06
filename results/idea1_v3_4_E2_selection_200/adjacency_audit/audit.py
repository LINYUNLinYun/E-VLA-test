"""Inspect executed masks without changing or rerunning closed-loop episodes."""
import collections
import hashlib
import json
import time
from pathlib import Path

ROOT = Path('/root/nfs/embodied/benchmark')
OUT = ROOT / 'results/idea1_v3_4_E2_selection_200'
DEST = OUT / 'adjacency_audit'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def longest_run(mask):
    longest = current = 0
    previous = None
    for layer in sorted(mask):
        current = current + 1 if previous is not None and layer == previous + 1 else 1
        longest = max(longest, current)
        previous = layer
    return longest


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert all(sha(ROOT / name) == value for name, value in manifest['source_files'].items())
    result = dict(
        time=time.time(), manifest_sha256=sha(OUT / 'manifest.json'),
        script_sha256=sha(Path(__file__)), conditions=[],
        note='Selection-event frequencies describe each method own trajectories; they are not a causal adjacency ablation.',
    )
    for cfg in manifest['conditions']:
        if cfg['phase'] != 'expand200':
            continue
        requests = fast_requests = selections = successes = 0
        selected_hist = collections.Counter()
        fast_hist = collections.Counter()
        selected_pair_hist = collections.Counter()
        masks = collections.Counter()
        infer_hashes = {}
        violations = []
        for case in manifest['cases']:
            key = f"{case['suite']}_task{case['task']:02d}_init{case['init']:02d}"
            ref = manifest['reuse'][cfg['id']].get(key)
            directory = Path(ref['directory']) if ref else OUT / cfg['id'] / 'episodes' / case['suite'] / f"task{case['task']:02d}" / f"init{case['init']:02d}"
            data = json.loads((directory / 'result.json').read_text())
            assert data['valid'] and data['case'] == case
            successes += int(data['success'])
            infer = directory / 'infer.jsonl'
            infer_hashes[str(infer)] = sha(infer)
            if ref:
                assert infer_hashes[str(infer)] == ref['hashes']['infer.jsonl']
            for line in infer.read_text().splitlines():
                q = json.loads(line)
                route = q['route']
                mask = route['selected_skip']
                run = longest_run(mask)
                requests += 1
                assert len(mask) == cfg['skip_count'] and len(set(mask)) == len(mask)
                assert all(1 <= i <= 17 for i in mask)
                if run > 2:
                    violations.append(dict(case=case, step=q['step'], selected_skip=mask))
                if route['mask_refresh']:
                    selections += 1
                    selected_hist[run] += 1
                    selected_pair_hist[sum(i+1 in mask for i in mask)] += 1
                    masks[tuple(mask)] += 1
                if route['path'] == 'fast':
                    fast_requests += 1
                    assert route['skip'] == mask
                    fast_hist[longest_run(route['skip'])] += 1
        row = dict(
            condition=cfg['id'], episodes=200, success=successes,
            requests=requests, selection_events=selections, fast_requests=fast_requests,
            longest_run_at_selection=dict(sorted(selected_hist.items())),
            longest_run_at_actual_fast=dict(sorted(fast_hist.items())),
            number_of_adjacent_pairs_at_selection=dict(sorted(selected_pair_hist.items())),
            fraction_selections_with_adjacent_pair=1-selected_hist[1]/selections,
            largest_run_observed=max(selected_hist), violations=violations,
            most_frequent_masks=[dict(skip=list(mask), selection_events=n) for mask,n in masks.most_common(5)],
        )
        result['conditions'].append(row)
        (DEST / f"{cfg['id']}_infer_hashes.json").write_text(json.dumps(infer_hashes,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    result['passed'] = all(not r['violations'] and r['largest_run_observed'] <= 2 for r in result['conditions'])
    assert result['passed']
    (DEST / 'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines = [
        '# 200例选层连续性核验', '',
        '核验冻结配置、评分候选生成及全部8个扩样条件的实际逐请求记录。所有条件原本均禁止三层及以上连续跳过，允许相邻两层；Full选定mask与Fast实际执行mask分别核对。没有重新运行闭环。', '',
        '|条件|成功数|Full选层次数|选层含相邻两层比例|实际Fast请求数|最大连续跳层|三连及以上违规|',
        '|---|---|---|---|---|---|---|',
    ]
    for row in result['conditions']:
        lines.append(f"|{row['condition']}|{row['success']}/200|{row['selection_events']}|{100*row['fraction_selections_with_adjacent_pair']:.2f}%|{row['fast_requests']}|{row['largest_run_observed']}|{len(row['violations'])}|")
    lines += [
        '', '因此，已报告的Jacobian无补偿结果就是“最多连续跳两层”的实测结果。它低于余弦的差距不能由违反这个约束的三连跳解释。相邻两层本身是否仍有害，需要另做最多连续一层的受控对照；本表频率来自各自自由闭环，不能据此判定连续性导致成功率差异。',
        '', '证据：preparations/idea1_v3_4_E2_cycle3/common.py中combinations(k)排除任意{i,i+1,i+2}同时被选中，choose在合法集合中最小化评分；新统一policy的Fast分支再检查legal。逐记录核验及hash见results/idea1_v3_4_E2_selection_200/adjacency_audit/。',
    ]
    (ROOT/'docs/experiments/idea1/idea1_v3_4_E2_selection_adjacency_audit.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
