"""Publish only validator-complete 50-episode suite results; keep pending explicit."""
from collections import Counter
import json

from scripts.experiment10.common import RESULTS, ROOT, SUITES, dump_json, sha256


def main():
    rows = []
    config = json.loads((ROOT / 'scripts/experiment10/closed_loop_config.json').read_text())
    for specification in config['conditions']:
        condition = RESULTS / 'closed_loop' / specification['name']
        for suite in SUITES:
            root = condition / suite / 'run'
            validated, episodes_path = root / 'validated.json', root / 'episode_log.json'
            if not validated.exists():
                rows.append({'condition': condition.name, 'suite': suite, 'status': 'not_validated'})
                continue
            validation = json.loads(validated.read_text())
            if validation['condition'] != specification:
                raise RuntimeError('Condition differs from experiment configuration')
            episodes = json.loads(episodes_path.read_text())
            expected = {(t, i) for t in range(10) for i in range(5)}
            keys = {(e['task_id'], e['episode_idx']) for e in episodes}
            if not validation['complete'] or validation['episodes'] != 50 or len(episodes) != 50 or keys != expected:
                raise RuntimeError('Invalid completed coverage: '+str(root))
            if any(e.get('failure_reason') for e in episodes):
                raise RuntimeError('Runtime errors must not be counted as normal task failure')
            if validation['full_video_decodes'] != 50:
                raise RuntimeError('Missing video checks')
            for episode in episodes:
                for field in ('video_path', 'trace_path', 'infer_path', 'telemetry_path'):
                    if not (root / episode[field]).is_file():
                        raise RuntimeError('Missing artifact '+str(root / episode[field]))
            successful = sum(bool(e['success']) for e in episodes)
            task_rows = [{'task_id': task, 'successes': sum(bool(e['success']) for e in episodes if e['task_id'] == task),
                          'episodes': 5} for task in range(10)]
            stages = Counter(e.get('furthest_semantic_phase', 'unknown') for e in episodes if not e['success'])
            rows.append({'condition': condition.name, 'suite': suite, 'status': 'validated_complete',
                         'successes': successful, 'episodes': 50, 'success_rate': successful/50,
                         'task_results': task_rows, 'failure_highest_stage_counts': dict(stages),
                         'stage_warning': 'Highest reached stage is not the first error or its causal location',
                         'validated_sha256': sha256(validated), 'episode_log_sha256': sha256(episodes_path),
                         'timing_warning': 'No dedicated timing claim; startup and prompt graph warmup included in logs'})
    output = RESULTS / 'closed_loop_analysis'
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / 'validated_results.json', {'suite_rows': rows,
              'whole_experiment_complete': False, 'only_validator_complete_results_counted': True})
    print(json.dumps([{k: row[k] for k in ('condition', 'suite', 'status', 'successes', 'episodes') if k in row}
                      for row in rows], indent=2), flush=True)


if __name__ == '__main__':
    main()
