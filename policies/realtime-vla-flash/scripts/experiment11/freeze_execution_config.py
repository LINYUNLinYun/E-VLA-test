"""Create the formal config namespace, separate from the already-run smoke."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.experiment10.common import dump_json
from scripts.experiment11.protocol import RESULTS
from scripts.experiment11.build_candidates import sha256


def main():
    protocol_dir = RESULTS / 'protocol'
    manifest_path = protocol_dir / 'candidate_manifest.json'
    candidate_config = json.loads((protocol_dir / 'closed_loop_config.json').read_text())
    manifest = json.loads(manifest_path.read_text())
    by_id = {row['variant_id']: row for row in manifest['candidates']}
    reference = {condition['name']: condition for condition in manifest['reference_conditions']}
    conditions = []
    for condition in candidate_config['conditions']:
        if condition['name'] == 'Full18_D3_R5':
            item = dict(condition)
            item['name'] += '_FORMAL'
            item['candidate_id'] = 'Full18_D3_R5'
            conditions.append(item)
        elif condition['name'] == 'V13_Custom_D3_R5':
            item = dict(condition)
            item['name'] += '_FORMAL'
            item['candidate_id'] = 'V13_Custom'
            conditions.append(item)
        else:
            row = next((c for c in manifest['candidates'] if c['condition']['name'] == condition['name']), None)
            if row is None:
                raise RuntimeError('Condition is not in the frozen candidate manifest: ' + condition['name'])
            item = dict(condition)
            item['name'] += '_FORMAL'
            item['candidate_id'] = row['variant_id']
            conditions.append(item)
    if len(conditions) != 31 or len({c['name'] for c in conditions}) != 31:
        raise RuntimeError('Formal condition matrix is incomplete or duplicated')
    config = {'schema': 3,
              'purpose': 'Formal 6200-episode matrix; namespaced separately from smoke outputs',
              'candidate_manifest_sha256': sha256(manifest_path),
              'source_candidate_config_sha256': sha256(protocol_dir / 'closed_loop_config.json'),
              'conditions': conditions}
    output = protocol_dir / 'closed_loop_config_formal.json'
    if output.exists():
        saved = json.loads(output.read_text())
        if saved != config:
            raise RuntimeError('Formal config already exists with a different frozen protocol')
    else:
        dump_json(output, config)
    print(json.dumps({'config': str(output), 'config_sha256': sha256(output),
                      'candidate_manifest_sha256': config['candidate_manifest_sha256'],
                      'conditions': len(conditions), 'episodes': len(conditions) * 200,
                      'gpu_jobs': len(conditions) * 4}, indent=2))


if __name__ == '__main__':
    main()
