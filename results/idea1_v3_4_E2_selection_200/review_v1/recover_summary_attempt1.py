"""Offline schema adapter; original frozen experiment sources remain intact."""
import json
import time
from pathlib import Path

import numpy as np

from preparations.idea1_v3_4_E2_selection_200 import finish
from preparations.idea1_v3_4_E2_selection_200.common import OUT, ROOT, sha
from preparations.idea1_v3.io import atomic


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    legacy_paths = {
        Path(ref['directory']): compensation
        for bank, compensation in [('fixed_none', 'none'), ('fixed', 'extrapolate')]
        for ref in manifest[bank].values()
    }
    assert len(legacy_paths) == 400
    original_header = finish.header
    adapted = []

    def legacy_header(directory, result=None):
        directory = Path(directory)
        if directory not in legacy_paths:
            return original_header(directory, result)
        if result is None:
            result = json.loads((directory / 'result.json').read_text())
        cfg = result['config']
        compensation = legacy_paths[directory]
        assert cfg['skip'] == [2, 3, 5, 14, 16, 17]
        assert cfg['mode'] == 'periodic' and cfg['period'] == 2
        assert cfg['H'] == 50 and cfg['R'] == 5 and cfg['D'] == 2 and cfg['refresh_D'] == 10
        assert cfg['compensation'] == (None if compensation == 'none' else 'extrapolate')
        identity = json.loads((directory / 'identity.json').read_text())
        assert identity['config'] == cfg
        requests = finish.rows(directory)
        assert requests[0]['route']['path'] == 'full'
        assert 'selected_skip' not in requests[0]['route']
        for i, request in enumerate(requests):
            full = i % 3 == 0
            assert request['route']['path'] == ('full' if full else 'v12')
            assert len(request['timing']['nodes']) == (11 if full else 3)
        with np.load(directory / 'arrays.npz') as arrays:
            h = dict(
                result=result, identity=finish.identity(directory),
                first_mask=cfg['skip'].copy(),
                raw0=arrays['raw_0'].copy(), actions0=arrays['actions_0'].copy(),
                fast_input=None if len(requests)<2 else requests[1]['input_hash'],
            )
            if len(requests)>1:
                h.update(raw1=arrays['raw_1'].copy(), actions1=arrays['actions_1'].copy())
        adapted.append(dict(
            directory=str(directory), first_mask_source='result.config.skip',
            inferred_mask=cfg['skip'], requests=len(requests),
            result_sha256=sha(directory/'result.json'),
            identity_sha256=sha(directory/'identity.json'),
            infer_sha256=sha(directory/'infer.jsonl'),
        ))
        return h

    review = OUT / 'review_v1'
    atomic(review / 'started.json', dict(
        time=time.time(), reason='Legacy global fixed reference stores skip in result.config.skip, not route.selected_skip.',
        frozen_finish_sha256=sha(ROOT/'preparations/idea1_v3_4_E2_selection_200/finish.py'),
        adapter_sha256=sha(Path(__file__)),
        original_failed_json_sha256=sha(OUT/'failed.json'),
    ))
    finish.header = legacy_header
    finish.main()
    assert len(adapted) == 400
    atomic(review / 'adapted_fixed_references.json', adapted)
    atomic(OUT/'aggregation_recovery.json', dict(
        passed=True, time=time.time(), original_failure_preserved='failed.json',
        adapter='review_v1/recover_summary.py', adapted_legacy_references=400,
        frozen_source_hashes_unchanged=True, evaluation_reruns=0,
        scope='Offline legacy-header reading only; original route, prefix and statistical checks retained.',
    ))


if __name__ == '__main__':
    main()
