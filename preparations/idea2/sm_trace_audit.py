"""CPU-only check of actual CUPTI Green Context attribution and stage ordering.

An unrestricted primary-context tail is resource eligibility, not evidence
that each tail kernel occupied every SM. Timing from this profile is diagnostic.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from .layer_trace import _read_database


def audit(path, *, early_sms, tail_sms=0, prefix_sms, steps=10, group=None):
    kernels, mapping = _read_database(path)
    connection = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    contexts = [dict(r) for r in connection.execute('SELECT * FROM TARGET_INFO_CUDA_CONTEXT_INFO')]
    connection.close()
    index = defaultdict(list)
    for c in contexts:
        index[c['processId'], c['deviceId'], c['contextId']].append(c)
    rows = [r for r in kernels if r['frame_id'] and r['branch'] in ('AE', 'Prefill')
            and (group is None or r['group'] == group)]
    errors, counts = [], Counter()
    frames = defaultdict(list)
    iterations = defaultdict(set)
    for r in rows:
        frames[r['frame_id']].append(r)
        if r['mapping_errors']:
            errors.append(dict(kind='mapping', kernel=r['kernel_rowid'], detail=r['mapping_errors']))
        if r['stage'] == 'VLM':
            phase, expected = 'VLM', prefix_sms
        elif r['branch'] == 'AE' and r['iteration'] is not None:
            iterations[r['frame_id']].add(r['iteration'])
            phase, expected = ('AE1', early_sms) if r['iteration'] == 1 else ('AE_tail', tail_sms)
        else:
            continue
        green = r['green_context']
        # globalPid has VM/system information above its 24-bit local PID.
        # Refuse ambiguous cross-VM matches rather than selecting one silently.
        matches = index.get(((r['process'] >> 24) & 0xffffff, r['device'], green), []) if green else []
        context = matches[0] if len(matches) == 1 else None
        actual = 0 if green == 0 else (context.get('numMultiprocessors') if context else None)
        counts[(phase, green, actual)] += 1
        if actual != expected:
            errors.append(dict(kind='wrong_resource', kernel=r['kernel_rowid'], phase=phase,
                               green_context=green, expected_sms=expected, actual_sms=actual))
    boundaries = []
    for frame, items in frames.items():
        ae1 = [r for r in items if r['branch'] == 'AE' and r['iteration'] == 1]
        tail = [r for r in items if r['branch'] == 'AE' and (r['iteration'] or 0) >= 2]
        vlm = [r for r in items if r['stage'] == 'VLM']
        if not ae1 or not tail or not vlm or iterations[frame] != set(range(1, steps + 1)):
            errors.append(dict(kind='incomplete_frame', frame=frame, iterations=sorted(iterations[frame])))
            continue
        tail_start = min(r['start'] for r in tail)
        prior_end = max(r['end'] for r in ae1 + vlm)
        ok = tail_start >= prior_end
        boundaries.append(dict(frame=frame, passed=ok, model_kernel_gap_us=(tail_start - prior_end)/1000))
        if not ok:
            errors.append(dict(kind='tail_started_before_producers_finished', frame=frame))
    if not rows or not boundaries:
        errors.append(dict(kind='missing_model_frames'))
    return dict(passed=not errors, source=str(path), expected=dict(early_sms=early_sms,
                tail_sms=tail_sms, prefix_sms=prefix_sms, num_steps=steps), contexts=contexts,
                counts=[dict(phase=k[0], green_context=k[1], sms=k[2], kernels=v)
                        for k,v in counts.items()], boundaries=boundaries, errors=errors,
                graph_mapping=mapping,
                scope='All attributed VLM/AE-iteration model kernels; 0 SM means unrestricted primary context, not zero or measured full occupancy. Boundary checks use model kernels; Event/Euler ordering is separately checked by runtime lifecycle gates.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--early-sms', type=int, required=True)
    parser.add_argument('--prefix-sms', type=int, required=True)
    parser.add_argument('--tail-sms', type=int, default=0)
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--group')
    args = parser.parse_args()
    result = audit(args.sqlite, early_sms=args.early_sms, prefix_sms=args.prefix_sms,
                   tail_sms=args.tail_sms, steps=args.num_steps, group=args.group)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ('passed','counts','errors')}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
