"""Prepare, inspect and serve adjustable mixed-skip budgets."""
import argparse
from fractions import Fraction
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COSTS = ROOT / 'results/idea1_v3_4_E2_mixed_cost/hardware_probe/cost_table.json'
DEFAULT_OUTPUT = ROOT / 'results/idea1_v3_4_E2_budgeted_skip/budgets'


def fraction(value):
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise argparse.ArgumentTypeError('Use a decimal fraction or a ratio such as 1/3') from error


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Build a CPU-only budget profile from measured hardware costs')
    prepare.add_argument('--cost-table', type=Path, default=DEFAULT_COSTS)
    prepare.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    target = prepare.add_mutually_exclusive_group(required=True)
    target.add_argument('--save-ms', type=float, dest='saved_ms')
    target.add_argument('--save-fraction', type=fraction, dest='saved_fraction',
                        help='Fraction of complete VLM encoder time to save, e.g. 0.3 or 1/3')
    prepare.add_argument('--tolerance-ms', type=float, default=.5)
    prepare.add_argument('--reference-vlm-ms', type=float,
                         help='Explicit measured complete VLM baseline; otherwise use hot keep records')
    select = commands.add_parser('select', help='Select a mask from saved (17,2) module scores; no model requests')
    select.add_argument('--budget', type=Path, required=True)
    select.add_argument('--scores', type=Path, required=True)
    select.add_argument('--score-key', help='Required array key for an NPZ with multiple score arrays')
    select.add_argument('--weighted', action='store_true')
    select.add_argument('--output', type=Path)
    serve = commands.add_parser('serve', help='Start a serial B1 native policy service with adjustable budgets')
    serve.add_argument('--budget', type=Path, required=True)
    serve.add_argument('--gpu', type=int, required=True)
    serve.add_argument('--port', type=int, default=0)
    serve.add_argument('--output', type=Path, required=True)
    serve.add_argument('--weighted', action='store_true')
    serve.add_argument('--actors', type=int, choices=(1, 2), default=1)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'prepare':
        from .budget import prepare_budget
        budget = prepare_budget(args.cost_table, args.output, saved_ms=args.saved_ms,
            saved_fraction=args.saved_fraction, tolerance_ms=args.tolerance_ms,
            reference_vlm_ms=args.reference_vlm_ms)
        result = {key: budget[key] for key in ('artifact_path', 'budget_id', 'budget_hash',
            'target_saved_ms', 'anchor_saved_ms', 'requested_saved_ms', 'requested_saved_fraction',
            'reference_vlm_ms', 'effective_window_ms', 'max_reachable_saved_ms', 'candidate_count')}
    elif args.command == 'select':
        import numpy as np
        from .budget import load_budget
        from preparations.idea1_v3_4_E2_mixed_cost.selection import select_mixed, mask_record
        budget = load_budget(args.budget)
        if args.scores.suffix == '.json':
            scores = np.asarray(json.loads(args.scores.read_text()))
        elif args.scores.suffix == '.npz':
            with np.load(args.scores, allow_pickle=False) as archive:
                key = args.score_key or (archive.files[0] if len(archive.files) == 1 else None)
                if key is None:
                    raise ValueError('--score-key is required for multiple NPZ arrays')
                scores = archive[key].copy()
        else:
            scores = np.load(args.scores, allow_pickle=False)
        a, f = select_mixed(scores, budget, args.weighted)
        result = mask_record(a, f, budget, scores=scores, cost_weighted=args.weighted)
        result['cost_weighted'] = args.weighted
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    else:
        from .server import serve
        serve(args)
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
