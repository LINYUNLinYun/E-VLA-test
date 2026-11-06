"""CPU E4 preparation and explicit analysis stages. Default: no work or writes."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=('sample-check', 'freeze', 'reproduce', 'tune', 'evaluate', 'decompose'))
    parser.add_argument('--source', type=Path, help='Original experiment11 source (read only)')
    parser.add_argument('--v2', type=Path, help='Completed v2 attempt (read only)')
    parser.add_argument('--output', type=Path, help='.../results/idea1_v2_1_multimodel_kalman/<attempt>/e4_kalman')
    parser.add_argument('--launch', action='store_true', help='Explicitly execute this CPU stage')
    parser.add_argument('--dry-run', action='store_true', help='Preview only; overrides --launch')
    parser.add_argument('--resume', action='store_true', help='tune only: reuse validated complete candidate files')
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    preview = dict(command=args.command, dry_run=not args.launch or args.dry_run,
                   source=str(args.source), v2=str(args.v2), output=str(args.output),
                   max_candidate_slots=211, folds=5, calibration_points=634, evaluation_points=6345,
                   model_calls=0, environment_actions=0, gpu_jobs=0,
                   stages='freeze -> reproduce -> tune -> evaluate -> decompose',
                   completion='Core CPU stages only; final report/figures/cluster CIs/cross tables pending')
    if preview['dry_run']:
        print(json.dumps(preview, indent=2))
        return
    if args.output is None and args.command != 'sample-check':
        parser.error('--output required for execution')
    if args.command in ('sample-check', 'reproduce', 'tune', 'evaluate') and args.source is None:
        parser.error('--source required for this stage')
    if args.command == 'reproduce' and args.v2 is None:
        parser.error('--v2 required for reproduction')
    if args.resume and args.command != 'tune':
        parser.error('--resume is supported only for tune')
    from . import pipeline
    if args.command == 'sample-check':
        print(json.dumps(pipeline.sample_check(args.source), indent=2, allow_nan=False))
        return
    root = pipeline.output_root(args.output)
    if args.command == 'freeze':
        result = pipeline.freeze(root)
    elif args.command == 'reproduce':
        result = pipeline.reproduce(root, args.source, args.v2)
    elif args.command == 'tune':
        result = pipeline.tune(root, args.source, args.resume)
    elif args.command == 'evaluate':
        result = pipeline.evaluate(root, args.source)
    else:
        result = pipeline.decomposition(root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
