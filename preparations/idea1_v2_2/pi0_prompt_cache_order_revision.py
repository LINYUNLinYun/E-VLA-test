"""Order-controlled gate only; preserves the failed v1 cache implementation."""
import argparse
from pathlib import Path

from common import atomic_json, sha
import pi0_prompt_cache_revision as original
from throughput_revision import read

HERE = Path(__file__).resolve()


def validate_order(identity):
    if identity.get('reference_order') != list(range(8)) or identity.get('cached_order') != list(range(8)):
        raise ValueError('Both independent servers must receive the identical ABAB request order')
    controller = identity.get('gate_controller', {})
    if (controller.get('source') != str(HERE) or controller.get('sha256') != sha(HERE)
            or controller.get('scope') != 'gate_only_order_control'
            or controller.get('failed_revision_manifest_sha256') != sha(Path(controller['failed_revision'])/'revision_manifest.json')):
        raise ValueError('Order controller or preserved failed revision identity differs')


def prepare(attempt, base_revision, revision, failed_revision):
    failed_revision = Path(failed_revision).resolve()
    failed, _ = original.verify_revision(failed_revision)
    if failed['reference_order'] != [0, 2, 4, 6, 1, 3, 5, 7] or failed['cached_order'] != list(range(8)):
        raise ValueError('Expected preserved original grouped-reference gate')
    identity = original.prepare(attempt, base_revision, revision)
    identity.update(reference_order=list(range(8)), cached_order=list(range(8)),
        gate_controller=dict(source=str(HERE), sha256=sha(HERE), scope='gate_only_order_control',
            failed_revision=str(failed_revision), failed_revision_manifest_sha256=sha(failed_revision/'revision_manifest.json'),
            scientific_change='none; identical original cache, kernels, masks, observations, seeds and padding'),
        diagnostic_contrast='fresh reference and cached servers both receive index order 0..7; no output tolerance')
    validate_order(identity)
    atomic_json(Path(revision)/'revision_manifest.json', identity)
    return identity


def gate(revision, gpu):
    identity, _ = original.verify_revision(revision)
    validate_order(identity)
    return original.gate(revision, gpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'gate'))
    parser.add_argument('--revision', type=Path, required=True)
    parser.add_argument('--attempt', type=Path)
    parser.add_argument('--base-revision', type=Path)
    parser.add_argument('--failed-revision', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.operation == 'prepare':
        if any(value is None for value in (args.attempt, args.base_revision, args.failed_revision)):
            parser.error('prepare requires attempt, base-revision and failed-revision')
        prepare(args.attempt, args.base_revision, args.revision, args.failed_revision)
    else:
        gate(args.revision, args.gpu)


if __name__ == '__main__':
    main()
