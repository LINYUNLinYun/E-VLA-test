"""Original sequential server with an explicitly receipted pi0-only cache patch."""
import argparse
import os
from pathlib import Path

from common import gpu_lease, sha
import runtime_models
import runtime_server
from pi0_prompt_cache import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--reference', action='store_true')
    args = parser.parse_args()
    from pi0_prompt_cache_revision import verify_revision
    verify_revision(args.revision)
    original = runtime_models.NativePolicy
    def create(model, output_dir):
        policy = original(model, output_dir)
        if not args.reference:
            identity, _ = verify_revision(args.revision)
            from pi0_prompt_cache import function_identity
            if function_identity(type(policy.pi0.backend).prepare) != identity['original_prepare']:
                raise ValueError('Actual loaded Backend.prepare identity differs from reviewed function')
            install(policy, output_dir, sha(Path(args.revision)/'revision_manifest.json'))
        return policy
    runtime_models.NativePolicy = create
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.isdecimal():
        raise ValueError('One physical GPU and its inherited exclusive lease are required')
    args.model = 'pi0'
    with gpu_lease(gpu):
        runtime_server.serve(args)


if __name__ == '__main__':
    main()
