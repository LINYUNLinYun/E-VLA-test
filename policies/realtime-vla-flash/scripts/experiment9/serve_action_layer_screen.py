"""Experiment-only server adapter for one Action Expert layer skip; no production edits."""
import argparse
import dataclasses
import logging
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))


def register_layer(runtime, layer):
    pattern=f'ae_skip_l{layer}'
    indices=tuple(i for i in range(18) if i!=layer-1)
    runtime._VALID_ABLATION_LAYER_PATTERNS=runtime._VALID_ABLATION_LAYER_PATTERNS | {pattern}
    runtime._ABLATION_LAYER_INDICES[pattern]=indices
    original=runtime._spec_runtime_module
    def load(filename,name):
        module=original(filename,name)
        if filename=='pi0_infer.py':
            module._ACTION_LAYER_PATTERNS[pattern]=indices
        return module
    runtime._spec_runtime_module=load
    return pattern


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--skip-action-layer',type=int,choices=range(1,19),required=True)
    extra,remainder=parser.parse_known_args()
    import tyro
    from scripts.spec import spec_serve_policy as server
    pattern=register_layer(server._triton_runtime,extra.skip_action_layer)
    args=tyro.cli(server.Args,args=remainder)
    if args.vlm_layer_pattern!='vlm_full18' or args.denoise_steps!=10 or not args.force_full_each_round:
        raise ValueError('Screening requires full VLM, D10 and force-full rounds')
    args=dataclasses.replace(args,backend='triton',action_layer_pattern=pattern)
    logging.basicConfig(level=logging.INFO,force=True)
    server.main(args)


if __name__=='__main__':
    main()
