"""Guarded native-interface audit, real-demo calibration and expert replay entrypoint."""
import argparse
import json
from pathlib import Path
from .native_envs import (formal_calvin_contract,native_calvin_config_audit,official_calvin_sequences,
    replay_calvin_demonstrations,calibrate_calvin_native,replay_pusht_expert,write_json,digest,
    SIMPLER_NATIVE_TASKS,simpler_native_contract,simpler_native_asset_audit,diagnose_calvin_demo,calibrate_simpler_native,capture_calvin_native_input)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=('calvin-audit','calvin-sequences','calvin-demo','calvin-calibration','calvin-demo-diagnostic','calvin-capture','simpler-audit','simpler-calibration','pusht-replay'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--sequence-file')
    parser.add_argument('--count',type=int,default=1000)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--task',choices=tuple(SIMPLER_NATIVE_TASKS),default='google_robot_pick_coke_can')
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    if not args.launch:
        print(json.dumps(dict(dry_run=True,gpu_used=False,**vars(args))));return
    output=Path(args.output)
    if args.stage=='calvin-audit':
        result=dict(status='complete',contract=formal_calvin_contract(),native_assets=native_calvin_config_audit())
        write_json(output/'receipt.json',result)
    elif args.stage=='calvin-sequences':
        rows=official_calvin_sequences(args.count,args.workers)
        result=dict(status='complete',contract=formal_calvin_contract(),sequences=rows,sequence_sha256=digest(rows),
            selection='canonical full1000 generated first; preregistered prefix, never regenerate smaller differentlydistributed set')
        write_json(output/'sequences.json',result)
    elif args.stage=='calvin-demo': result=replay_calvin_demonstrations(output)
    elif args.stage=='calvin-calibration': result=calibrate_calvin_native(output)
    elif args.stage=='calvin-demo-diagnostic': result=diagnose_calvin_demo(output)
    elif args.stage=='calvin-capture':
        if not args.sequence_file: parser.error('--sequence-file required')
        result=capture_calvin_native_input(args.sequence_file,output)
    elif args.stage=='simpler-audit':
        result=simpler_native_asset_audit()
        write_json(output/'receipt.json',result)
    elif args.stage=='simpler-calibration':
        if args.gpu is None or not args.gpu_uuid: parser.error('Explicit assigned --gpu and --gpu-uuid required')
        result=calibrate_simpler_native(args.task,args.gpu,args.gpu_uuid,output)
    else:
        if not args.dataset: parser.error('--dataset required for actual expert replay')
        result=replay_pusht_expert(args.dataset,output,args.count)
    print(json.dumps({k:result[k] for k in ('status','passed','successes','total','sequence_sha256') if k in result}))


if __name__=='__main__': main()
