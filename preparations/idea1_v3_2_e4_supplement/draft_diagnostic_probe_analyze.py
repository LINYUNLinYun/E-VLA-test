"""CPU input-probe evidence, separating executed prefix, future chunk and padded raw channels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .draft_diagnostic_audit import DEFAULT_OUT,dump
from .draft_diagnostic_language import action_difference


def compare_raw(first,second):
    a,b=np.asarray(first,np.float32),np.asarray(second,np.float32)
    if a.shape!=b.shape or a.shape!=(50,32):raise ValueError('Expected raw50x32')
    different=np.flatnonzero(np.any(a[:,:7]!=b[:,:7],axis=1))
    return dict(all_bitwise_equal=bool(np.array_equal(a,b)),effective7_bitwise_equal=bool(np.array_equal(a[:,:7],b[:,:7])),
        effective7_max_abs=float(np.max(np.abs(a[:,:7]-b[:,:7]))),
        first5_effective7_bitwise_equal=bool(np.array_equal(a[:5,:7],b[:5,:7])),
        first5_effective7_max_abs=float(np.max(np.abs(a[:5,:7]-b[:5,:7]))),
        padding25_bitwise_equal=bool(np.array_equal(a[:,7:],b[:,7:])),padding25_max_abs=float(np.max(np.abs(a[:,7:]-b[:,7:]))),
        different_effective_rows_zero_based=different.tolist(),
        first_different_effective_row_zero_based=int(different[0]) if len(different) else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=DEFAULT_OUT);a=p.parse_args();rows=[]
    for path in sorted((a.directory/'language_same_observation').glob('*_responses.jsonl')):
        recorded=[json.loads(line) for line in path.open()];first={}
        for record in recorded:first.setdefault(record['arm'],record)
        for arm1,arm2 in (('actual','mother_bddl'),('mother_bddl','legacy_filename'),('actual','legacy_filename')):
            x,y=first[arm1],first[arm2];xr,yr=x['response'],y['response']
            def valid_scales(response):
                return dict(normalized_state=response['feature_scales']['normalized_state'],
                    visual_preencoder=dict(valid=False,reason='Old diagnostic service read uninitialized encoder_x visual rows in encode=False draft route; discard even when finite. Actual projected prefix lives in spec._vision_prefix_buf.'))
            rows.append(dict(head=path.name.removesuffix('_responses.jsonl'),first_arm=arm1,second_arm=arm2,
                first_instruction=x['instruction'],second_instruction=y['instruction'],
                decoded=action_difference(xr['actions'],yr['actions']),raw=compare_raw(xr['raw_actions'],yr['raw_actions']),
                first_feature_scales=valid_scales(xr),second_feature_scales=valid_scales(yr),
                input_hash_equal=xr['input_hash']==yr['input_hash'],source_path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    dump(a.directory/'language_probe_full_chunk_audit.json',dict(comparisons=rows,count=len(rows),
        interpretation='First R5 is the executed prefix. Future H50 differences show conditioning sensitivity but do not describe executed motion before replanning. Raw32 channels0:7 are effective native action channels;7:32 are padding. Input hash includes instruction, so prompt-only pairs should differ there.'))
    print(json.dumps(dict(comparisons=len(rows))))


if __name__=='__main__':main()
