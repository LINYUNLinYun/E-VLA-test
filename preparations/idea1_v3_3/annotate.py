"""Append engineering provenance sidecars; never rewrite completed evidence."""
import hashlib
import json
from pathlib import Path
import re
from preparations.idea1_v3.io import atomic
from .analyze import OUT,records
from .manifest import cases

_models={}


def model_gpu(pid):
    if pid not in _models:
        paths=list(OUT.glob('workers/gpu*/loaded_protocol.json'))+list(OUT.glob('workers/gpu*/history/*/loaded_protocol.json'))+list(OUT.glob('stages/*/gpu*/loaded_protocol.json'))
        for p in paths:
            data=json.loads(p.read_text()); protocol=data.get('protocol',{})
            match=re.search(r'/gpu(\d+)(?:/|$)',str(p))
            if match and 'server_pid' in protocol:_models[protocol['server_pid']]=(int(match.group(1)),str(p))
    return _models[pid]


def case(row):
    directory=Path(row['_directory']);output=directory/'engineering_revision.json'
    if output.exists():return
    identity_path=directory/'identity.json';identity=json.loads(identity_path.read_text());pid=identity['protocol']['server_pid'];gpu,model_receipt=model_gpu(pid)
    # Original first80 records predate the physical EGL correction. The only
    # mixed condition has an explicit durable56 boundary, not an mtime guess.
    cfg=row['config'];cid=row['condition'];key=row['case'];first=key in cases(cfg['model'])
    corrected=identity['protocol'].get('E2_gate_revision')=='strict-c1' or not first
    basis='C1 dispatcher/stage uses physical EGL mapping' if corrected else 'original dispatcher assigned EGL ordinal equal to CUDA physical index'
    if cfg['model']=='xiaomi' and cfg['experiment']=='E4':
        if cid=='E4_tau0.05':
            corrected=cases('xiaomi').index(key)>=56;basis='explicit durable56 episode handoff; fixed manifest order'
        elif cid not in ('E4_tau0.025','E4_tau0.1'):
            corrected=True;basis='condition started by corrected EGL driver after tau.05/tau.1 handoffs'
    render=gpu if corrected else 3-gpu
    index=3-gpu if corrected else gpu
    sidecar=dict(case=key,condition=cid,model_gpu=gpu,model_server_pid=pid,model_service_receipt=model_receipt,render_gpu=render,egl_index=index,
        engineering_revision='egl-physical-binding-v2' if corrected else 'legacy-ordinal-binding',binding_basis=basis,
        identity_sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest(),result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
        scientific_manifest_hash=row['manifest_hash'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if corrected:
        receipt=OUT/f'egl_parity_gpu{gpu}.json'
        if not receipt.exists() or not json.loads(receipt.read_text())['passed']:raise ValueError('Missing physical binding parity receipt')
        sidecar['initial_rgb_state_parity_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()
    atomic(output,sidecar)


if __name__=='__main__':
    groups=records()
    for rows in groups.values():
        for row in rows:case(row)
