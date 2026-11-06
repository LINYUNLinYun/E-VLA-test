"""Fresh-model-process installation and the diagnosed-recovery completion gate."""
import hashlib
import json
from pathlib import Path


def kernel_digest():
    return hashlib.sha256(Path(__file__).with_name('safe_qk.py').read_bytes()).hexdigest()


def require_recovery_pass(guard):
    guard=Path(guard).resolve(); data=json.loads(guard.read_text())
    if data.get('status')!='valid' or data.get('revision')!='qk-row-bounds-v1':
        raise RuntimeError('Single-key diagnosed recovery has not completed validly')
    if data.get('source_sha256',{}).get('safe_qk.py')!=kernel_digest():
        raise RuntimeError('Recovery checked a different safety kernel revision')
    reference=Path(data['reference_receipt']); receipt=json.loads(reference.read_text())
    if receipt.get('passed') is not True or receipt.get('source_sha256')!=kernel_digest():
        raise RuntimeError('Single-key QK reference check has not passed this kernel revision')
    attempt=guard.parent.parent.parent
    manifest=json.loads((attempt/'manifest.json').read_text())
    row=next(row for row in manifest['tasks'] if row['key']==data['key'])
    directory=attempt/'episodes'/row['experiment']/row['condition']/row['execution']/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"
    completion=json.loads((directory/'complete.json').read_text())
    if completion.get('valid') is not True or completion.get('key')!=data['key']:
        raise RuntimeError('Diagnosed recovery completion marker is not valid')
    return dict(guard=str(guard),key=data['key'],reference_receipt=str(reference),
        kernel_sha256=kernel_digest(),revision='qk-row-bounds-v1')


def install_in_model_process(receipt):
    """One first-QK check per worker/bench; subsequent contexts reuse that check."""
    import sys
    from preparations.idea1_v3.safe_qk import install,bounded_matmul_abT_scale
    receipt=Path(receipt)
    if not receipt.exists():
        install(receipt)
        return
    previous=json.loads(receipt.read_text())
    if previous.get('passed') is not True or previous.get('source_sha256')!=kernel_digest():
        raise RuntimeError('Existing worker QK reference receipt failed or has a different revision')
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'policies/realtime-vla-flash'))
    from scripts.spec.triton import pi0_infer as kernels
    kernels.matmul_abT_scale=bounded_matmul_abT_scale
