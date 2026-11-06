import json,os,time
from pathlib import Path
import torch
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_selection_cost.common import ROOT,sha,write_csv

OUT=ROOT/'results/idea1_v3_4_selection_async'
BANK=ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json'
REFERENCE=ROOT/'results/idea1_v3_4_selection_cost/benchmark/gradients.npz'


def setup(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',SPEC_TRITON_DRAFT_GRAPH='0',
        SPEC_TRITON_VERIFY_GRAPH='0',SPEC_TRITON_VERIFY_SEMANTICS_GRAPH='0')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    from preparations.idea1_v3.safe_qk import install
    from preparations.idea1_v3.policy import TritonBackend
    install(str(out/'qk_receipt.json'));tick=time.perf_counter();backend=TritonBackend()
    atomic(out/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        model_load_s=time.perf_counter()-tick,protocol=backend.protocol,
        source_files={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
        bank_sha256=sha(BANK),reference_sha256=sha(REFERENCE),time=time.time()))
    return backend


def samples():return json.loads(BANK.read_text())['samples']


def prepare(backend,sample):
    from preparations.kv_reuse_probe.probe import load_observation
    obs,prompt=load_observation(sample['current']);seed=sample['current']['seed']
    c,_=backend.prepare(obs,prompt,());_,raw,_=backend.sample(c,seed,10)
    backend.verify(c,c,raw,seed,sample['suite'])
    return c,seed
