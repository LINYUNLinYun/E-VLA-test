"""Bounded gate-access wait followed by a pinned, resumable model download."""
import json,os,time
from pathlib import Path

os.environ.setdefault('HF_HOME','/root/.cache/huggingface')
os.environ.setdefault('HF_HUB_CACHE','/root/nfs/embodied/benchmark/shared/caches/huggingface/hub')
from huggingface_hub import HfApi,snapshot_download
from huggingface_hub.errors import GatedRepoError
from experiments.pi05_hidden_probe.common import write_json

base=Path(__file__).resolve().parent
out=Path('/root/nfs/embodied/benchmark/shared/checkpoints/pi05_hidden_probe/paligemma-3b-mix-224')
repo='google/paligemma-3b-mix-224';api=HfApi();start=time.monotonic()
api.whoami()
for attempt in range(60):
    try:
        api.auth_check(repo,token=True)
        break
    except GatedRepoError:
        if attempt==0:print('HF login valid; waiting for model-specific access terms (maximum 30 minutes)',flush=True)
        write_json(base/'target_download_status.json',{'state':'awaiting_model_access','hf_authentication_valid':True,'repository':repo,'attempt':attempt+1,'elapsed_seconds':time.monotonic()-start})
        time.sleep(30)
else:
    write_json(base/'target_download_status.json',{'state':'model_access_still_denied','hf_authentication_valid':True,'repository':repo,'elapsed_seconds':time.monotonic()-start})
    raise SystemExit('Model access still HTTP403; account must accept/request the model terms.')
info=api.model_info(repo)
print('Access authorized; downloading pinned revision',info.sha,flush=True)
write_json(base/'target_download_status.json',{'state':'downloading','repository':repo,'revision':info.sha})
snapshot_download(repo,revision=info.sha,token=True,local_dir=out,
    allow_patterns=['*.json','*.model','*.safetensors','tokenizer*','special_tokens_map.json'],max_workers=4)
files=[{'path':str(p.relative_to(out)),'size':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and '.cache' not in p.parts]
write_json(out/'download_identity.json',{'repository':repo,'revision':info.sha,'files':files})
write_json(base/'target_download_status.json',{'state':'complete','repository':repo,'revision':info.sha,'path':str(out),'elapsed_seconds':time.monotonic()-start})
print('DOWNLOAD_COMPLETE',out,flush=True)
