"""Run the protocol with the independently validated cached text generator."""
import argparse,json,hashlib
from pathlib import Path
from cached_target import CachedHFBackend
from experiments.pi05_hidden_probe.backend import Backend
from experiments.pi05_hidden_probe.common import load_config,select_rows,prepare_run,write_json
from experiments.pi05_hidden_probe.workflow import integrity_check,run_probe

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--limit-groups',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
base=Path(__file__).resolve().parent
validation=json.loads((base/'cached_generation_validation.json').read_text())
if not validation['passed']:raise RuntimeError('Cached generator has not passed its preregistered gate')
if validation['implementation_sha256']!=hashlib.sha256((base/'cached_target.py').read_bytes()).hexdigest():raise RuntimeError('Cached implementation changed after validation')
config=load_config(a.config);rows=select_rows(config,a.limit_groups)
spec={k:v for k,v in config['target_model'].items() if k not in ['backend','decode_implementation']}
target=CachedHFBackend(**spec)
source=Backend(**config['model'])
out,identity=prepare_run(config,rows,source,target,command='cached_probe')
write_json(out/'model_audit.json',{'source':source.audit(),'target':target.audit(),'identity':identity})
report=integrity_check(source,target,rows[0],config);report['identity']=identity
write_json(out/'patch_integrity.json',report)
if not report['passed']:raise RuntimeError('Integrity gate failed')
print(run_probe(config,rows,source,target,out,identity,resume=a.resume),flush=True)
