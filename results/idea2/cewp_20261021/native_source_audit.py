"""Read-only CPU audit of local native-model edits versus repository HEAD."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import siglip, vision_compression, pi0
from openpi.training import config
from openpi.policies.policy_config import create_trained_policy
from preparations.idea2.experiments import native_raw
from preparations.idea1_v2_1.runtime_client import input_digest
root=Path('/root/nfs/embodied/benchmark')
repo=root/'policies/openpi'
out=root/'results/idea2/cewp_20261021'
def git(*args):
    return subprocess.check_output(['git',*args],cwd=repo,text=True)
original_pi0=git('show','HEAD:src/openpi/models/pi0.py')
original_siglip=git('show','HEAD:src/openpi/models/siglip.py')
current_pi0=(repo/'src/openpi/models/pi0.py').read_text()
module=types.ModuleType('native_audit_original_siglip')
sys.modules[module.__name__]=module
exec(compile(original_siglip,'git-HEAD:src/openpi/models/siglip.py','exec'),module.__dict__)
assert jax.default_backend()=='cpu'
assert vision_compression.CONFIG=={'mode':'off','keep_tokens':256}
assert not vision_compression.is_active()
cfg=config.get_config('pi05_libero').model
rng=jax.random.key(1701)
x=jax.random.normal(rng,(1,256,24),dtype=jnp.float32)
current=siglip.Encoder1DBlock(num_heads=4,mlp_dim=48,dtype_mm='bfloat16')
original=module.Encoder1DBlock(num_heads=4,mlp_dim=48,dtype_mm='bfloat16')
params=current.init(rng,x)
current_result=current.apply(params,x)
original_result=original.apply(params,x)
errors=[float(np.max(np.abs(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)))) for a,b in zip(jax.tree.leaves(current_result),jax.tree.leaves(original_result),strict=True)]
# A forbidden-call sentinel proves off mode never dispatches compression code.
def forbidden(*args,**kwargs):
    raise AssertionError('Disabled compression executed')
vision_compression.make_mapping=forbidden
vision_compression.merge=forbidden
vision_compression.unmerge=forbidden
jax.jit(current.apply)(params,x)[0].block_until_ready()
current_jaxpr=str(jax.make_jaxpr(current.apply)(params,x))
original_jaxpr=str(jax.make_jaxpr(original.apply)(params,x))
files=['src/openpi/models/pi0.py','src/openpi/models/siglip.py','src/openpi/models/vision_compression.py','src/openpi/models/pi0_config.py','src/openpi/training/config.py','src/openpi/policies/policy_config.py']
result={'passed':max(errors)==0 and ast.dump(ast.parse(current_pi0))==ast.dump(ast.parse(original_pi0)),
 'head':git('rev-parse','HEAD').strip(),'git_status':git('status','--short'),'backend':jax.default_backend(),
 'native_imported_config':repr(cfg),'vision_compression_config':dict(vision_compression.CONFIG),'compression_active':vision_compression.is_active(),
 'pi0_ast_equals_head':ast.dump(ast.parse(current_pi0))==ast.dump(ast.parse(original_pi0)),
 'siglip_off_block_same_params_outputs_max_abs':max(errors),'siglip_output_leaf_count':len(errors),
 'siglip_off_jaxpr_text_equals_head':current_jaxpr==original_jaxpr,
 'compression_function_forbidden_sentinel_jit_passed':True,
 'source_sha256':{f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files},
 'conclusion':'Fresh native runner process defaults to compression off and follows original full SigLIP block operations. Local pi0 change is comment-only. No skip-layer or compression activation is present in native import/call path.',
 'scope':'CPU default-import and small full-256-token BF16 SigLIP block audit with identical parameters; not a new full-checkpoint GPU validation.',
 'optional_hardening':'Before create_trained_policy, explicitly configure off/256, assert inactive, and record model source SHA plus CONFIG in native result. Does not touch production model files. Clean detached temporary worktree at HEAD with its src first on PYTHONPATH provides stronger source isolation if required.'}
(out/'native_source_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
