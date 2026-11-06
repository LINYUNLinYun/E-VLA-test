"""Pinned INT-ACT Bridge/WidowX native contract, CPU preparation only by default.

No Google embodiment claim, LIBERO draft substitution, tokenizer reconstruction,
download, or import-time model constructor. Gated tokenizer access is an explicit
stop condition, not permission to use a mirror or fabricate its configuration.
"""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
INTACT=Path("/policies/INT-ACT")
LEROBOT=INTACT/"third_party/lerobot"
INTACT_REV="66ae3ed4719c68cb2f3f22868015a7a171b04e1f"
LEROBOT_REV="35f6e02315dcb3fa25f3a740478265c6c793a95e"
CHECKPOINT=ROOT/"shared/checkpoints/e4_supplement/juexzz/INTACT-pi0-finetune-bridge"
STATS=INTACT/"config/dataset/bridge_statistics.json"
DEFAULT_ROT=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    for directory,expected in ((INTACT,INTACT_REV),(LEROBOT,LEROBOT_REV)):
        actual=subprocess.check_output(["git","-C",str(directory),"rev-parse","HEAD"],text=True).strip()
        if actual!=expected: raise ValueError("Pinned native source revision mismatch: "+str(directory))


def official_adapter(*,use_bf16=True):
    """Actual released CPU preprocessing adapter, no model construction."""
    verify_sources()
    sys.path.insert(0,str(INTACT))
    from src.experiments.env_adapters.simpler import BridgeSimplerAdapter
    config=SimpleNamespace(env=SimpleNamespace(image_size=(224,224),
        action_normalization_type="bound",state_normalization_type="bound",dataset_statistics_path=str(STATS)),
        use_bf16=use_bf16,seed=7)
    return BridgeSimplerAdapter(config)


def bridge_proprio(eef_xyz_quaternion_wxyz_openness):
    """Independent native formula: base-frame XYZ and relative-to-topdown Euler."""
    pose=np.asarray(eef_xyz_quaternion_wxyz_openness,np.float64)
    if pose.shape!=(8,) or not np.isfinite(pose).all(): raise ValueError("Native Bridge eef pose must be finite8")
    q=pose[3:7]
    if not np.isclose(np.linalg.norm(q),1,atol=1e-6,rtol=0): raise ValueError("Explicit native unit wxyz quaternion required")
    matrix=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    return np.r_[pose[:3],Rotation.from_matrix(matrix@DEFAULT_ROT.T).as_euler("xyz"),pose[7]]


def bound_normalize(state,stats):
    low,high=(np.asarray(stats["proprio"][key],np.float64) for key in ("p01","p99"))
    return np.clip(2*(np.asarray(state)-low)/(high-low+1e-8)-1,-1,1)


def decode_bridge_actions(model_actions,stats):
    """Independent native physical decoder: first6 quantiles, Euler→rotvec, gripper."""
    actions=np.asarray(model_actions,np.float64)
    if actions.ndim!=2 or actions.shape[1]!=7 or not np.isfinite(actions).all():
        raise ValueError("Native Bridge model actions require finite Nx7")
    low,high=(np.asarray(stats["action"][key],np.float64)[:6] for key in ("p01","p99"))
    physical=(actions[:,:6]+1)/2*(high-low)+low
    # Official denormalize_bound does not clip model outputs. Preserve this.
    return np.c_[physical[:,:3],Rotation.from_euler("xyz",physical[:,3:6]).as_rotvec(),
        2*(actions[:,6]>.5)-1]


def cpu_contract():
    verify_sources()
    from .policy_native import native_checkpoint_header
    config=json.loads((CHECKPOINT/"config.json").read_text())
    if (config["chunk_size"],config["n_action_steps"],config["num_steps"])!=(4,4,10):
        raise ValueError("Released native Bridge H4/R4/D10 contract mismatch")
    if set(config["normalization_mapping"].values())!={"IDENTITY"}:
        raise ValueError("Bridge external normalizer must not be applied twice")
    paths=[Path(__file__),STATS,INTACT/"src/experiments/env_adapters/base.py",
        INTACT/"src/experiments/env_adapters/simpler.py",INTACT/"src/experiments/policies/policy_wrapper.py",
        INTACT/"src/utils/geometry.py",INTACT/"src/utils/pipeline.py",
        LEROBOT/"lerobot/common/policies/pi0/modeling_pi0.py",
        LEROBOT/"lerobot/common/policies/pi0/paligemma_with_expert.py",
        LEROBOT/"lerobot/common/policies/pi0/configuration_pi0.py"]
    return dict(status="cpu_contract_complete_runtime_blocked",model_constructed=False,gpu_used=False,
        H=4,R=4,D=10,embodiment="Bridge/WidowX, not native Google",normalization="external p01/p99bound; state clip; no outputclip; model IDENTITY",
        checkpoint_header=native_checkpoint_header(CHECKPOINT/"model.safetensors",key_prefix="model."),
        config_sha256=sha(CHECKPOINT/"config.json"),normalizer_sha256=sha(STATS),
        source_revisions={str(INTACT):INTACT_REV,str(LEROBOT):LEROBOT_REV},source_sha256={str(p):sha(p) for p in paths},
        tokenizer=dict(required="Normally authorized local google/paligemma-3b-pt-224 HF directory",
            gate="Official metadata returned401; stop, no mirrors/credentials/reconstructed substitute",
            max_length=72,newline="native prepare_language appends newline",padding="right/max_length"),
        environment_required=dict(python="3.10.12",transformers="4.48.3",torch="2.6.0"),
        runtime_status="Not validated: strict all-key model load, native tokenizer parity, actual model outputs, Full/V14 GPU gates remain required")


@contextmanager
def authorized_local_tokenizer(directory):
    if directory is None: raise PermissionError("AUTHORIZED_NATIVE_HF_TOKENIZER_DIRECTORY_REQUIRED")
    directory=Path(directory)
    if not directory.is_dir() or not (directory/"tokenizer_config.json").is_file():
        raise PermissionError("Complete normally authorized native tokenizer directory required")
    from transformers import AutoTokenizer
    original=AutoTokenizer.from_pretrained
    def local(cls,name,*args,**kwargs):
        if str(name)!="google/paligemma-3b-pt-224": raise RuntimeError("Unexpected tokenizer URI; downloads forbidden")
        return original(str(directory),local_files_only=True)
    AutoTokenizer.from_pretrained=classmethod(local)
    try: yield
    finally: AutoTokenizer.from_pretrained=original


class NativeBridgePolicy:
    """Future native loader: refuses construction before normal tokenizer authority."""
    def __init__(self,*,tokenizer_directory=None,launch=False):
        if tokenizer_directory is None:
            raise PermissionError("AUTHORIZED_NATIVE_HF_TOKENIZER_DIRECTORY_REQUIRED; no substitution")
        if not launch: raise RuntimeError("EXPLICIT_GPU_LAUNCH_GRANT_REQUIRED")
        verify_sources()
        if importlib.metadata.version("transformers")!="4.48.3":
            raise RuntimeError("Use isolated project .venv with native pinned transformers4.48.3; do not mutate oldenv")
        from preparations.idea1_v3_2.residual import require_idle_gpu
        self.gpu_preflight=require_idle_gpu()
        sys.path.insert(0,str(LEROBOT))
        from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
        import safetensors.torch
        config=PI0Config.from_pretrained(str(CHECKPOINT),local_files_only=True)
        with authorized_local_tokenizer(tokenizer_directory):
            self.model=PI0Policy(config)
        safetensors.torch.load_model(self.model,str(CHECKPOINT/"model.safetensors"),strict=True,device="cpu")
        self.model=self.model.to("cuda").eval()
        self.adapter=official_adapter()
        self.runtime_certified=False


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    args=parser.parse_args();path=Path(args.output)
    if path.exists(): raise FileExistsError(path)
    result=cpu_contract();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream: json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(path),status=result["status"])))


if __name__=="__main__": main()
