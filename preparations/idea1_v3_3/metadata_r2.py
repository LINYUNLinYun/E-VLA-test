"""Metadata-only successor; original frozen scientific Policy stays untouched."""
import hashlib
from pathlib import Path
import time
import numpy as np
from .policy import Policy as OriginalPolicy
from preparations.idea1_v3.policy import verification_seed


class Policy(OriginalPolicy):
    def __init__(self,backend):
        super().__init__(backend)
        self.protocol.update(metadata_revision='v33-noise-r2',metadata_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            noise_hash_semantics='last_model_noise_hash is historical; draft/full/verify fields identify only work actually performed this request')
        receipt=Path(__file__).resolve().parents[2]/'results/idea1_v3_3/metadata_revision_r2.json'
        if receipt.exists(): self.protocol['metadata_execution_receipt_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()

    def predict(self,obs,instruction,suite,seed):
        result=super().predict(obs,instruction,suite,seed)
        tick=time.perf_counter()
        route=result['route']; path=route['path']; rejected=route.get('rejected_path')
        model_noise=hashlib.sha256(np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32).tobytes()).hexdigest()
        result['last_model_noise_hash']=result.pop('noise_hash',None)
        result['full_noise_hash']=model_noise if path=='full' else None
        is_draft=path in ('pure','v14_verified','periodic_draft') or rejected=='v14_verified'
        result['draft_noise_hash']=model_noise if is_draft else None
        verified='distance' in route
        result['verify_noise_hash']=hashlib.sha256(np.random.default_rng(verification_seed(seed)).standard_normal((50,32),dtype=np.float32).tobytes()).hexdigest() if verified else None
        result['metadata_revision']='v33-noise-r2'
        result['timing']['metadata_wall_ms']=(time.perf_counter()-tick)*1000
        result['timing']['policy_wall_ms']+=result['timing']['metadata_wall_ms']
        return result
