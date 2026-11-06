"""Recover full-dictionary observation summaries from already saved layer tokens.

No VLA forward or SAE training is performed. Only N x dictionary summaries
are saved; the dense N x token x dictionary codes are never retained.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .check_public_sae import load_official_loader


def ensure_dictionary_readouts(collection, folders, assets):
    cfg=json.loads((collection/'collection_config.json').read_text())
    missing=[p for p in folders if not (p/'readouts_all_features.npz').exists()]
    if not missing:
        with np.load(folders[0]/'readouts_all_features.npz',allow_pickle=False) as data:
            return {str(layer):data[f'feature_ids{layer}'].tolist() for layer in (5,16)}
    loader=load_official_loader(assets)
    saes={layer:loader.load_sae(assets/f'saes/per_token/paligemma/sae_layer{layer}.safetensors',device='cpu') for layer in (5,16)}
    ids={str(layer):list(range(sae.encoder.out_features)) for layer,sae in saes.items()}
    torch.set_float32_matmul_precision('high')
    for folder in missing:
        receipt=json.loads((folder/'complete.json').read_text())
        torch.set_num_threads(receipt.get('threads',8))
        cache=torch.load(folder/'prefix_tensors.pt',map_location='cpu',weights_only=True)
        groups=dict(all=list(range(len(cache['kinds']))),
                    image=[i for i,label in enumerate(cache['kinds']) if label.startswith('camera_')],
                    instruction=[i for i,label in enumerate(cache['kinds']) if label=='instruction'])
        arrays={}
        with np.load(folder/'readouts.npz',allow_pickle=False) as published:
            for layer,sae in saes.items():
                x=cache[f'L{layer}'].reshape(-1,cache[f'L{layer}'].shape[-1]).float()
                with torch.inference_mode():z=sae.encode(x)
                for kind,indexes in groups.items():
                    value=z[indexes].max(0).values.numpy()
                    key=f'a{layer}_{kind}'
                    if not np.array_equal(value[cfg['feature_ids'][str(layer)]],published[key]):
                        raise ValueError('Full-dictionary summary disagrees with the existing candidate readout; inspect numerical settings')
                    arrays[key]=value
                arrays[f'feature_ids{layer}']=np.arange(sae.encoder.out_features,dtype=np.int64)
        temporary=folder/'readouts_all_features.npz.tmp'
        with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
        temporary.replace(folder/'readouts_all_features.npz')
        print(json.dumps(dict(event='dictionary_summary_saved',id=receipt['id'],VLM_forwards=0,agrees_with_existing_candidate_columns=True)),flush=True)
    metadata=dict(public_revision=cfg['public_revision'],checkpoint=cfg['checkpoint'],device=cfg['device'],
                  source='Cached valid L5/L16 token samples, unchanged official SAE loader/encoder',
                  candidate_pool='All released SAE dictionary coordinates; association learned on discovery observations only',
                  VLM_forwards=0,SAE_training_steps=0)
    (collection/'dictionary_readout_metadata.json').write_text(json.dumps(metadata,indent=2))
    return ids
