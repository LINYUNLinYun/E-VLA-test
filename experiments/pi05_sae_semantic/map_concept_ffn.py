"""Finite VLM-tail interventions from cached native L5 states; no AE/rollout.

The L5 trigger alone defines each concept cohort. Full L16 need not trigger the
concept, and ground-truth concept labels do not censor the intervention cohort.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .check_public_sae import load_official_loader
from .native_probe import NativeProbe
from .select_sae_concepts import balanced_weights, weighted_quantile

CANDIDATES = (6, 8, 10, 12, 14, 15)


def calibration_entry_supported(split,mask,statistically_supported,early_calibrated,late_evidence_episodes):
    # L16 is an offline feature-response readout, not a second online semantic
    # classifier. All L5-selected cases remain in the measured cohort.
    return bool(split=='calibration' and len(mask)==2 and statistically_supported
                and early_calibrated and late_evidence_episodes>=2)


def semantic_change(full, skipped, features):
    tau = np.array([f["tau"] for f in features])
    scale = np.array([f["b"] for f in features])
    weights = np.array([f["weight"] for f in features])
    uf, us = (np.asarray(full)-tau)/scale, (np.asarray(skipped)-tau)/scale
    return (float(np.abs(uf-us) @ weights), float(np.abs(np.clip(uf, 0, 1)-np.clip(us, 0, 1)) @ weights))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collection", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default="shared/checkpoints/openpi/pi05_libero_pytorch_idea2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--split", choices=("discovery", "calibration"), default="discovery")
    p.add_argument("--max-observations", type=int, default=32)
    p.add_argument('--evidence-level',choices=('both_layers_calibrated','l5_calibrated'),default='both_layers_calibrated',
                   help='l5_calibrated is exploratory: late features need discovery association, not a second semantic classifier')
    p.add_argument("--mask", action="append", help="Two nonadjacent FFNs, e.g. --mask 6,8; at most three masks")
    args = p.parse_args()
    collection, selection, assets, output = map(lambda x:Path(x).resolve(), (args.collection, args.selection, args.assets, args.output))
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((collection / "collection_config.json").read_text())
    summary = json.loads((selection / "semantic_selection_summary.json").read_text())
    if summary["partial"]:
        raise ValueError("Freeze selection on the completed pilot before fitting an intervention matrix")
    if cfg["device"] != args.device or cfg["checkpoint"] != str(Path(args.checkpoint).resolve()):
        raise ValueError("Collection and intervention backend/checkpoint differ; validate migration explicitly first")
    definitions = json.loads((selection / "concept_features.json").read_text())
    if args.evidence_level=='both_layers_calibrated':
        names=summary['ready_concepts']
    else:
        names=[name for name,item in definitions.items() if item['layers']['5']['supported']
               and item['layers']['16']['features']
               and any(f['discovery_metrics']['positive_tasks']>=2 for f in item['layers']['16']['features'])]
    ready = {name:definitions[name] for name in names}
    token_pool=summary.get('token_pool','all')
    if not ready:
        (output / "matrix_status.json").write_text(json.dumps(dict(state="no_supported_concepts",
            reason="No concept meets the requested evidence level; no model forwards launched",evidence_level=args.evidence_level), indent=2))
        return
    masks = [(m,) for m in CANDIDATES]
    if args.mask:
        masks = [tuple(sorted(int(s) for s in value.split(","))) for value in args.mask]
        if len(masks)>3 or any(len(m)!=2 or len(set(m))!=2 or not set(m)<=set(CANDIDATES) or m[1]-m[0]==1 for m in masks):
            raise ValueError("Combination phase allows at most three masks, each two distinct nonadjacent candidate FFNs")
    if not 0 < args.max_observations <= 128:
        raise ValueError("Use a bounded discovery/calibration sample of at most 128 observations")
    rows = json.loads((selection / "score_observations.json").read_text())
    scores = np.load(selection / "semantic_scores.npz", allow_pickle=False)
    pool = [i for i,row in enumerate(rows) if row["split"]==args.split]
    active_by_index = {i:[name for name,item in ready.items()
        if scores[f"L5_{item['category']}_{item['concept']}_{token_pool}"][i] >= item["layers"]["5"]["theta"]] for i in pool}
    # Round-robin concepts, preferring distinct tasks/episodes. Selection uses
    # only the already frozen L5 trigger, never L16 evidence or intervention outcomes.
    indexes,task_counts,episode_counts = [],defaultdict(int),defaultdict(int)
    while len(indexes)<args.max_observations:
        added=False
        for name in ready:
            eligible=[i for i in pool if i not in indexes and name in active_by_index[i]]
            if not eligible or len(indexes)>=args.max_observations:
                continue
            i=min(eligible,key=lambda i:(task_counts[(rows[i]['suite'],rows[i]['task'])],
                episode_counts[(rows[i]['suite'],rows[i]['task'],rows[i]['init'])],i))
            indexes.append(i);task_counts[(rows[i]['suite'],rows[i]['task'])]+=1
            episode_counts[(rows[i]['suite'],rows[i]['task'],rows[i]['init'])]+=1;added=True
        if not added:
            break
    cases = []
    for i in indexes:
        active = active_by_index[i]
        # An inactive observation cannot enter any D_c; preserve its identity in the manifest.
        cases.append(dict(index=i, active_concepts=active, **rows[i]))
    manifest = dict(phase="combination" if args.mask else "single", split=args.split, masks=masks,
                    checkpoint=cfg["checkpoint"], device=args.device, cases=[dict(id=r["id"], active_concepts=r["active_concepts"]) for r in cases],
                    baseline_source="Cached native full L16 from collection; no repeated SigLIP/L0-L5/AE/flow matching",
                    selection_method="Round-robin frozen L5 concepts, preferring distinct tasks/episodes; no L16/label censoring",
                    source_pool_observations=len(pool), source_active_observations=sum(bool(v) for v in active_by_index.values()),
                    feature_selection=str(selection), concepts=list(ready), max_observations=args.max_observations)
    manifest['token_pool']=token_pool
    manifest['evidence_level']=args.evidence_level
    manifest['feature_selection_sha256']=hashlib.sha256((selection/'concept_features.json').read_bytes()).hexdigest()
    manifest['selected_concepts_snapshot']=ready
    path = output / "intervention_manifest.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("Intervention configuration changed; choose a separate output directory")
    path.write_text(json.dumps(manifest, indent=2))
    work = [(row,mask) for row in cases if row["active_concepts"] for mask in masks]
    pending = [(row,mask) for row,mask in work if not (output/"measurements"/row["id"]/('_'.join(map(str,mask))+'.json')).exists()]
    shared=collection/'ffn_intervention_cache'
    shared.mkdir(parents=True,exist_ok=True)
    shared_config=dict(schema=1,checkpoint=cfg['checkpoint'],device=cfg['device'],public_revision=cfg.get('public_revision'),
                       sampling=cfg.get('sampling'),capture='Full released SAE dictionary maxima for all/image/instruction token pools')
    shared_manifest=shared/'cache_manifest.json'
    if shared_manifest.exists() and json.loads(shared_manifest.read_text())!=shared_config:
        raise ValueError('Shared intervention cache belongs to a different model or sampling protocol')
    shared_manifest.write_text(json.dumps(shared_config,indent=2))
    def forward_cache_path(row,mask):
        return shared/row['id']/('_'.join(map(str,mask))+'.npz')
    if pending:
        need_model=any(not forward_cache_path(row,mask).exists() for row,mask in pending)
        probe=NativeProbe(Path(args.checkpoint).resolve(), args.device, args.threads) if need_model else None
        sae=(load_official_loader(assets).load_sae(assets/"saes/per_token/paligemma/sae_layer16.safetensors", device="cpu")
             if need_model else None)
        readout_file=summary.get('readout_file','readouts.npz')
        if summary.get('candidate_pool')=='full_dictionary':
            with np.load(collection/'observations'/rows[0]['id']/readout_file,allow_pickle=False) as archive:
                columns={int(f):i for i,f in enumerate(archive['feature_ids16'])}
        else:
            columns = {int(f):i for i,f in enumerate(cfg["feature_ids"]["16"])}
        # Each module forward reads all selected concept features at once.
        for row in cases:
            needed = [mask for r,mask in pending if r["id"]==row["id"]]
            if not needed:
                continue
            folder = collection/"observations"/row["id"]
            cache=None
            with np.load(folder/readout_file, allow_pickle=False) as archive:
                baseline = archive[f"a16_{token_pool}"].copy()
            for mask in needed:
                started = time.perf_counter()
                cached_path=forward_cache_path(row,mask)
                cache_hit=cached_path.exists()
                if cache_hit:
                    with np.load(cached_path,allow_pickle=False) as stored:
                        if str(stored['input_sha256'].item())!=row['input_sha256']:
                            raise ValueError('Intervention cache input identity changed')
                        activation=stored[f'a16_{token_pool}'].copy()
                else:
                    if cache is None:
                        cache=torch.load(folder/'prefix_tensors.pt',map_location='cpu',weights_only=True)
                        attention,positions=cache['attention'].to(probe.backend.device),cache['positions'].to(probe.backend.device)
                    hidden=probe.tail(cache['hidden5'],attention,positions,cache['selected'],mask)
                    with torch.inference_mode():
                        z=sae.encode(hidden.reshape(-1,hidden.shape[-1]).float())
                    kinds=cache.get('kinds',['unspecified']*len(z))
                    groups=dict(all=list(range(len(z))),image=[i for i,k in enumerate(kinds) if k.startswith('camera_')],
                                instruction=[i for i,k in enumerate(kinds) if k=='instruction'])
                    values={f'a16_{kind}':z[indexes].max(0).values.numpy() for kind,indexes in groups.items() if indexes}
                    values['input_sha256']=np.asarray(row['input_sha256'])
                    cached_path.parent.mkdir(parents=True,exist_ok=True)
                    temporary_cache=cached_path.with_suffix('.npz.tmp')
                    with temporary_cache.open('wb') as stream:np.savez_compressed(stream,**values)
                    temporary_cache.replace(cached_path)
                    activation=values[f'a16_{token_pool}']
                changes = {}
                for name in row["active_concepts"]:
                    features = ready[name]["layers"]["16"]["features"]
                    ids = [f["feature_id"] for f in features]
                    skipped = activation[ids]
                    full = baseline[[columns[f] for f in ids]]
                    d,e = semantic_change(full, skipped, features)
                    evidence=float(np.clip((full-np.array([f['tau'] for f in features]))/np.array([f['b'] for f in features]),0,1)
                                   @ np.array([f['weight'] for f in features]))
                    changes[name] = dict(D=d,E=e,full_candidate_activation=full.tolist(),skip_candidate_activation=skipped.tolist(),
                                         late_baseline_evidence=evidence)
                target = output/"measurements"/row["id"]
                target.mkdir(parents=True, exist_ok=True)
                receipt = dict(id=row["id"], mask=mask, active_concepts=row["active_concepts"], changes=changes,
                               seconds=time.perf_counter()-started, input_sha256=row["input_sha256"],
                               reused_forward=cache_hit,forward_cache=str(cached_path),token_pool=token_pool)
                temporary = target/('_'.join(map(str,mask))+'.json.tmp')
                temporary.write_text(json.dumps(receipt, indent=2)); temporary.replace(temporary.with_suffix(''))
                print(json.dumps(dict(event="tail_intervention_complete", id=row["id"],mask=mask,seconds=receipt["seconds"])), flush=True)
    matrix = []
    for mask in masks:
        for name in ready:
            cohort = [row for row in cases if name in row["active_concepts"]]
            measurements = [json.loads((output/"measurements"/row["id"]/('_'.join(map(str,mask))+'.json')).read_text())["changes"][name] for row in cohort]
            episodes = {(r["suite"],r["task"],r["init"]) for r in cohort}
            supported = len(cohort)>=2 and len(episodes)>=2
            evidence_episodes={(r['suite'],r['task'],r['init']) for r,m in zip(cohort,measurements) if m['late_baseline_evidence']>0}
            late_calibrated=ready[name]['layers']['16']['supported']
            weights = balanced_weights(cohort) if cohort else None
            d = weighted_quantile(np.array([m["D"] for m in measurements]),weights,.9) if supported else None
            e = weighted_quantile(np.array([m["E"] for m in measurements]),weights,.9) if supported else None
            matrix.append(dict(mask=','.join(map(str,mask)),concept=name,split=args.split,D=d,E=e,supported=supported,
                               observations=len(cohort),episodes=len(episodes),tasks=len({(r["suite"],r["task"]) for r in cohort}),
                               late_semantic_calibrated=late_calibrated,late_baseline_evidence_episodes=len(evidence_episodes),
                               routing_supported=calibration_entry_supported(args.split,mask,supported,
                                   ready[name]['layers']['5']['supported'],len(evidence_episodes))))
    with (output/"concept_ffn_matrix.csv").open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(matrix[0]));writer.writeheader();writer.writerows(matrix)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(ready)
    labels = [','.join(map(str,m)) for m in masks]
    fig,axes = plt.subplots(1,2,figsize=(max(8,len(names)*2.5),max(4,len(masks)*.6)),constrained_layout=True)
    if args.evidence_level=='l5_calibrated':
        fig.suptitle('Exploratory: L5 calibrated; late feature associations from discovery')
    for axis,key,title in zip(axes,("D","E"),("Unclipped feature change Q90","Clipped evidence change Q90")):
        values=np.array([[next(r[key] if r[key] is not None else np.nan for r in matrix
                               if r['mask']==label and r['concept']==name) for name in names] for label in labels])
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dddddd')
        im=axis.imshow(values,aspect='auto',cmap=cmap,vmin=0)
        axis.set_xticks(range(len(names)),names,rotation=25,ha='right')
        axis.set_yticks(range(len(labels)),labels);axis.set_ylabel('Skipped FFN indices (0-based)');axis.set_title(title)
        fig.colorbar(im,ax=axis,shrink=.7)
    fig.savefig(output/'concept_ffn_matrix.png',dpi=180)
    fig.savefig(output/'concept_ffn_matrix.pdf');plt.close(fig)
    receipts=[json.loads((output/'measurements'/row['id']/('_'.join(map(str,mask))+'.json')).read_text()) for row,mask in work]
    new_forwards=sum(not r.get('reused_forward',False) for r in receipts)
    (output/"matrix_status.json").write_text(json.dumps(dict(state="complete", selected_observations=len(cases),
        intervention_forwards=new_forwards,reused_forward_results=len(work)-new_forwards,condition_measurements=len(work),
        baseline_from_cache=len(cases), concepts=list(ready), matrix=matrix),indent=2))
    print(json.dumps(dict(event="matrix_complete",new_forwards=new_forwards,reused_forwards=len(work)-new_forwards,concepts=list(ready))),flush=True)


if __name__ == "__main__":
    main()
