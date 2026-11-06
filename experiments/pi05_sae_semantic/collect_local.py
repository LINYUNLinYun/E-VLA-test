"""Collect bounded native pi0.5 readouts with the unchanged released SAE encoder.

Each observation is saved independently. SIGTERM requests a stop at the next
observation boundary; completed observations can be resumed on the same device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from .check_public_sae import load_official_loader
from .native_probe import NativeProbe


def bounded_observations(rows, limit):
    if limit >= len(rows):
        return rows
    # Expansion must cover every task/init, not the first few long trajectories.
    episodes = {}
    for row in rows:
        episodes.setdefault((row['suite'],row['task'],row['init']),[]).append(row)
    ordered=[]
    for group in episodes.values():
        remaining=list(range(len(group)));chosen=[]
        while remaining:
            index=remaining[0] if not chosen else max(remaining,key=lambda k:min(abs(k-j) for j in chosen))
            chosen.append(index);remaining.remove(index)
        ordered.append([group[i] for i in chosen])
    return [group[i] for i in range(max(map(len,ordered))) for group in ordered if i<len(group)][:limit]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observations", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--preflight", required=True)
    p.add_argument("--checkpoint", default="shared/checkpoints/openpi/pi05_libero_pytorch_idea2")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--wait-for-preflight-pid", type=int)
    p.add_argument("--public-tail-check")
    p.add_argument("--thread-check", help="Saved numerical check for a CPU setting other than the original 8 threads")
    p.add_argument("--selection-output")
    p.add_argument("--matrix-output")
    p.add_argument("--matrix-observations",type=int,default=12)
    p.add_argument("--reuse-collection", help="Reuse completed compatible pilot captures when expanding to a new output")
    args = p.parse_args()
    output, assets = Path(args.output).resolve(), Path(args.assets).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    preflight_path = Path(args.preflight) / "native_preflight.json"
    while not preflight_path.exists():
        if not args.wait_for_preflight_pid:
            raise FileNotFoundError("Native no-op/split-tail check must pass before collection")
        try:
            os.kill(args.wait_for_preflight_pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError("Preflight process exited without a receipt; inspect it before restarting") from exc
        if time.time() - started > 1800:
            raise TimeoutError("Bounded preflight wait expired")
        time.sleep(5)
    preflight = json.loads(preflight_path.read_text())
    if not preflight["noop_equal"] or not preflight["tail_equal"]:
        raise ValueError("Preflight did not establish native and split-tail equality")
    if str(Path(args.checkpoint).resolve()) != preflight["checkpoint"] or args.device != preflight["device"]:
        raise ValueError("Collection device/checkpoint must match the validated native preflight")
    if args.threads!=8:
        if not args.thread_check:
            raise ValueError("Validate the alternative CPU threading setting against saved native tensors")
        thread_check=json.loads(Path(args.thread_check).read_text())
        if not thread_check['passed'] or thread_check['threads']!=args.threads or thread_check['checkpoint']!=preflight['checkpoint']:
            raise ValueError("CPU thread check does not match this collection")
    def postprocess():
        if not args.selection_output:
            return
        status=dict(state='semantic_selection_running',updated_unix=time.time())
        (output/'postprocessing_status.json').write_text(json.dumps(status,indent=2))
        subprocess.run([sys.executable,'-u','-m','experiments.pi05_sae_semantic.select_sae_concepts',
                        '--collection',str(output),'--assets',str(assets),'--output',args.selection_output],check=True)
        if args.matrix_output:
            status.update(state='initial_module_matrix_running',updated_unix=time.time())
            (output/'postprocessing_status.json').write_text(json.dumps(status,indent=2))
            subprocess.run([sys.executable,'-u','-m','experiments.pi05_sae_semantic.map_concept_ffn',
                '--collection',str(output),'--selection',args.selection_output,'--assets',str(assets),
                '--output',args.matrix_output,'--checkpoint',preflight['checkpoint'],'--device',args.device,
                '--threads',str(args.threads),'--max-observations',str(args.matrix_observations)],check=True)
        status.update(state='postprocessing_complete',updated_unix=time.time())
        (output/'postprocessing_status.json').write_text(json.dumps(status,indent=2))
    rows = bounded_observations([json.loads(line) for line in Path(args.observations).read_text().splitlines() if line.strip()],args.limit)
    if any(row["split"] == "heldout" for row in rows):
        raise ValueError("Heldout observations cannot enter rule discovery/calibration")
    # Interleave tasks/splits so early partial data covers the matrix rather than one trajectory.
    groups = {}
    for row in rows:
        groups.setdefault((row["suite"], row["task"]), []).append(row)
    rows = [group[i] for i in range(max(map(len, groups.values()))) for group in groups.values() if i < len(group)]
    concepts = json.loads((assets / "concepts/concept_features_pi05.json").read_text())
    feature_ids = {layer: sorted({f for entries in concepts[f"paligemma_layer_{layer}"].values()
                                  for entry in entries.values() for f in entry["feature_indices"]}) for layer in (5, 16)}
    asset_manifest = json.loads((assets / "asset_manifest.json").read_text())
    config = dict(schema=1, checkpoint=str(Path(args.checkpoint).resolve()), device=args.device,
                  public_revision=asset_manifest["revision"], feature_ids=feature_ids,
                  sampling="Native instruction span, up to 32 text tokens; valid camera tokens fill a 64-token budget",
                  observations=[r["id"] for r in rows], preflight=str(preflight_path.resolve()))
    config_path = output / "collection_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError("Collection configuration changed; use a separate output directory")
    config_path.write_text(json.dumps(config, indent=2))
    if args.reuse_collection:
        previous=Path(args.reuse_collection).resolve()
        prior=json.loads((previous/'collection_config.json').read_text())
        for key in ('schema','checkpoint','device','public_revision','feature_ids','sampling'):
            if json.loads(json.dumps(config[key]))!=prior[key]:
                raise ValueError(f"Pilot reuse configuration mismatch: {key}")
        reused=[]
        for row in rows:
            source=previous/'observations'/row['id'];target=output/'observations'/row['id']
            if target.exists() or not (source/'complete.json').exists():
                continue
            completed=json.loads((source/'complete.json').read_text())
            if completed['observation_path']!=row['observation_path'] or (row.get('input_sha256') and completed['input_sha256']!=row['input_sha256']):
                raise ValueError('Pilot observation identity changed')
            target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source,target_is_directory=True)
            reused.append(row['id'])
        (output/'reused_pilot_observations.json').write_text(json.dumps(dict(source=str(previous),ids=reused),indent=2))
    stopped = [False]
    def stop_at_boundary(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, stop_at_boundary)
    signal.signal(signal.SIGINT, stop_at_boundary)
    todo = [r for r in rows if not (output / "observations" / r["id"] / "complete.json").exists()]
    if not todo:
        print(json.dumps(dict(event="already_complete", observations=len(rows))), flush=True)
        postprocess()
        return
    print(json.dumps(dict(event="loading_native", device=args.device, remaining=len(todo))), flush=True)
    probe = NativeProbe(Path(args.checkpoint).resolve(), args.device, args.threads)
    if args.public_tail_check and not (output / "public_tail_check.json").exists():
        from .public_tail_check import check_public_tail
        check_public_tail(probe, args.public_tail_check, preflight, output / "public_tail_check.json")
    loader = load_official_loader(assets)
    saes = {layer: loader.load_sae(assets / f"saes/per_token/paligemma/sae_layer{layer}.safetensors", device="cpu")
            for layer in (5, 16)}
    print(json.dumps(dict(event="loaded", seconds=probe.load_seconds)), flush=True)
    with (output / "runtime_history.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(pid=os.getpid(),device=args.device,threads=args.threads,
                                    started_unix=time.time(),remaining_observations=len(todo)))+"\n")
    for row in todo:
        if stopped[0]:
            break
        folder = output / "observations" / row["id"]
        folder.mkdir(parents=True, exist_ok=True)
        path = Path(row["observation_path"])
        identity = hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get("input_sha256") and row["input_sha256"] != identity:
            raise ValueError(f"Input changed since manifest: {path}")
        begin = time.perf_counter()
        print(json.dumps(dict(event="observation_started", id=row["id"], split=row["split"])), flush=True)
        embedded, attention, positions, layout = probe.prepare(path)
        with torch.inference_mode():
            _, captures = probe.full(embedded, attention, positions, layout["selected_positions"])
        readouts, diagnostics = {}, {}
        for layer in (5, 16):
            sae = saes[layer]
            x = captures[layer].reshape(-1, captures[layer].shape[-1]).float()
            with torch.inference_mode():
                z = sae.encode(x)
                recon = sae.decode(z)
            ids = feature_ids[layer]
            masks = dict(all=list(range(len(x))),
                         instruction=[i for i, label in enumerate(layout["selected_kinds"]) if label == "instruction"],
                         image=[i for i, label in enumerate(layout["selected_kinds"]) if label.startswith("camera_")])
            for kind, indexes in masks.items():
                if indexes:
                    readouts[f"a{layer}_{kind}"] = z[indexes][:, ids].max(0).values.numpy()
            readouts[f"feature_ids{layer}"] = np.asarray(ids, dtype=np.int64)
            diagnostics[str(layer)] = dict(normalized_mse=float((((recon-x) / sae.std) ** 2).mean()),
                                            mean_nonzero_features=float((z != 0).sum(-1).float().mean()))
        np.savez_compressed(folder / "readouts.npz", **readouts)
        torch.save(dict(hidden5=captures["full_L5"], attention=attention.cpu(), positions=positions.cpu(),
                        selected=layout["selected_positions"], kinds=layout["selected_kinds"],
                        L5=captures[5], L16=captures[16]), folder / "prefix_tensors.pt")
        receipt = {**row, "input_sha256": identity, "layout": layout, "diagnostics": diagnostics,
                   "seconds": time.perf_counter()-begin, "device": args.device, "threads": args.threads, "schema": 1}
        temporary = folder / "complete.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2))
        temporary.replace(folder / "complete.json")
        with (output / "completed_observations.jsonl").open("a") as stream:
            stream.write(json.dumps(receipt) + "\n")
        done = sum((output / "observations" / r["id"] / "complete.json").exists() for r in rows)
        status = dict(pid=os.getpid(), updated_unix=time.time(), completed=done, expected=len(rows), state="running",
                      last_id=row["id"], last_seconds=receipt["seconds"], stopped=stopped[0])
        (output / "collection_status.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(dict(event="observation_complete", **status)), flush=True)
    complete = all((output / "observations" / r["id"] / "complete.json").exists() for r in rows)
    (output / "collection_status.json").write_text(json.dumps(dict(pid=os.getpid(), complete=complete,
        expected=len(rows), completed=sum((output / "observations" / r["id"] / "complete.json").exists() for r in rows),
        state="complete" if complete else "stopped", stopped=stopped[0], updated_unix=time.time()), indent=2))
    print(json.dumps(dict(event="collection_complete" if complete else "collection_stopped", expected=len(rows))), flush=True)
    if complete:
        postprocess()


if __name__ == "__main__":
    main()
