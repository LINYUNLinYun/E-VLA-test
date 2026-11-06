"""Collect actual fresh-serial observations; CPU-only, immutable explicit writes."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
import random

import numpy as np

from .common import ROOT,SUITES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose(groups,count,seed):
    """Round-robin scenes, with quantiles spanning each complete trajectory."""
    rng = random.Random(seed)
    keys = sorted(groups)
    order = keys[:]
    rng.shuffle(order)
    budget = math.ceil(count/len(order))
    queues = {}
    for key in order:
        candidates = groups[key]
        offset = rng.random()
        indexes = list(dict.fromkeys(min(len(candidates)-1,int((i+offset)*len(candidates)/budget)) for i in range(budget)))
        remainder = [index for index in range(len(candidates)) if index not in indexes]
        rng.shuffle(remainder)
        queues[key] = iter([candidates[index] for index in indexes+remainder])
    selected,seen = [],set()
    while len(selected)<count:
        progress = False
        for key in order:
            for item in queues[key]:
                if item["input_hash"] not in seen:
                    selected.append(item)
                    seen.add(item["input_hash"])
                    progress = True
                    break
            if len(selected)==count:
                break
        if not progress:
            raise ValueError(f"Only {len(selected)} unique actual inputs available; need {count}")
    return selected


def collect(source,per_suite=25,seed=3101):
    from preparations.idea1_v2_1.runtime_client import input_digest
    source = Path(source).resolve()
    bank,receipts,missing = [],[],[]
    for suite in SUITES:
        groups = {}
        for task,init in itertools.product((0,5),(0,1)):
            directory = source/suite/f"task{task:02d}"/f"init{init:02d}"
            if not (directory/"complete.json").exists():
                missing.append(str(directory))
                continue
            result = json.loads((directory/"result.json").read_text())
            if not result["valid"] or result["mode"]!="fresh_serial" or result["interval"]!=1:
                raise ValueError("Bank source is not a valid original fresh-serial episode")
            candidates = []
            infer_path = directory/"infer.jsonl"
            infer_sha = sha(infer_path)
            for row in [json.loads(line) for line in infer_path.read_text().splitlines()]:
                path = Path(row["input_path"])
                path = path if path.is_absolute() else ROOT/path
                actual_sha = sha(path)
                if actual_sha!=row["input_file_sha256"]:
                    raise ValueError(f"Source input artifact changed: {path}")
                with np.load(path,allow_pickle=False) as payload:
                    obs = {name:payload[name].copy() for name in payload.files if name not in ("seed","instruction")}
                    instruction,noise_seed = str(payload["instruction"].item()),int(payload["seed"].item())
                identity = input_digest(obs,instruction)
                if identity!=row["input_hash"] or noise_seed!=row["seed"]:
                    raise ValueError("Saved actual observation/noise identity differs from inference record")
                candidates.append(dict(obs=obs,instruction=instruction,suite=suite,seed=noise_seed,input_hash=identity,
                                       source=dict(path=str(path),sha256=actual_sha,infer_path=str(infer_path),infer_sha256=infer_sha,
                                                   task=task,init=init,request=row["request"],step=row["step"],
                                                   episode_requests=result["predictions"])))
            groups[(task,init)] = candidates
        if groups:
            selected = choose(groups,per_suite,seed+SUITES.index(suite)*100003)
            bank.extend(selected)
            receipts.append(dict(suite=suite,selected=len(selected),unique=len({row["input_hash"] for row in selected}),
                                 available_records=sum(len(group) for group in groups.values()),
                                 scene_selection={f"task{task}/init{init}":[row["source"]["request"] for row in selected
                                                   if row["source"]["task"]==task and row["source"]["init"]==init]
                                                  for task,init in sorted(groups)}))
    return bank,dict(source=str(source),per_suite=per_suite,seed=seed,missing_episodes=missing,suites=receipts,
                     algorithm="Round-robin scenes; ceil(25/4)=7 quantile positions across each complete trajectory with seeded fractional offset; deterministic seeded fill for duplicates",
                     policy_inputs="Actual saved original fresh_serial_r1 NPZ, SHA and input digest checked against raw inference records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--per-suite",type=int,default=25)
    parser.add_argument("--seed",type=int,default=3101)
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    bank,receipt = collect(args.source,args.per_suite,args.seed)
    if receipt["missing_episodes"]:
        raise RuntimeError(f"Wait for all16 original fresh episodes: {receipt['missing_episodes']}")
    require_count = args.per_suite*len(SUITES)
    if len(bank)!=require_count or len({row["input_hash"] for row in bank})!=require_count:
        raise ValueError("Distinct actual input budget not satisfied")
    receipt.update(records=len(bank),globally_unique=len({row["input_hash"] for row in bank}),
                   selected=[dict(input_hash=row["input_hash"],suite=row["suite"],source=row["source"]) for row in bank])
    if args.write:
        output = Path(args.output)
        report = output.with_suffix(output.suffix+".receipt.json")
        if output.exists() or report.exists():
            raise FileExistsError("Actual-observation bank and receipt are immutable")
        output.parent.mkdir(parents=True,exist_ok=True)
        raw = pickle.dumps(bank,protocol=5)
        with output.open("xb") as stream:
            stream.write(raw)
        receipt.update(output=str(output.resolve()),bank_sha256=hashlib.sha256(raw).hexdigest())
        with report.open("x") as stream:
            json.dump(receipt,stream,indent=2)
    print(json.dumps(dict(dry_run=not args.write,records=receipt["records"],globally_unique=receipt["globally_unique"],
                          suites=receipt["suites"]),indent=2))


if __name__=="__main__":
    main()
