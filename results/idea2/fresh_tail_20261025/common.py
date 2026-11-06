"""CPU-only configuration and quality metrics shared by fresh-tail drivers."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CONFIGS={f"sm{sms}_tail{tail}":dict(ae_sm_min=sms,fresh_tail_steps=tail)
         for sms in (16,32) for tail in (0,1,2)}
DIMENSIONS=("translation_x","translation_y","translation_z","rotation_0","rotation_1","rotation_2","gripper")


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def delta(reference,candidate):
    a,b=np.asarray(reference,np.float64),np.asarray(candidate,np.float64)
    if a.shape!=(10,7) or b.shape!=(10,7) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Expected finite decoded H10 x 7 action chunks")
    answer={}
    for name,length in (("R5",5),("H10",10)):
        error=np.abs(a[:length]-b[:length])
        answer[name]=dict(mae=float(error.mean()),max_abs=float(error.max()),
            per_dimension={dimension:dict(mae=float(error[:,index].mean()),max_abs=float(error[:,index].max()))
                           for index,dimension in enumerate(DIMENSIONS)})
    return answer


def write_csv(path,rows):
    def flatten(row,prefix=""):
        result={}
        for key,value in row.items():
            if isinstance(value,dict):result.update(flatten(value,prefix+key+"_"))
            else:result[prefix+key]=value
        return result
    rows=[flatten(row) for row in rows]
    with Path(path).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))) if rows else ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(tuple,list)) else v for k,v in row.items()})
