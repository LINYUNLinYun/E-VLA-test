"""Frozen matched-cost decision reader. Never falls back to legacy costs."""
import hashlib
import json
from .analyze import OUT


def receipt(scope='final'):
    path=OUT/('cost_gate_pi0_screen.json' if scope=='pi0_screen' else 'cost_gate_final.json')
    if not path.exists():return None
    data=json.loads(path.read_text())
    if data.get('status')!='frozen':return None
    for entry in data.get('hashed_inputs',[]):
        from pathlib import Path
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Matched cost input changed: '+entry['path'])
    return data


def entry(cid,data):
    return data.get('conditions',{}).get(cid,{})


def eligible(cid,data):
    row=entry(cid,data)
    return row.get('verification')=='confirmed' and row.get('eligible_cost_gate') is True


def rank(cid,data):
    row=entry(cid,data)
    if row.get('verification')!='confirmed':return float('inf')
    value=row.get('matched_cost_ratio')
    return float(value) if value is not None else float('inf')
