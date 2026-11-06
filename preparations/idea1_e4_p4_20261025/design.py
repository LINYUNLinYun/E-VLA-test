"""Bounded four-arm P2/P4 Xiaomi refresh study, fixed before new outcomes."""
import hashlib
import json
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_3.manifest import SUITES

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/idea1_e4_p4_20261025'


def main():
    cs=[dict(id=f'p{p}_{m}',experiment='E4',model='xiaomi',refresh_period=p,mode=m,tau=.8,draft_steps=2,full_steps=2) for p,m in [(2,'periodic'),(4,'periodic'),(2,'verified'),(4,'verified')]]
    cases=[dict(suite=s,task=t,init=8) for t in [0,5,1,2,3,4,6,7,8,9] for s in SUITES]
    base=[0,1,3,2];schedule=[dict(case=c,condition_order=[cs[(v+i)%4]['id'] for v in base]) for i,c in enumerate(cases)]
    plan=ROOT/'docs/experiments/idea1/idea1_e4_p4_20261025_plan.md'
    m=dict(status='frozen_scientific_design',conditions=cs,cases=cases,schedule=schedule,profile_case_indices=list(range(8)),budget_valid=160,technical_attempt_limit_per_episode=2,
        counterbalance='Four-arm Williams base[0,1,3,2],rotate case-index mod4; each arm each position10 times; same case all4 arms same GPU3/model service.',
        periodic='P consecutive draft proposals allowed, then Full. Every actual Full (startup/periodic/spatial zero rejection) resets counter. Partial positive acceptance still one draft round; no gripper protection. Actual draft_rounds stored unchanged.',
        numerical='Frozen Xiaomi skip12 indices4,5,6,13,14,15,22,23,24,31,32,33; H10/R10/D2 for Full and draft; verification q=.10/.05 with native forward time1-q,strict spatial prefix tau=.8. Only period comparator changes in copied Policy.',
        warmup='Before every arm same initial observation: FullD2,skip12D2,dual verifier,3 FullD2 canonical; last2 raw+decoded exact. Real startup must match warm canonical and other arms. No Pi metadata/Kalman warmup.',
        costs='Server predict+entire ack wall per actual action primary; predict/ack separate; load/common warm/RPC/environment/reset/IO separate. Shared ack retains inherited filter.observe in all arms.',
        environment='Fresh offscreen environment per arm; officialinit8/seed7/10settling/20Hz/native limits. No full videos; actual EEF/action telemetry.',
        seed='Frozen v3.3 seed(suite,task,init,replan),independent of arm/order.',
        inputs=[dict(path=str(plan),sha256=hashlib.sha256(plan.read_bytes()).hexdigest())])
    m['manifest_hash']=digest(m);target=OUT/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=m:raise ValueError('Immutable design')
    atomic(target,m);print(target)


if __name__=='__main__':main()
