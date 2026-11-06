"""Same-case/same-physical-card closed-loop costs for confirmation cohorts."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from preparations.idea1_v3.io import atomic
from .analyze import OUT,records,summary


def run():
    groups=records();result=[]
    def key(r):return tuple(r['case'][k] for k in ('suite','task','init'))
    def binding(r):
        p=Path(r['_directory'])/'engineering_revision.json';data=json.loads(p.read_text());return data['model_gpu'],data['render_gpu']
    candidate='E4_tau0.8';left={key(r):r for r in groups[candidate]}
    for reference in ['E4_periodic','xiaomi_pure_d2','xiaomi_full_d2']:
        right={key(r):r for r in groups[reference]};paired=defaultdict(list);excluded=[]
        for k in sorted(left.keys()&right.keys()):
            a,b=left[k],right[k];aa,bb=binding(a),binding(b)
            if aa!=bb or aa[0]!=aa[1]:excluded.append(dict(case=list(k),candidate_binding=aa,reference_binding=bb));continue
            paired[str(aa[0])].append((a,b))
        for gpu,pairs in paired.items():
            for cohort in ['all_matched','new220','new_hot_init3plus']:
                own=[(a,b) for a,b in pairs if cohort=='all_matched' or (a['case']['init']>=2 if cohort=='new220' else a['case']['init']>=3)]
                if not own:continue
                a,b=summary([x for x,y in own]),summary([y for x,y in own])
                result.append(dict(candidate=candidate,reference=reference,physical_gpu=int(gpu),cohort=cohort,n=len(own),candidate_policy_ms_per_action=a['ms_per_action'],reference_policy_ms_per_action=b['ms_per_action'],
                    ratio=a['ms_per_action']/b['ms_per_action'],difference_ms_per_action=a['ms_per_action']-b['ms_per_action'],candidate_actions=a['actions'],reference_actions=b['actions'],
                    cases=[dict(case=x['case'],candidate_result_sha256=hashlib.sha256((Path(x['_directory'])/'result.json').read_bytes()).hexdigest(),reference_result_sha256=hashlib.sha256((Path(y['_directory'])/'result.json').read_bytes()).hexdigest()) for x,y in own],
                    excluded_binding_cases=excluded))
    data=dict(time=time.time(),rows=result,scope='Same official initial-case keys and same physical model/render GPU; full original policy wall / each policy actual actions. These are closed-loop cohort observations, not same-input backend request timings. Fixed condition order and cold/runtime history remain limitations. new_hot_init3plus is a sensitivity cohort, not formal quality deletion or promotion gate.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic(OUT/'reports/matched_confirmation_costs.json',data);return data


if __name__=='__main__':run()
