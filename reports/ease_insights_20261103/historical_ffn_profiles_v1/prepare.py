"""Metadata-first freeze of forty historical first-Full FFN score vectors only."""
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
ROOT=Path('/root/nfs/embodied/benchmark');REPORT=Path(__file__).resolve().parent
SOURCE=ROOT/'results/idea1_v3_4_E2_module_skip';MEMBER='module_score_0.npy'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def need(v,msg):
    if not v:raise ValueError(msg)

def main():
    if (REPORT/'inventory.json').exists():raise ValueError('Inventory exists; do not replace frozen exposure scope')
    m=json.loads((SOURCE/'manifest.json').read_text());cases=m['small_cases'];rows=[]
    need(len(cases)==40 and {r['task'] for r in cases}=={1,3,5,7,9} and {r['init'] for r in cases}=={19,20},'Unexpected old scope')
    for case in cases:
        p=SOURCE/'B6_F-R+'/ 'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
        identity=json.loads((p/'identity.json').read_text());complete=json.loads((p/'complete.json').read_text())
        with (p/'infer.jsonl').open() as f:first=json.loads(f.readline())
        cfg=identity['config'];route=first['route'];protocol=identity['protocol']
        need(identity['case']==case and identity['manifest_hash']==m['manifest_hash']==complete['manifest_hash'] and complete['valid'],'Old record identity/technical completion differs')
        need(cfg['id']=='B6_F-R+' and cfg['family']=='ffn' and cfg['H']==50 and cfg['R']==5 and cfg['period']==5 and cfg['scoring']=='raw','Old profile condition differs')
        need(first['round']==first['step']==0 and route['path']=='full' and route['D']==10 and route['skip']==[] and route['scoring_mode']=='original' and route['jacobian_model_steps']==10 and route['jacobian_output_steps']==[1,2,3,4,5],'Not an initial unpruned native Full profile')
        need(protocol['checkpoint_sha256']=='c423a508742f8461fa17d8fb42951837b4419804842ed27a4890a9903042f891','Old model differs')
        with zipfile.ZipFile(p/'arrays.npz') as z:
            raw=z.read(MEMBER);stream=io.BytesIO(raw);version=np.lib.format.read_magic(stream)
            shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
            need(shape==(17,) and not fortran and dtype.kind=='f','Need one 17-layer floating score vector')
        rows.append(dict(case=case,arrays_path=str(p/'arrays.npz'),selected_member=MEMBER,selected_member_sha256=hashlib.sha256(raw).hexdigest(),
            identity_path=str(p/'identity.json'),identity_sha256=sha(p/'identity.json'),first_input_hash=first['input_hash'],first_seed=first['seed'],
            score_shape=list(shape),score_dtype=str(dtype),checkpoint_sha256=protocol['checkpoint_sha256']))
    inventory=dict(time=time.time(),source_manifest=dict(path=str(SOURCE/'manifest.json'),sha256=sha(SOURCE/'manifest.json'),canonical_hash=m['manifest_hash']),
        cases=rows,records=40,selected_member=MEMBER,success_files_read=False,raw_action_arrays_read=False,score_values_decoded=False,score_statistics_computed=False)
    save(REPORT/'inventory.json',inventory)
    protocol=dict(time=time.time(),status='prospective_before_decoding_score_values',inventory=dict(path=str(REPORT/'inventory.json'),sha256=sha(REPORT/'inventory.json')),
        source_scope='All40small40 B6_F-R+ first native Fulls;four suites,five fixed tasks each,init19/20. No later frame,closed-loop outcome,other condition or heldout selection.',
        sources={str(REPORT/n):sha(REPORT/n) for n in ['prepare.py','analyze.py']},
        transformation='Require finite nonnegative17scores with positive sum; normalize each case to sum1,average two initials inside task,then equal tasks and suites.',
        masks=dict(component='ffn',layers=list(range(1,18)),skip_count=6,adjacency='no adjacent skipped layers',enumeration='lexical all6-subsets of17'),
        algorithm='Compute pooled minimal mask score;retain masks with pooled score<=1.25*minimum. For each suite choose its minimum-score retained mask,ties lexical. Consider six suite pairs and their two chosen masks.',
        proxy_eligibility='Masks differ;each target suite opposite relative score reduction>=.10;at least4/5 positive tasks in each direction;every task-LOO mean difference>0.',
        ranking='Descending minimum relative gap,descending sum relative gaps,ascending suite pair and masks.',
        diagnostic='For frozen two target suites,repeat mask recipe leaving each task ID out across all4suites;test on that task. Require>=4/5 positive heldout task directions in both suites to propose,otherwise no_proxy_candidate.',
        model_quality_claim=False,task_category_complementarity_claim=False,discovery_pvalues_computed=False,new_GPU_launch_authorized_by_this_file=False,
        limits='Historical D10 Jacobian action sensitivity is a proxy,not compensated FastD2 loss or closed-loop success. Five task profiles per suite. CurrentC7 remains frozen;only a separately preregistered new GPU study may test a proposed pair if needed.')
    save(REPORT/'protocol.json',protocol)
    print(json.dumps(dict(status='frozen',records=40,score_values_decoded=False,protocol_sha256=sha(REPORT/'protocol.json'))))
if __name__=='__main__':main()
