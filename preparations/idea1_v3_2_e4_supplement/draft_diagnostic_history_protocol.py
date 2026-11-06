"""CPU verification of all200 historical actual head/checkpoint/config records."""
import csv
import hashlib
from collections import Counter
from pathlib import Path
from .draft_diagnostic_audit import ROOT,DEFAULT_OUT,HEADS,read,dump,csvout


def main():
    index=ROOT/'policies/realtime-vla-flash/results/idea1_v3/e4_expand_init7_10_20261019T1454Z/summaries/with_history/trajectory_index.csv'
    manifest=read(ROOT/'results/idea1_v3_2_e4_supplement/libero_plus_20261022T074328Z/manifest.json')
    heads=manifest['checkpoint_metadata']['draft_heads'];base=manifest['checkpoint_metadata']['shared_base'];rows=[]
    for record in csv.DictReader(index.open()):
        path=Path(record['directory'])/'result.json';result=read(path);head=HEADS[record['suite']]
        expected=heads[head]['path'];checkpoint=result['checkpoint'];protocol=result['protocol']
        head_manifest=Path(expected).with_name('manifest.json')
        actual_head_path=checkpoint['draft_path'];head_match=Path(actual_head_path).resolve()==Path(expected).resolve()
        manifest_match=checkpoint['draft_manifest_sha256']==hashlib.sha256(head_manifest.read_bytes()).hexdigest()
        rows.append(dict(suite=record['suite'],head=head,task=int(record['task']),init=int(record['init']),
            success=result['success'],actions=result['actions'],predictions=result['predictions'],
            actual_checkpoint_path=actual_head_path,head_path_matches_new=head_match,head_manifest_matches_current=manifest_match,
            base_sha_matches_new=checkpoint['base_sha256']==base['sha256'],
            config_is_exact_E4=result['config']=={'experiment':'E4'},H=protocol['H'],R=protocol['R'],
            historical_init_vector_hash=result['init_vector_hash'],result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(rows)==200
    csvout(DEFAULT_OUT/'historical_actual_protocol.csv',rows)
    dump(DEFAULT_OUT/'historical_actual_protocol_receipt.json',dict(status='complete',count=len(rows),gpu_used=False,
        mismatched_heads=sum(not r['head_path_matches_new'] for r in rows),
        mismatched_head_manifests=sum(not r['head_manifest_matches_current'] for r in rows),
        mismatched_base_sha=sum(not r['base_sha_matches_new'] for r in rows),
        mismatched_E4_config=sum(not r['config_is_exact_E4'] for r in rows),
        H_R_counts={str(k):v for k,v in Counter((r['H'],r['R']) for r in rows).items()},
        observed_failure_caps={suite:dict(Counter(r['actions'] for r in rows if r['suite']==suite and not r['success'])) for suite in HEADS},
        limitations=['Historical head artifact path and small manifest are recorded; a historical pkl tensor/file SHA is absent, so these checks do not independently prove historical draft tensor bitwise identity.',
                     'with_history is the aggregation directory combining init6 and7..10 results. E4 inference branch uses current vision/language/state only; acknowledgement tracks step/pending bookkeeping.'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(dict(count=len(rows),head_mismatches=sum(not r['head_path_matches_new'] for r in rows),base_mismatches=sum(not r['base_sha_matches_new'] for r in rows)))


if __name__=='__main__':main()
