"""Read-only Plus ledger audit; technical failures never enter success denominator."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import time
from .native_envs import digest,write_json
from .evaluation import protocol_identity


def audit(root):
    root=Path(root);protocol=protocol_identity(root/'manifest.json');identity=protocol['manifest_hash']
    selection=protocol['case_selection'];groups={};errors=[]
    for name,key in [('perturbed350','selected'),('original40','original_baselines')]:
        cases=selection[key];expected={(i,mode):case for i,case in enumerate(cases) for mode in ['full_d10','v14_d10',case['matching_draft']]}
        completed={};technical=[];running=[]
        for path in (root/name).glob('case*/*/attempt*/result.json'):
            try:
                value=json.loads(path.read_text());index=int(path.parents[2].name[4:]);mode=path.parents[1].name
                case=expected.get((index,mode))
                if case is None or value.get('manifest_hash')!=identity or value.get('case_hash')!=digest(case) or value.get('mode')!=mode:
                    errors.append(dict(path=str(path),reason='identity_or_case_mismatch'));continue
                row=dict(index=index,mode=mode,status=value['status'],success=value.get('success'),steps=value.get('steps'),wall_s=value.get('wall_s'),path=str(path),category=case['category'],difficulty=case['difficulty_level'])
                if value['status']=='complete':
                    if type(value.get('success')) is not bool:
                        errors.append(dict(path=str(path),reason='complete_without_boolean_oracle'));continue
                    if (index,mode) in completed: errors.append(dict(path=str(path),reason='duplicate_completed_case'));continue
                    completed[(index,mode)]=row
                elif value['status']=='technical_failure': technical.append(row)
                else: running.append(row)
            except Exception as exc: errors.append(dict(path=str(path),reason=repr(exc)))
        by_mode=defaultdict(list);strata=defaultdict(list)
        for row in completed.values():
            mode='matching_draft' if row['mode'].startswith('draft_') else row['mode']
            by_mode[mode].append(row);strata[(row['category'],str(row['difficulty']),mode)].append(row)
        groups[name]=dict(target=len(expected),completed=len(completed),remaining=len(expected)-len(completed),technical_attempts=len(technical),
            running_attempts=len(running),by_mode={mode:dict(completed=len(rows),success=sum(row['success'] for row in rows),native_steps=sum(row['steps'] for row in rows)) for mode,rows in by_mode.items()},
            by_stratum=[dict(category=category,difficulty=difficulty,mode=mode,completed=len(rows),success=sum(row['success'] for row in rows)) for (category,difficulty,mode),rows in sorted(strata.items())],
            technical_failures=technical,running=running)
    return dict(status='audit_complete',audited_unix=time.time(),manifest_hash=identity,identity_errors=errors,groups=groups,
        limitations='Partial fixed-matrix progress only. Technical attempts excluded from oracle success denominator; no causal capacity/robustness conclusion from incomplete counts.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output');a=p.parse_args();result=audit(a.root)
    if a.output: write_json(a.output,result)
    print(json.dumps(dict(identity_errors=len(result['identity_errors']),groups={k:{key:v[key] for key in ('target','completed','remaining','technical_attempts','running_attempts','by_mode')} for k,v in result['groups'].items()})))


if __name__=='__main__': main()
