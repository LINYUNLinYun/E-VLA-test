"""Static actual native BDDL instruction corpus; no renderer/model."""
import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path
from bddl.parsing import scan_tokens
from .native_envs import ROOT,sha,write_json
from .libero_plus_env import PLUS,ORIGINAL


def language(path):
    # Execute verbatim official parser function only; no env/package import.
    source=PLUS/'libero/libero/envs/bddl_utils.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='get_problem_info')
    namespace=dict(scan_tokens=scan_tokens)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['get_problem_info'](str(path))['language_instruction']


def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--mapping',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    selection=json.loads(Path(a.selection).read_text());mapping={(row['suite'],row['task_name']):row for row in csv.DictReader(Path(a.mapping).open())}
    rows=[];counts=Counter()
    for case in selection['selected']+selection['original_baselines']:
        mother=ORIGINAL/'libero/libero/bddl_files'/case['mother_suite']/(case['mother_task']+'.bddl')
        path=Path(mapping[(case['mother_suite'],case['name'])]['effective_bddl']) if case['environment']=='libero_plus' else mother
        actual=language(path);reference=language(mother);same=actual==reference
        counts[(case['category'],'equal' if same else 'different')]+=1
        rows.append(dict(environment=case['environment'],name=case['name'],mother_suite=case['mother_suite'],category=case['category'],
            instruction=actual,mother_instruction=reference,equal_to_original_mother=same,effective_bddl=str(path),bddl_sha256=sha(path)))
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    result=dict(status='complete',selection_sha256=selection['selection_sha256'],cases=rows,count=len(rows),
        category_comparison={category:{key:counts[(category,key)] for key in ('equal','different')} for category in sorted(set(row['category'] for row in rows))},
        parser='Verbatim official get_problem_info + bddl.scan_tokens (lowercasing is native parser behavior)',
        source_sha256={str(Path(__file__)):sha(__file__),str(PLUS/'libero/libero/envs/bddl_utils.py'):sha(PLUS/'libero/libero/envs/bddl_utils.py')})
    write_json(output/'receipt.json',result);write_json(output/'instructions.json',[row['instruction'] for row in rows])
    print(json.dumps(dict(count=len(rows),category_comparison=result['category_comparison'])))


if __name__=='__main__': main()
