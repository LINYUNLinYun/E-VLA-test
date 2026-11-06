"""AST-only audit of selected robot perturbation classes and base placement dependencies."""
import ast
import hashlib
from pathlib import Path
import re
from .draft_diagnostic_audit import ROOT,DEFAULT_OUT,read,dump


def main():
    plus=Path('/policies/LIBERO-plus/libero/libero/envs')
    path=plus/'robots/mounted_panda.py';source=path.read_text()
    classes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.ClassDef)}
    manifest=read(ROOT/'results/idea1_v3_2_e4_supplement/libero_plus_half_20261022T1216Z/manifest.json')
    rows=[]
    for job in manifest['jobs']:
        case=job['case']
        if case['category']!='Robot Initial States':continue
        match=re.search(r'_initstate_(\d+)',case['name']);assert match
        variant=int(match.group(1));cls=classes['MountedPanda'+str(variant)]
        members=[n.name for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
        extras=[ast.dump(n) for n in cls.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.Expr))]
        only_init=members==['init_qpos'] and not extras and [ast.unparse(b) for b in cls.bases]==['MountedPanda']
        rows.append(dict(source_index=job['source_index'],head=case['matching_draft'],name=case['name'],variant=variant,
            class_name=cls.name,members=members,only_overrides_init_qpos=only_init,
            init_qpos_source=ast.get_source_segment(source,next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='init_qpos'))))
    dependencies=[]
    for file in [plus/'bddl_base_domain.py']+sorted((plus/'problems').glob('*.py')):
        tree=ast.parse(file.read_text())
        refs=[dict(line=n.lineno,expression=ast.unparse(n)) for n in ast.walk(tree)
              if isinstance(n,ast.Attribute) and n.attr=='init_state' and isinstance(n.ctx,ast.Load)]
        if refs:dependencies.append(dict(path=str(file),init_state_loads=refs))
    result=dict(status='complete',gpu_used=False,simulator_constructed=False,cases=rows,count=len(rows),
        all_selected_variants_only_override_init_qpos=all(r['only_overrides_init_qpos'] for r in rows),
        problem_and_base_domain_init_state_loads=dependencies,
        conclusion='Selected variant definitions inherit the same base class and only override init_qpos. This source-level evidence does not prove actual loaded initial vectors, controller synchronization or post-settle object identity; runtime physical checks remain required.',
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()})
    dump(DEFAULT_OUT/'robot_class_audit.json',result)
    print(dict(count=len(rows),all_only_init_qpos=result['all_selected_variants_only_override_init_qpos'],init_state_dependency_files=len(dependencies)))


if __name__=='__main__':main()
