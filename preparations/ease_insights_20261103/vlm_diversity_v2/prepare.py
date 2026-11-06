"""Freeze the broad-mask study, gated on independent Full health acceptance."""
import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import time
from .common import ROOT, SUITES, LIMITS, MASKS, CONDITIONS, ANALYSIS_FILES
from .common import DISCOVERY_PHASE, CONFIRMATION_PHASE, sha, digest, atomic_json, conditions, validate_output, read_manifest

PACKAGE=Path(__file__).resolve().parent
BASE=ROOT/'results/ease_insights_20261103/baseline_recovery_v1/calibration/manifest.json'
HEALTH=ROOT/'reports/ease_insights_20261103/baseline_recovery_v1/analysis.json'
STUDY=ROOT/'results/ease_insights_20261103/vlm_diversity_v2'
REPORT=ROOT/'reports/ease_insights_20261103/vlm_diversity_v2'


def protocol():
    path=REPORT/'protocol.json'
    if path.exists():
        return dict(path=str(path),sha256=sha(path))
    geometry=ROOT/'reports/ease_insights_20261103/future_path_diversity_preparation/plan.json'
    value=dict(revision='ease-vlm-diversity-v2',status='prospectively_frozen_before_new_path_outcomes',frozen_at=time.time(),
        conditions=conditions(),suite_order=list(SUITES),tasks=list(range(10)),
        proposal_basis=dict(path=str(geometry),sha256=sha(geometry),
            scope='Four geometric masks proposed before any outcomes for them. The LIBERO90 task pool was discarded after failed Full health; these proposals move to the complete common-suite pool.'),
        fixed_science=dict(H=50,R=5,refresh_D=10,fast_D=2,cycle=['FullD10','FastD2','FastD2'],
            VLM_layers_skipped=6,nominal_VLM_layers_retained=12,actual_fast_hidden_updates=11,
            AE_layers=18,vision_layers=27,compensation='own-path residual extrapolation',policy_action_limits=dict(LIMITS),settling_actions=10,environment_seed=7),
        health_gate=dict(calibration_init=44,minimum_total_successes=32,minimum_suite_successes=6,no_task_filtering=True),
        discovery=dict(inits=[45,46],episodes=400,conditions=list(CONDITIONS),
            usability='overall>=.75 and >=Full-.15',minimum_fixed_input_cost_ratio=1.15,
            selection='All six pairs and ordered distinct-suite contrasts. Require each opposite gap>=.10, at least two positive tasks, every task leave-one-out gap>0. Rank by minimum gap, sum gaps, minimum overall success, then lexical ids/suites. Freeze only after all400 accepted.',
            inferential_status='Exploratory selection; no discovery significance claim'),
        confirmation=dict(inits=[47,48,49],episodes=360,conditions='Full and exactly two frozen selected paths',
            primary='One-sided exact ten-task sign-flip randomization including ties; Holm2 across exactly two frozen directions',
            sensitivity='One-sided paired-case exact McNemar, Holm2, descriptive only',
            bootstrap_replicates=20000,bootstrap_seed=20261105,bootstrap_quantile_method='linear',
            verdict='Both directions positive, Holm2 primary p<.05, whole-task bootstrap lower>0, every task leave-one-out>0; both paths remain>=.75 and >=Full-.15 overall',
            no_reselection=True),new_episode_cap=760,
        termination='No qualified discovery pair means no confirmation. Valid policy failures are never rerun. This bounded branch ending does not complete the overall research goal.',
        interpretation='Suite-conditioned path behavior only; no universal semantic-category, runtime-selector or exclusive-GPU acceleration claim',
        all_previous_negative_evidence_preserved=True)
    atomic_json(path,value)
    return dict(path=str(path),sha256=sha(path))


def unused(base,inits):
    prior=base['initial_state_prior_use_audit']['catalog'];catalog=Path(prior['path'])
    if sha(catalog)!=prior['sha256']:
        raise ValueError('Official alias inventory changed')
    with catalog.open() as stream:
        table={(r['suite'],int(r['task'])):r for r in csv.DictReader(stream)}
    target={table[(s,t)]['full_bddl_identity'] for s in SUITES for t in range(10)}
    root=ROOT/'results/ease_insights_20261103'
    proc=subprocess.run(['rg','--files','--hidden','--no-ignore','-g','identity.json','-g','result.json','-g','episode_receipt.json',str(root)],text=True,capture_output=True)
    if proc.returncode not in (0,1) or proc.stderr.strip():
        raise RuntimeError('Prior-use inventory failed')
    matches=[];unresolved=[];inspected=0
    for name in sorted(set(proc.stdout.splitlines())):
        path=Path(name);indices=[int(p[4:]) for p in path.parts if p.startswith('init') and p[4:].isdigit()]
        if indices and not set(indices)&set(inits):
            continue
        value=json.loads(path.read_text());scene=value.get('case',{})
        if not all(k in scene for k in ('suite','task','init')) or scene['init'] not in inits:
            continue
        inspected+=1;row=table.get((scene['suite'],scene['task']))
        if row and row['full_bddl_identity'] in target:
            matches.append(dict(path=name,sha256=sha(path),case=scene))
        elif scene['suite'] in SUITES:
            unresolved.append(dict(path=name,case=scene))
    if matches or unresolved:
        raise ValueError('Requested official states not verified unused: '+json.dumps(dict(matches=matches,unresolved=unresolved)))
    return dict(passed=True,roots=[str(root)],requested_inits=inits,catalog=prior,
        inventoried_files=len(set(proc.stdout.splitlines())),matching_existing_outcomes=matches,unresolved=unresolved,
        inspected_requested_documents=inspected,outcome_success_not_used=True,time=time.time(),
        scope='Known EASE JSON outcomes and complete official physical-task aliases only; plans alone are not outcomes')


def prepare(stage='discovery',selection_path=None):
    pin=protocol()
    output=STUDY/('discovery' if stage=='discovery' else 'confirmation');destination=output/'manifest.json'
    if destination.exists():
        m=read_manifest(destination,verify_sources=True)
        return dict(reused=True,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(destination))
    if not HEALTH.exists():
        raise ValueError('Await all40 Full calibration cases and health acceptance')
    health=json.loads(HEALTH.read_text())
    if health.get('status')!='complete' or health.get('episodes')!=40 or not health.get('health_passed'):
        raise ValueError('Full calibration failed; do not launch broad-mask study on this pool')
    from preparations.ease_insights_20261103.baseline_recovery_v1.common import read_manifest as read_base
    base=read_base(BASE,verify_sources=True)
    if health.get('manifest_hash')!=base['manifest_hash']:
        raise ValueError('Full health evidence belongs to another calibration')
    choice=None;selected=None
    if stage=='confirm':
        from .selection import read_selection
        selection_path=Path(selection_path or STUDY/'discovery/analysis/selection.json')
        selected=read_selection(selection_path);choice=selected['selection']['selected']
        if selected['selection']['status']!='selected' or not choice:
            raise ValueError('No qualified frozen pair for confirmation')
        discovery=read_manifest(STUDY/'discovery/manifest.json',verify_sources=True)
        if selected['discovery_manifest_hash']!=discovery['manifest_hash']:
            raise ValueError('Selection belongs to another discovery')
    inits=[45,46] if stage=='discovery' else [47,48,49]
    prior=unused(base,list(range(45,50)) if stage=='discovery' else inits)
    # Recheck exact official files from the independently accepted CPU asset audit.
    official=copy.deepcopy(base['official_init_availability'])
    for row in official['cases']:
        if sha(row['bddl_path'])!=row['bddl_sha256'] or sha(row['init_file'])!=row['init_file_sha256']:
            raise ValueError('Official BDDL/init asset changed')
    official.update(calibration_init=None,discovery_inits=[45,46],reserved_confirmation_inits=[47,48,49])
    keys=('checkpoints','bank','environment','noise','science','quality_environment_reuse','technical_max_attempts','preflight')
    m={k:copy.deepcopy(base[k]) for k in keys}
    local={str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.glob('*.py'))}
    sources=dict(base['source_files']);sources.update(local)
    analysis={name:sha(ROOT/name) for name in ANALYSIS_FILES};sources.update(analysis)
    m.update(revision='ease-vlm-diversity-v2',phase=DISCOVERY_PHASE if stage=='discovery' else CONFIRMATION_PHASE,
        output_root=str(output),conditions=conditions(),planned_conditions=list(CONDITIONS) if stage=='discovery' else ['full_d10',*choice['paths']],
        suites=list(SUITES),tasks=list(range(10)),case_table=[dict(suite=s,task=t) for s in SUITES for t in range(10)],
        cases_per_condition=80 if stage=='discovery' else 120,official_init_availability=official,initial_state_prior_use_audit=prior,
        protocol_evidence=pin,design_evidence=pin,health_evidence=dict(path=str(HEALTH),sha256=sha(HEALTH)),
        local_runtime_sources=local,source_files=sources,analysis_sources=analysis,
        legacy_asset_reference=dict(path=str(BASE),sha256=sha(BASE),manifest_hash=base['manifest_hash']),
        resource_execution_revision=copy.deepcopy(base['resource_execution_revision']),new_scientific_episode_cap=760)
    if stage=='discovery':
        m.update(discovery_inits=inits,total_discovery_episodes=400,
            independent_validation=dict(status='locked_pending_complete_qualified_selection',inits=[47,48,49]))
    else:
        m.update(confirmation_inits=inits,total_confirmation_episodes=360,
            selection_evidence=dict(path=str(selection_path),sha256=sha(selection_path),selection_hash=selected['selection_hash'],frozen_selection=selected['selection'],evidence=selected['evidence']),
            discovery_reference=dict(root=str(STUDY/'discovery'),manifest_hash=discovery['manifest_hash'],manifest_sha256=sha(STUDY/'discovery/manifest.json'),valid_episodes_at_freeze=400),
            preregistered_hypotheses=copy.deepcopy(choice['directions']),heldout_preparation=dict(no_heldout_outcomes_before_freeze=True,inits=inits))
    m['manifest_hash']=digest(m);validate_output(output,m);atomic_json(destination,m)
    read_manifest(destination,verify_sources=True)
    return dict(reused=False,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(destination),episodes=400 if stage=='discovery' else 360,GPU_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--stage',choices=('discovery','confirm'),default='discovery')
    parser.add_argument('--selection');parser.add_argument('--protocol-only',action='store_true')
    args=parser.parse_args();print(json.dumps(protocol() if args.protocol_only else prepare(args.stage,args.selection)))
