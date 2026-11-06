"""Quick idea-validation stage, explicitly not the original full C01--C08 seal."""
import argparse
import json
from pathlib import Path
from common import atomic_json, sha
import quick_stage_scope_v2 as scope_module
import benchmark_light_revision as revision_module
import benchmark_light_analysis as benchmark
import benchmark_all_process_analysis as inherited
from benchmark_all_process_revision import authorize as old_authorize
from benchmark_revision_analysis import install_analysis_route


def read(path): return json.loads(Path(path).read_text())


def media_coverage(scope,attempt,indexes):
    """Reuse actual decode receipts, checking small identity tuples, not video bytes."""
    import independent_media_audit as media
    import independent_media_history_audit as history_media
    references={}; inputs=[]; sources={}
    main=set(scope['main_keys'])
    for filename in indexes:
        filename=Path(filename).resolve(); index=read(filename); evidence=filename.parents[2]
        if index.get('kind') not in ('history_aware_independent_media_coverage','independent_media_coverage'):
            raise ValueError('Unknown media index kind')
        if (index.get('source_manifest_sha256')!=sha(evidence/'source_manifest.json')
                or index.get('source_acceptance_sha256')!=sha(evidence/'source_acceptance.json')):
            raise ValueError('Media index source/acceptance changed')
        if evidence not in sources:
            if index.get('kind')=='history_aware_independent_media_coverage':
                identity,base,original=history_media.verify_source(attempt,evidence)
            elif index.get('kind')=='independent_media_coverage':
                base=evidence; original=media.verify_source(attempt,base)
            else:
                raise ValueError('Unknown media index kind')
            sources[evidence]=(base,original,sha(base/'source_manifest.json'),sha(base/'source_acceptance.json'))
        base,original,base_manifest_sha,base_acceptance_sha=sources[evidence]
        inputs.append(dict(path=str(filename),sha256=sha(filename),historical_partial=index.get('partial'),
            original_index_covered=index.get('covered_count'),source_manifest_sha256=sha(evidence/'source_manifest.json'),
            source_acceptance_sha256=sha(evidence/'source_acceptance.json')))
        for key,reference in index['receipts'].items():
            if key not in main: continue
            path=Path(reference['path']).resolve()
            if path!=base/'receipts'/f'{key}.json' or sha(path)!=reference['sha256']:
                raise ValueError('Preserved media receipt changed or escaped evidence store')
            saved=read(path); row=scope['valid_records'][key]; bound=saved['binding']; validated=row['result']
            if (saved.get('passed') is not True or saved.get('before')!=saved.get('after')
                    or saved.get('source_manifest_sha256')!=base_manifest_sha
                    or saved.get('source_acceptance_sha256')!=base_acceptance_sha
                    or any(bound.get(field)!=row[field] for field in ('key','condition','model','suite','task','init','protocol_hash','schedule','mask','token','validated_sha256'))
                    or bound.get('manifest_sha256')!=scope['manifest_sha256']
                    or bound.get('canonical_manifest_hash')!=scope['manifest_hash']
                    or bound.get('directory')!=str(Path(row['resolved_directory']))
                    or bound.get('result_sha256')!=validated['artifacts']['result.json']
                    or bound.get('video_sha256')!=validated['artifacts']['video.mp4']
                    or bound.get('actions')!=validated['actions']
                    or bound.get('expected_frames')!=validated['frames']
                    or saved.get('decoded_frames')!=validated['frames']):
                raise ValueError('Media receipt does not cover current effective canonical token/tuple')
            reference=dict(path=str(path),sha256=reference['sha256'])
            if key in references and references[key]!=reference:
                raise ValueError('Conflicting media receipts for one main key')
            references[key]=reference
    missing=sorted(main-set(references))
    return dict(main_expected=len(main),main_covered=len(references),main_covered_keys=sorted(references),
        main_missing_keys=missing,main_media_complete=not missing,original_full_media_complete=False,
        receipt_references=references,input_indexes=inputs,
        verification='Saved decode receipt SHA/source/tool and full canonical key-token-validation/result/video SHA tuple checked; historical logs and current videos not fully rehashed or decoded again')


def sample_raw(scope,attempt):
    """At most one success and one valid failure per model; no new predictions."""
    from runtime_client import check_prediction, legacy
    from common import seed
    from runtime_data import audited_protocol
    import numpy as np
    manifest=read(attempt/'condition_manifest.json'); conditions={c['id']:c for c in manifest['conditions']}
    chosen={}
    for key in sorted(scope['main_keys'],key=lambda k:(scope['valid_records'][k]['condition'],scope['valid_records'][k]['suite'],scope['valid_records'][k]['task'],scope['valid_records'][k]['init'])):
        entry=scope['valid_records'][key]; chosen.setdefault((entry['model'],entry['success']),entry)
    checks=[]
    for (model,success),entry in sorted(chosen.items()):
        path=Path(entry['resolved_directory'])/'infer.jsonl'
        with path.open() as stream: row=json.loads(stream.readline())
        check_prediction(row,conditions[entry['condition']],audited_protocol(attempt,model),seed(entry['suite'],entry['task'],entry['init'],0))
        if sha(row['input_path'])!=row['input_file_sha256'] or sha(row['prediction_path'])!=row['prediction_sha256']:
            raise ValueError('Representative input/prediction bytes changed')
        with np.load(row['input_path'],allow_pickle=False) as z:
            obs={k:z[k] for k in z.files if k not in ('instruction','seed')}
            if legacy.input_digest(obs,str(z['instruction'].item()))!=row['input_hash']:
                raise ValueError('Representative observation content differs')
        checks.append(dict(key=entry['key'],model=model,success=success,infer=str(path),
            noise_sha256=row['noise_file_sha256'],raw_prediction_sha256=row['raw_prediction_file_sha256'],input_sha256=row['input_file_sha256'],passed=True))
    return checks


def route_benchmarks(revision, view):
    """Recheck small samples and process evidence; do not rehash/reload raw NPZ."""
    identity=revision_module.verify_revision(revision); manifest=read(revision/'condition_manifest.json')
    routes={}; evidence={}
    for model in identity['required_models']:
        origin=Path(identity['inherited_db']['revision']) if model=='db-cogact' else revision
        events=inherited.verify_attempt_events if model=='db-cogact' else benchmark.verify_attempt_events
        authorize=old_authorize if model=='db-cogact' else revision_module.authorize
        selection_path=origin/'selected'/f'{model}.json'; selected=read(selection_path)
        root=Path(selected['attempt']).resolve()
        if (selected['model']!=model or selected['revision_manifest_sha256']!=sha(origin/'revision_manifest.json')
                or selected['completion_sha256']!=sha(root/'revision_complete.json')):
            raise ValueError('Benchmark selection changed')
        execution,proof,monitor=events(origin,root,model,'benchmark')
        approval,approval_path=authorize(origin,model,'benchmark')
        events(origin,Path(approval['gate_receipt']).parent,model,'allocation_gate')
        warm,measure=revision_module.POLICY[model]
        conditions=[c for c in manifest['conditions'] if c['model']==model]
        directory=root/'benchmark'/model; complete=directory/'complete.json'
        if (read(complete)!=dict(conditions=len(conditions),points=32,warmup=warm,measurement=measure,exclusive=True)
                or sha(complete)!=proof['scientific_complete_sha256']):
            raise ValueError('Mixed benchmark completion scope differs')
        expected={f"p{p:02d}_r{r:02d}_{c['id']}.json" for p in range(32) for r in range(warm+measure) for c in conditions}
        if {p.name for p in (directory/'samples').iterdir()}!=expected:
            raise ValueError('Missing or unexpected benchmark JSON samples')
        seen=set()
        for name in sorted(expected):
            row=read(directory/'samples'/name)
            benchmark.verify_call_boundary(monitor,row['all_process_guard'],name[:-5],seen)
            if (row.get('audit',{}).get('instrumentation_mode')!='minimal'
                    or row.get('timing',{}).get('shared_condition') is not False
                    or row.get('calls_after')!=row.get('calls_before',-100)+1
                    or not row.get('raw_prediction_file_sha256') or not row.get('noise_file_sha256')):
                raise ValueError('Independent minimal raw-source binding missing')
            for field in ('noise_path','server_prediction_path'):
                if not Path(row[field]).resolve().is_relative_to(root/'native'/model):
                    raise ValueError('Raw source belongs to another benchmark attempt')
        if seen!={s for s,event in monitor.items() if event['label'].endswith((':before',':after'))}:
            raise ValueError('Unaccounted benchmark monitor boundaries')
        routes[model]=str(directory)
        evidence[model]=dict(revision=str(origin),attempt=str(root),selection_sha256=sha(selection_path),
            completion_sha256=sha(root/'revision_complete.json'),authorization_sha256=sha(approval_path),
            conditions=len(conditions),calls=len(expected),measurement_denominator_per_condition=32*measure,
            raw_validation='Existing original benchmark validation/evidence reused; nested raw not rehashed by quick adapter')
    if sum(e['calls'] for e in evidence.values())!=22528:
        raise ValueError('Quick stage must retain full mixed 22528 benchmark calls')
    (view/'benchmark').mkdir()
    for model,path in routes.items(): (view/'benchmark'/model).symlink_to(path,target_is_directory=True)
    return evidence


def finalize(attempt,ledger_snapshot,drain_receipt,revision,view,media_indexes=(),reuse_evidence=()):
    attempt,revision,view=map(lambda p:Path(p).resolve(),(attempt,revision,view))
    identity=revision_module.verify_revision(revision)
    if Path(identity['source_attempt']).resolve()!=attempt:
        raise ValueError('Benchmark and quick stage belong to different attempts')
    scope=scope_module.build(attempt,ledger_snapshot,drain_receipt)
    raw_samples=sample_raw(scope,attempt)
    media_proof=media_coverage(scope,attempt,media_indexes)
    view.mkdir(parents=True,exist_ok=False)
    for name in ('condition_manifest.json','episodes','offline','parity','runtime_acceptance.json','native'):
        source=attempt/name; (view/name).symlink_to(source,target_is_directory=source.is_dir())
    (view/'episode_ledger.json').symlink_to(Path(ledger_snapshot).resolve())
    (view/'protocols').symlink_to(revision/'protocols',target_is_directory=True)
    atomic_json(view/'quick_stage_scope.json',scope)
    atomic_json(view/'retry_history_index.json',scope['retry_history'])
    atomic_json(view/'quick_stage_media_coverage.json',media_proof)
    evidence=route_benchmarks(revision,view)
    reuse={str(Path(path).resolve()):sha(path) for path in (*media_indexes,*reuse_evidence)}
    import runtime_analysis_quick_v2 as analysis
    source_hashes={str(Path(path).resolve()):sha(path) for path in
        (__file__,scope_module.__file__,analysis.__file__,revision_module.__file__,benchmark.__file__)}
    install_analysis_route(analysis,view,attempt)
    analysis.analyze(view)
    completion,_=analysis.audit(view,strict=True)
    extras=[scope['valid_records'][key] for key in scope['extra_keys']]
    supplement=[]
    for condition in read(view/'condition_manifest.json')['conditions']:
        for suite in (*scope_module.SUITES,'overall'):
            selected=[entry for entry in extras if entry['condition']==condition['id'] and (suite=='overall' or entry['suite']==suite)]
            supplement.append(dict(condition=condition['id'],suite=suite,denominator=len(selected),
                successes=sum(entry['success'] for entry in selected),
                success_rate=sum(entry['success'] for entry in selected)/len(selected) if selected else None,
                interpretation='Uneven retained extras; not part of paired main batch'))
    analysis.table(view/'supplement_extras.csv',supplement)
    atomic_json(view/'retained_extras_index.json',dict(keys=scope['extra_keys'],records={key:scope['valid_records'][key] for key in scope['extra_keys']}))
    for path,expected in {**source_hashes,**reuse}.items():
        if sha(path)!=expected: raise ValueError('Quick analysis source/reused evidence changed')
    scope_module.load_scope(view)
    receipt=dict(schema=1,kind='quick_idea_validation_stage_v2_not_full_C_seal',
        quick_stage_complete=completion['quick_stage_complete'],original_full_complete=scope['original_full_complete'],
        original_planned=10800,main_expected=6480,main_inits=[6,7,8],main_per_condition=120,main_per_suite=30,
        retained_extras=len(scope['extra_keys']),source_attempt=str(attempt),analysis_view=str(view),
        full_preserved_ledger_snapshot=scope['ledger_snapshot'],ledger_snapshot_sha256=scope['ledger_snapshot_sha256'],
        scope_sha256=sha(view/'quick_stage_scope.json'),history_sha256=sha(view/'retry_history_index.json'),
        benchmark=evidence,benchmark_total_calls=22528,offline_expected=1728,
        analysis_input_receipt_sha256=sha(view/'analysis_input_receipt.json'),
        supplement_table_sha256=sha(view/'supplement_extras.csv'),
        retained_extras_index_sha256=sha(view/'retained_extras_index.json'),
        media_coverage_sha256=sha(view/'quick_stage_media_coverage.json'),
        main_media_complete=media_proof['main_media_complete'],main_media_covered=media_proof['main_covered'],
        analysis_sources_sha256=sha(view/'analysis_sources.json'),
        completion_audit_sha256=sha(view/'completion_audit.json'),source_hashes=source_hashes,
        reused_evidence_hashes=reuse,media_indexes=[dict(path=str(Path(p).resolve()),sha256=sha(p)) for p in media_indexes],
        representative_raw_checks=raw_samples,
        independent_all_media_rechecked=False,independent_all_raw_rechecked=False,
        independent_C01_C08_all_passed=False,verification_scope=scope['verification_scope'])
    atomic_json(view/'quick_stage_final_receipt.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('attempt','ledger-snapshot','drain-receipt','benchmark-revision','view'): p.add_argument('--'+name,required=True)
    p.add_argument('--media-index',action='append',default=[])
    p.add_argument('--reuse-evidence',action='append',default=[])
    args=p.parse_args()
    finalize(args.attempt,args.ledger_snapshot,args.drain_receipt,args.benchmark_revision,args.view,args.media_index,args.reuse_evidence)


if __name__=='__main__': main()

