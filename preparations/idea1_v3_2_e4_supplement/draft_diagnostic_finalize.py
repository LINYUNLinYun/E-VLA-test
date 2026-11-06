"""CPU-only unified evidence summary; invalid visual-buffer scales are always excluded."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .draft_diagnostic_audit import DEFAULT_OUT,HEADS,read,dump


def validity(directory):
    return dict(status='excluded_visual_metric',affected_scope='Every visual_preencoder RMS/L2 in the first diagnostic service responses, finite and nonfinite alike.',
        invalid_fields=['feature_scales.visual_preencoder.rms','feature_scales.visual_preencoder.l2'],
        reason='Draft route calls prepare(encode=False). vision_encoder fills vision_x, and spec._project_vision_prefix fills its separate _vision_prefix_buf. encoder_x[:512] is not populated in that route and can contain uninitialized memory. The first diagnostic service read those unused encoder_x rows.',
        prohibited_interpretations=['No visual feature explosion claim from inf/large values.',
                                    'No feature stability or similarity claim from finite equal values.'],
        still_valid=['Normalized state scales: copied from fresh transformed state.',
                     'Raw/decoded action outputs and actual route counters.',
                     'Recorded GPU3 cross-device bitwise equality gate and same-input repeat equality.'],
        action_scope='The invalid metric is calculated only after policy.predict returns; encoder_x visual rows are not consumed by the learned draft prefix path. It does not alter the draft output or any fixed head weights.',
        no_model_restart=True,source_service_sha256=hashlib.sha256(Path(__file__).with_name('draft_diagnostic_service.py').read_bytes()).hexdigest())


def response_check(response):
    errors=[]
    for key,shape in (('actions',(50,7)),('raw_actions',(50,32))):
        array=np.asarray(response[key],np.float32)
        if array.shape!=shape or not np.isfinite(array).all():errors.append('invalid_'+key)
    observed=response['observed_calls']
    for key,expected in (('vision_encoder_calls',1),('learned_head_calls',1),('ae_layer_calls',0),('vlm_prefix_calls',0),('verify_calls',0),('constructor_full_record_calls',0)):
        if observed.get(key,0)!=expected:errors.append('unexpected_'+key)
    return errors


def finalize(directory):
    directory=Path(directory);dump(directory/'metric_validity.json',validity(directory))
    summary=dict(status='partial',heads=list(HEADS.values()),metric_validity='metric_validity.json',
        interpretation='Report each released head separately. Fixed-case interventions are local mechanism evidence; no new whole-suite score is implied.',
        original40=read(directory/'original_protocol40/analysis.json') if (directory/'original_protocol40/analysis.json').exists() else None,
        language_probe=read(directory/'language_probe_full_chunk_audit.json') if (directory/'language_probe_full_chunk_audit.json').exists() else None,
        restore8=[],physical_input12=[],posthoc4=[],response_validation_errors=[])
    plan=read(directory/'paired_plan.json')['cases']
    for row in plan:
        if row['category'] in ('Language Instructions','Sensor Noise'):
            path=directory/'restore8'/('case%04d'%row['source_index'])/'result.json'
            v2=directory/'capture_v2/restore8'/('case%04d'%row['source_index'])/'result.json'
            if v2.exists():path=v2
            result=read(path) if path.exists() else {}
            summary['restore8'].append(dict(head=row['head'],category=row['category'],source_index=row['source_index'],
                result_path=str(path),
                status=result.get('status','missing'),actual_reproduction_gate_passed=result.get('actual_reproduction_gate',{}).get('passed'),
                old_actual_success=result.get('actual_reproduction_gate',{}).get('old_actual_success'),new_restored_success=result.get('success'),
                old_actual_steps=result.get('actual_reproduction_gate',{}).get('old_actual_steps'),new_steps=result.get('steps'),
                paired_effect=result.get('paired_effect'),language_semantic_scope=result.get('language_semantic_scope'),
                physics_all_bitwise_equal=all(result.get('physical_bitwise_equal',{}).values()) if result.get('physical_bitwise_equal') else None))
        if row['category']!='Language Instructions':
            path=directory/'physical_input_pairs'/('case%04d'%row['source_index'])/'receipt.json'
            v2=directory/'capture_v2/physical_input_pairs'/('case%04d'%row['source_index'])/'receipt.json'
            if v2.exists():path=v2
            v3=directory/'capture_v3/physical_input_pairs'/('case%04d'%row['source_index'])/'receipt.json'
            if v3.exists():path=v3
            v4=directory/'capture_v4/physical_input_pairs'/('case%04d'%row['source_index'])/'receipt.json'
            if v4.exists():path=v4
            v5=directory/'capture_v5/physical_input_pairs'/('case%04d'%row['source_index'])/'receipt.json'
            if v5.exists():path=v5
            result=read(path) if path.exists() else {}
            textures=result.get('intervention',{}).get('changed_textures',[])
            approximate=any(t.get('approximate_mother_appearance',False) for t in textures)
            substitution=('resampled_approximate' if approximate else 'exact_texture_substitution') if row['category']=='Background Textures' else 'strict_input_intervention'
            summary['physical_input12'].append(dict(head=row['head'],category=row['category'],source_index=row['source_index'],
                result_path=str(path),
                substitution_scope=substitution,approximate_auxiliary_arm=approximate,
                status=result.get('status','missing'),single_factor_gate=result.get('valid_single_factor_input_pair'),
                actions=result.get('actions'),intervention=result.get('intervention'),state_geometry=result.get('state_geometry')))
    for index in range(4):
        path=directory/'posthoc_object_settle4'/('episode%03d'%index)/'result.json'
        result=read(path) if path.exists() else {};gatepath=path.with_name('reference_gate.json')
        summary['posthoc4'].append(dict(episode_index=index,status=result.get('status','missing'),
            success=result.get('success'),steps=result.get('steps'),design=result.get('design'),
            reference_gate=read(gatepath) if gatepath.exists() else None))
    streams=list((directory/'language_same_observation').glob('*_responses.jsonl'))
    streams+=list((directory/'physical_input_pairs').glob('case*/responses.jsonl'))
    streams+=list((directory/'restore8').glob('case*/infer.jsonl'))
    streams+=list((directory/'capture_v2/restore8').glob('case*/infer.jsonl'))
    streams+=list((directory/'capture_v2/physical_input_pairs').glob('case*/responses.jsonl'))
    streams+=list((directory/'capture_v3/physical_input_pairs').glob('case*/responses.jsonl'))
    streams+=list((directory/'capture_v4/physical_input_pairs').glob('case*/responses.jsonl'))
    streams+=list((directory/'capture_v5/physical_input_pairs').glob('case*/responses.jsonl'))
    checked=0
    counts=dict(restore=0,language=0,physical_input=0)
    for path in streams:
        with path.open() as stream:
            for index,line in enumerate(stream):
                response=json.loads(line)['response'];errors=response_check(response);checked+=1
                group='language' if path.name.endswith('_responses.jsonl') else 'restore' if path.name=='infer.jsonl' else 'physical_input'
                counts[group]+=1
                if errors:summary['response_validation_errors'].append(dict(path=str(path),row=index,errors=errors))
    summary['validated_response_count']=checked
    summary['validated_response_counts']=counts
    summary['reuse_reference_audit']=[]
    for version in ('capture_v2','capture_v3','capture_v4','capture_v5'):
        for name in ('reused_restore_results.json','reused_input_results.json'):
            linkpath=directory/version/name
            if not linkpath.exists():continue
            for link in read(linkpath):
                source=Path(link['path']);actual=hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None
                summary['reuse_reference_audit'].append(dict(manifest=str(linkpath),source=str(source),source_index=link['source_index'],
                    sha256_matches=actual==link['sha256'],complete=source.exists() and read(source).get('status')=='complete'))
    summary['restore_completed']=sum(r['status']=='complete' for r in summary['restore8'])
    summary['restore_gate_failed']=sum(r['status']=='actual_reproduction_gate_failed' for r in summary['restore8'])
    summary['physical_input_completed']=sum(r['status']=='complete' for r in summary['physical_input12'])
    summary['physical_input_gate_failed']=sum(r['status']=='intervention_gate_failed' for r in summary['physical_input12'])
    summary['physical_input_strict_completed']=sum(r['status']=='complete' and not r['approximate_auxiliary_arm'] for r in summary['physical_input12'])
    summary['physical_input_approximate_completed']=sum(r['status']=='complete' and r['approximate_auxiliary_arm'] for r in summary['physical_input12'])
    summary['texture_substitution_counts']={scope:sum(r['status']=='complete' and r['substitution_scope']==scope for r in summary['physical_input12'])
        for scope in ('exact_texture_substitution','resampled_approximate')}
    summary['remaining_limitations']=['Long case252 is a preregistered1024-to512 PIL LANCZOS2D RGB approximate appearance arm, not exact mother texel restoration; v4 exact shape failure is preserved.',
        'Robot input changes require action changes and are not model error or closed-loop failure evidence. Background probes establish nuisance sensitivity only.',
        'No rescue in four fixed noise restore cases does not establish absence of noise effects elsewhere.',
        'All first-service visual_preencoder RMS/L2 values are invalid and excluded.']
    cells=[]
    paths=list((directory/'original_protocol40').glob('episode*/result.json'))+list((directory/'posthoc_object_settle4').glob('episode*/result.json'))
    for path in paths:
        result=read(path);design=result['design']
        if design['head']!='draft_object' or design['init_index']!=0 or design['prompt']!='legacy_filename':continue
        if result.get('status')!='complete':continue
        sourceplan=read(path.parent.parent/'protocol_plan.json');settle=sourceplan['settle'][design['settle']]
        with np.load(path.with_name('initial_observation.npz'),allow_pickle=False) as archive:
            aperture=float(np.abs(archive['robot0_gripper_qpos']).sum()*1000)
        with np.load(path.with_name('initial_physical.npz'),allow_pickle=False) as archive:qvel_norm=float(np.linalg.norm(archive['qvel']))
        cells.append(dict(task=design['case']['task_index'],settle_steps=settle['steps'],settle_gripper_command=settle['action'][-1],
            success=result['success'],steps=result['steps'],initial_two_finger_aperture_mm=aperture,
            initial_generalized_qvel_norm=qvel_norm,qvel_units='Mixed generalized joint/object coordinates; not meters/second.',source=str(path)))
    summary['posthoc_settle_duration_by_gripper']=sorted(cells,key=lambda r:(r['task'],r['settle_steps'],r['settle_gripper_command']))
    if summary['restore_completed']+summary['restore_gate_failed']==8 and summary['physical_input_completed']+summary['physical_input_gate_failed']==12 and summary['original40'] and summary['original40']['status']=='complete' and summary['language_probe'] and all(r['status']=='complete' for r in summary['posthoc4']):
        references_ok=all(r['sha256_matches'] and r['complete'] for r in summary['reuse_reference_audit'])
        summary['status']='complete' if not summary['response_validation_errors'] and references_ok else 'validation_failed'
    dump(directory/'diagnostic_summary.json',summary)
    return {k:summary[k] for k in ('status','restore_completed','restore_gate_failed','physical_input_completed','physical_input_gate_failed','validated_response_count')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=DEFAULT_OUT)
    print(json.dumps(finalize(p.parse_args().directory)))
