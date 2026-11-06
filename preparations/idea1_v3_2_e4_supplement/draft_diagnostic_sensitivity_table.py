"""CPU tables of fixed-case action sensitivity, explicitly separate from closed-loop effects."""
import argparse
import json
from pathlib import Path
from .draft_diagnostic_audit import DEFAULT_OUT,read,dump,csvout
from .draft_diagnostic_finalize import finalize


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=DEFAULT_OUT);a=p.parse_args()
    finalize(a.directory);summary=read(a.directory/'diagnostic_summary.json');rows=[]
    for record in summary['physical_input12']:
        valid=record['status']=='complete';action=record.get('actions') or {};state=record.get('state_geometry') or {}
        textures=(record.get('intervention') or {}).get('changed_textures',[])
        rows.append(dict(head=record['head'],category=record['category'],source_index=record['source_index'],status=record['status'],
            substitution_scope=record['substitution_scope'],approximate_auxiliary_arm=record['approximate_auxiliary_arm'],
            xyz_first5_l2_mean=action.get('xyz_first5_l2_mean'),xyz_first5_l2_max=action.get('xyz_first5_l2_max'),
            rotation_first5_l2_mean=action.get('rotation_first5_l2_mean'),
            rotation_x_first5_rms=action.get('rotation_x_first5_rms'),rotation_y_first5_rms=action.get('rotation_y_first5_rms'),rotation_z_first5_rms=action.get('rotation_z_first5_rms'),
            gripper_first5_sign_disagreements=action.get('gripper_first5_sign_disagreements'),
            input_eef_translation_mm=state.get('eef_translation_l2_m',0)*1000 if valid else None,
            input_SO3_radians=state.get('physical_rotation_geodesic_radians') if valid else None,
            native_axisangle_delta_l2=state.get('native_axisangle_delta_l2') if valid else None,
            near_2pi_branch_jump=state.get('near_2pi_branch_jump_candidate') if valid else None,
            texture_names='|'.join(t['texture_name'] for t in textures),source=record['result_path']))
    csvout(a.directory/'input_sensitivity12.csv',rows)
    complete=[r for r in rows if r['status']=='complete']
    dump(a.directory/'input_sensitivity12_receipt.json',dict(status='complete' if len(complete)==12 else 'partial',
        completed=len(complete),cases=rows,
        strict_input_pairs=summary['physical_input_strict_completed'],approximate_auxiliary_pairs=summary['physical_input_approximate_completed'],
        background_substitutions=summary['texture_substitution_counts'],
        units='Action xyz/rotation differences are native decoded controller action units. Input EEF translation is actual world position difference in mm; SO3 is a physical orientation geodesic in radians.',
        interpretation='Each head has one metadata-selected robot/noise/background case. Eleven are strict pairs; Long background252 is a preregistered resampled approximate mother appearance arm, not exact mother texture restoration. Robot action changes are not errors or failure causes; background establishes nuisance sensitivity only. No rescue in four fixed noise cases does not generalize to all noise. Closed-loop results are separate in diagnostic_summary.json.',
        invalid_metric='All old visual_preencoder RMS/L2, finite or infinite, are excluded; see metric_validity.json.'))
    print(json.dumps(dict(completed=len(complete),target=12)))


if __name__=='__main__':main()
