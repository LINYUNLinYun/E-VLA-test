"""Read-only audit of invalid native100 pairing; never changes old results."""
import argparse
from pathlib import Path
import numpy as np
from preparations.idea1_v3_2.transfer_protocol import sha
from preparations.idea1_v3_1.common import atomic_json


def audit(root):
    root=Path(root); rows=[]
    for sequence in sorted(root.glob('seq*')):
        paths=[sequence/mode/'attempt000/initial_native_observation.npz' for mode in ('full_d10','v14_d10')]
        if not all(path.exists() for path in paths):continue
        with np.load(paths[0]) as first,np.load(paths[1]) as second:
            comparisons={}
            for key in first.files:
                a,b=first[key],second[key]
                comparisons[key]=dict(bitwise_equal=bool(np.array_equal(a,b)),max_abs=float(np.max(np.abs(a.astype(float)-b.astype(float)))),
                    different_elements=int(np.count_nonzero(a!=b)))
            rows.append(dict(sequence=sequence.name,inputs=[dict(path=str(p),sha256=sha(p)) for p in paths],
                comparisons=comparisons,full_robot_obs=first['robot_obs'].tolist(),v14_robot_obs=second['robot_obs'].tolist(),
                first14_robot_bitwise_equal=bool(np.array_equal(first['robot_obs'][:14],second['robot_obs'][:14]))))
    return dict(status='complete',scope='CPU read-only initial-observation audit, NOT task-success attribution',
        source_sha256=sha(__file__),rows=rows,invalid_for_strict_paired_quality=True,
        reason='Native reset restores physical joints/scene positions but does not restore robot.gripper_action; warm resets also retain physical dynamics/contact history, evidenced by actual scene differences.',
        disposition='Preserve all original files; exclude this whole interrupted protocol from corrected formal100 scoring, including apparently matching individual cases.',
        stopped_own_processes=[3383399,3393371,3392365],plus_untouched=True,
        repair_proposal='Fresh independent environment per sequence-condition; restore actual stored gripper command from official target before native reset; preserve native joint/motor width reset and one physics step; gate actual robot/scene/RGB equality before first quality call.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episodes',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    atomic_json(output,audit(a.episodes))


if __name__=='__main__':main()
