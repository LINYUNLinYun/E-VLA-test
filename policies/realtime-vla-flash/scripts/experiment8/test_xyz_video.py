import json
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np

from scripts.experiment8.render_xyz_video import read_telemetry


def test_h264_video_frame_alignment(tmp_path):
    # Three pre-action scene frames, three post-action states: frame 0 has no XYZ.
    for name, offset in [('baseline', 0), ('variant', .02)]:
        video = tmp_path / f'{name}.mp4'
        with imageio.get_writer(video, fps=10, codec='libx264', pixelformat='yuv420p') as writer:
            for i in range(3):
                frame = np.zeros((32,32,3),dtype=np.uint8)
                frame[...,0] = 80 + i * 40
                writer.append_data(frame)
        rows = [{'step':i,'when':'post_action','eef_pos':[i*.01, offset, 1]} for i in range(3)]
        (tmp_path / f'{name}.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    output = tmp_path / 'paired.mp4'
    script = Path(__file__).with_name('render_xyz_video.py')
    subprocess.run([sys.executable, str(script), '--baseline-video', str(tmp_path/'baseline.mp4'),
        '--variant-video', str(tmp_path/'variant.mp4'), '--baseline-telemetry', str(tmp_path/'baseline.jsonl'),
        '--variant-telemetry', str(tmp_path/'variant.jsonl'), '--variant-label', 'synthetic variant',
        '--output', str(output)], check=True, capture_output=True)
    report = json.loads(output.with_suffix('.json').read_text())
    assert report['encoding_validation']['codec_name'] == 'h264'
    assert report['frame_index'][0]['matching_post_action_step'] is None
    assert report['frame_index'][0]['paired_error_m'] is None
    assert report['frame_index'][1]['matching_post_action_step'] == 0
    assert np.isclose(report['frame_index'][1]['paired_error_m'], .02)


def test_invalid_telemetry_rejected(tmp_path):
    import pytest
    path = tmp_path/'invalid.jsonl'
    path.write_text(json.dumps({'step':0,'when':'pre_action','eef_pos':[0,0,0]}))
    with pytest.raises(ValueError, match='post-action'):
        read_telemetry(path)
