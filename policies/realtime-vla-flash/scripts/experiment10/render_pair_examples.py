"""Render indexed actual-state/RGB videos on CPU, preserving N+1 alignment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess

from scripts.experiment10.common import ROOT, RESULTS, SUITES, dump_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    parser.add_argument('--variant', choices=('Full18_D1', 'Full18_D2', 'Full18_D4'), required=True)
    args = parser.parse_args()
    directory = RESULTS / 'closed_loop_analysis/xyz' / args.variant / args.suite
    index_path = directory / 'paired_xyz_index.json'
    index = json.loads(index_path.read_text())
    roots = [RESULTS / 'closed_loop' / c / args.suite / 'run' for c in ('Full18_D10', args.variant)]
    episodes = [{(e['task_id'], e['episode_idx']): e for e in json.loads((r/'episode_log.json').read_text())}
                for r in roots]

    def render(example):
        key = example['task'], example['init']
        output = directory / ('task%02d_init%02d_synchronized.mp4' % key)
        provenance = output.with_suffix('.provenance.json')
        sources = []
        for root, mapping in zip(roots, episodes):
            episode = mapping[key]
            sources.extend([root / episode['video_path'], root / episode['telemetry_path']])
        evidence = {str(p): sha256(p) for p in sources}
        if output.exists():
            if not provenance.exists() or json.loads(provenance.read_text())['source_sha256'] != evidence:
                raise RuntimeError('Existing video lacks matching provenance; preserve and inspect '+str(output))
            saved = json.loads(provenance.read_text())
            if saved['video_sha256'] != sha256(output) or saved['renderer_sha256'] != sha256(ROOT/'scripts/experiment8/render_xyz_video.py'):
                raise RuntimeError('Video or renderer changed; preserve and inspect '+str(output))
            from scripts.experiment8.render_xyz_video import validate_video
            report = json.loads(output.with_suffix('.json').read_text())
            validate_video(output, len(report['frame_index']))
            return {'task': key[0], 'init': key[1], 'output': str(output), 'reused': True}
        command = [str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/experiment8/render_xyz_video.py'),
                   '--baseline-video', str(sources[0]), '--baseline-telemetry', str(sources[1]),
                   '--variant-video', str(sources[2]), '--variant-telemetry', str(sources[3]),
                   '--baseline-label', 'Full18-D10', '--variant-label', args.variant,
                   '--alignment', 'state-synchronous', '--output', str(output)]
        with output.with_suffix('.log').open('w') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        report = json.loads(output.with_suffix('.json').read_text())
        if report['encoding_validation']['codec_name'] != 'h264':
            raise RuntimeError('Unexpected video codec')
        dump_json(provenance, {'source_sha256': evidence, 'video_sha256': sha256(output),
                              'renderer_sha256': sha256(ROOT/'scripts/experiment8/render_xyz_video.py')})
        return {'task': key[0], 'init': key[1], 'output': str(output), 'reused': False}

    with ThreadPoolExecutor(max_workers=2) as pool:
        videos = list(pool.map(render, index['examples']))
    dump_json(directory / 'synchronized_video_index.json', {'complete': True, 'videos': videos,
              'source_index_sha256': sha256(index_path), 'alignment': 'state-synchronous N+1',
              'full_decode_verified_by_renderer': True})
    print(args.suite, len(videos), flush=True)


if __name__ == '__main__':
    main()
