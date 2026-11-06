#!/usr/bin/env python3
"""Render all historical new-failure XYZ pairs and balanced H264 examples."""
import argparse
import csv
import html
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiment6.visualize_failures import select_balanced_cases
from scripts.experiment8.plot_paired_xyz import read_trajectory, compare_trajectories, plot_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-analysis', required=True, type=Path)
    parser.add_argument('--telemetry-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--max-videos', type=int, default=12)
    args = parser.parse_args()
    with args.paired_analysis.open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['cohort'] == 'new_failure']
    selected = {(row['condition_id'], row['sample_id']) for row in select_balanced_cases(rows, args.max_videos)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(rows):
        condition, sample = row['condition_id'], row['sample_id']
        stem = f'{condition}__{sample}'
        baseline = args.telemetry_dir / 'Control' / f'{sample}.telemetry.jsonl'
        variant = args.telemetry_dir / condition / f'{sample}.telemetry.jsonl'
        comparison = compare_trajectories(read_trajectory(baseline), read_trajectory(variant),
            baseline_label='Historical Control / D3', variant_label=condition + ' / D3')
        png = args.output_dir / 'plots' / (stem + '.png')
        pdf = png.with_suffix('.pdf')
        png.parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(comparison, baseline_label='Historical Control / D3', variant_label=condition + ' / D3',
                        png_path=png, pdf_path=pdf)
        summary = {key:value for key,value in comparison.items() if key != '_plot_data'}
        png.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
        video = None
        if (condition, sample) in selected:
            video = args.output_dir / 'videos' / (stem + '.mp4')
            if not video.exists():
                subprocess.run([sys.executable, str(ROOT / 'scripts/experiment8/render_xyz_video.py'),
                    '--baseline-video', row['control_video'], '--variant-video', row['candidate_video'],
                    '--baseline-telemetry', str(baseline), '--variant-telemetry', str(variant),
                    '--variant-label', condition + ' / D3', '--output', str(video)], check=True)
            else:
                raise FileExistsError('Refusing to silently reuse unvalidated gallery video: ' + str(video))
        records.append({'condition':condition, 'sample_id':sample,
            'png':str(png.relative_to(args.output_dir)), 'pdf':str(pdf.relative_to(args.output_dir)),
            'video':str(video.relative_to(args.output_dir)) if video else None, 'summary':summary})
        print(f'[{index+1}/{len(rows)}] {stem}', flush=True)
    (args.output_dir / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    cards = []
    for record in records:
        video_tag = f'<video controls preload="metadata" src="{html.escape(record["video"])}"></video>' if record['video'] else ''
        cards.append(f'<article><h2>{html.escape(record["condition"])} · {html.escape(record["sample_id"])}</h2>'
                     f'<img loading="lazy" src="{html.escape(record["png"])}">{video_tag}</article>')
    document = '''<!doctype html><meta charset="utf-8"><title>Measured XYZ comparisons</title>
<style>body{font:16px sans-serif;max-width:1400px;margin:24px auto}img,video{width:100%;height:auto}article{margin:40px 0}</style>
<h1>Historical D3 Control / pruned variants: measured world XYZ</h1>
<p>Blue is Historical Control / D3, not the canonical D10 baseline. Orange is the pruned variant / D3.
Videos preserve source-recorded timing, not guaranteed real-time simulation timing. Only common valid steps contribute to error.
Initial XYZ and terminal video frames were not saved in the old experiment. H264/decode checks do not replace testing your player.</p>'''
    (args.output_dir / 'index.html').write_text(document + '\n'.join(cards))


if __name__ == '__main__':
    main()
