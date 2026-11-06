"""CPU-only export/attribution of existing D10 Nsight reports. No model imports."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path('/root/nfs/embodied/benchmark')
OUT = Path(__file__).resolve().parent
RUN = OUT.parent
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
from preparations.idea2 import layer_trace


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    cases = ['d10_full_serial_profile', 'd10_full_overlap_profile',
             'd10_fixed16_profile', 'd10_combined_profile', 'd10_dynamic16_profile',
             'd10_encoder64_vlmfull_profile_resumed']
    if len(sys.argv) > 1:
        cases = sys.argv[1:]
    receipt = []
    baseline_path = OUT / 'd10_full_serial_profile/report.json'
    for name in cases:
        trace_name = ('d10_encoder64_vlmfull_profile_20261024_v2.nsys-rep'
                      if name == 'd10_encoder64_vlmfull_profile_resumed'
                      else name + '_20261024_v1.nsys-rep')
        source = ROOT / 'nsys/idea2_experiment' / trace_name
        sqlite = source.with_suffix('.sqlite')
        destination = OUT / name
        destination.mkdir(exist_ok=True)
        item = dict(name=name, source=str(source), sqlite=str(sqlite), export_reused=sqlite.exists())
        if not sqlite.exists():
            command = ['nsys', 'export', '--type', 'sqlite', '--output', str(sqlite), str(source)]
            with (destination / 'export.log').open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            item['export_command'] = command
        report_path = destination / 'report.json'
        previous = RUN / 'd10_dynamic16_trace'
        if name == 'd10_dynamic16_profile' and not report_path.exists() and (previous / 'report.json').exists():
            for old in previous.iterdir():
                if old.is_file():
                    shutil.copy2(old, destination / old.name)
            item['analysis_reused_from'] = str(previous)
        elif not report_path.exists():
            command = [sys.executable, '-m', 'preparations.idea2.layer_trace', str(sqlite),
                       '--output', str(destination)]
            with (destination / 'analysis.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
            item['analysis_command'] = command
        else:
            item['analysis_reused_from'] = str(report_path)
        report = json.loads(report_path.read_text())
        if name != 'd10_full_serial_profile':
            baseline = json.loads(baseline_path.read_text())
            comparison = layer_trace.compare_signatures(baseline, report)
            save(destination / 'signature_comparison.json', comparison)
            layer_trace.write_csv(destination / 'signature_comparison.csv', comparison['matches'])
            item['signature_matches'] = len(comparison['matches'])
            item['signature_unmatched'] = [comparison['baseline_unmatched_types'], comparison['candidate_unmatched_types']]
        item['attribution'] = report['attribution']
        item['groups'] = list(report['groups'])
        receipt.append(item)
        save(destination / 'offline_receipt.json', item)
        print(json.dumps(dict(name=name, requests=len(report['requests']),
                              attribution=report['attribution'], matches=item.get('signature_matches'))), flush=True)
    save(OUT / 'latest_run_receipt.json', dict(
        scope='Existing Nsight reports only; CPU export and analysis; no GPU work.',
        analysis_source_sha256=hashlib.sha256(Path(layer_trace.__file__).read_bytes()).hexdigest(),
        cases=receipt))


if __name__ == '__main__':
    main()
