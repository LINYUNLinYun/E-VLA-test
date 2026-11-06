#!/usr/bin/env python3
"""Build a static source inventory without importing or executing project code."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOTS = ('scripts', 'tests', 'experiments', 'profiling', 'preparations', 'docs/experiments', 'policies', 'LIBERO', 'calvin', 'SimplerEnv')
SKIP = {'.git', '__pycache__', 'node_modules', 'site-packages', '.mypy_cache', '.pytest_cache', '.ruff_cache', 'results', 'logs', 'checkpoints', 'videos'}
# Keep package directories named data/dataset; they can contain real loaders.
EXTENSIONS = {'.py', '.sh', '.bash', '.c', '.cc', '.cpp', '.h', '.hpp', '.cu', '.cuh', '.ipynb', '.js', '.ts', '.tsx', '.jsx'}


def inspect(path: Path) -> dict:
    raw = path.read_bytes()
    source = raw.decode('utf-8', errors='replace')
    row = dict(path=path.relative_to(ROOT).as_posix(), language=path.suffix[1:], lines=len(source.splitlines()), sha256=hashlib.sha256(raw).hexdigest(), summary='', symbols='', imports='', cli_flags='', static_status='text')
    if path.suffix == '.py':
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError) as exc:
            row['static_status'] = f'parse_error:{type(exc).__name__}'
        else:
            row['static_status'] = 'parsed'
            row['summary'] = ' '.join((ast.get_docstring(tree) or '').split())[:400]
            row['symbols'] = '; '.join(f'{n.name}:{n.lineno}' for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
            row['imports'] = '; '.join(sorted({n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}))
            row['cli_flags'] = '; '.join(sorted({a.value for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument' for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith('-')}))
    if not row['summary']:
        row['summary'] = ' '.join(line.lstrip('#/ *').strip() for line in source.splitlines()[:20] if line.strip().startswith(('#', '//')) and not line.startswith('#!'))[:400]
    return row


def main():
    rows, skipped, errors = [], [], []
    for root in ROOTS:
        def walk_error(exc):
            errors.append(str(exc))
        for base, dirs, files in os.walk(ROOT / root, onerror=walk_error):
            keep = []
            for name in sorted(dirs):
                child = Path(base) / name
                if (name == 'datasets' and child.parent.name == 'libero') or name in SKIP or name.startswith(('.venv', '.cache')) or child.is_symlink() or (child / 'pyvenv.cfg').is_file():
                    skipped.append(child.relative_to(ROOT).as_posix())
                else:
                    keep.append(name)
            dirs[:] = keep
            for name in sorted(files):
                path = Path(base) / name
                if path.suffix in EXTENSIONS and not path.is_symlink():
                    try:
                        rows.append(inspect(path))
                    except OSError as exc:
                        errors.append(f'{path.relative_to(ROOT)}: {exc}')
        print(f'{root}: scanned', flush=True)
    rows.sort(key=lambda row: row['path'])
    out = ROOT / 'docs/code'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'inventory.tsv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter('/'.join(row['path'].split('/')[:2]) if row['path'].startswith('policies/') else row['path'].split('/')[0] for row in rows)
    report = dict(generated_at=datetime.now(timezone.utc).isoformat(), roots=ROOTS, source_files=len(rows), source_lines=sum(row['lines'] for row in rows), counts=dict(sorted(counts.items())), languages=dict(Counter(row['language'] for row in rows)), skipped_directories=skipped, read_errors=errors, parse_errors=[row['path'] for row in rows if row['static_status'].startswith('parse_error')], method='Static file content and Python AST; no project imports, execution, GPU queries, or verification of runtime correctness.')
    (out / 'inventory_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('source_files', 'source_lines', 'counts', 'read_errors', 'parse_errors')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
