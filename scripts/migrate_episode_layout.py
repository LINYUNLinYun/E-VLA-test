#!/usr/bin/env python3
"""Migrate LIBERO episode outputs from a flat layout to task/episode folders.

Old: episodes/task00_ep000_task_description_success/
New: episodes/task00/ep000_success/

The migration is idempotent and updates episode_log.json paths. It refuses to
overwrite an existing destination so a partial or ambiguous migration is never
silently accepted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil


LEGACY_RE = re.compile(
    r"^task(?P<task_id>\d+)_ep(?P<episode_idx>\d+)_(?P<description>.*)_(?P<status>success|failure)$"
)
TASK_RE = re.compile(r"^task(?P<task_id>\d+)(?:_.*)?$")


def _task_dir_name(task_id: int, description: str) -> str:
    description_segment = re.sub(r"[^0-9A-Za-z]+", "_", description).strip("_").lower()
    return f"task{task_id:02d}_{description_segment}"


def _write_json(path: Path, payload: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _migrate_run(run_dir: Path, *, dry_run: bool) -> tuple[int, int, int]:
    episode_log_path = run_dir / "episode_log.json"
    episodes_dir = run_dir / "episodes"
    if not episode_log_path.is_file() or not episodes_dir.is_dir():
        return 0, 0, 0

    records = json.loads(episode_log_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"expected a list in {episode_log_path}")
    records_by_key = {
        (int(record["task_id"]), int(record["episode_idx"])): record
        for record in records
        if isinstance(record, dict) and "task_id" in record and "episode_idx" in record
    }

    episode_moves: list[tuple[Path, Path]] = []
    task_moves: list[tuple[Path, Path]] = []
    task_metadata: dict[int, str] = {
        int(record["task_id"]): str(record.get("task_description") or f"task {int(record['task_id']):02d}")
        for record in records
        if isinstance(record, dict) and "task_id" in record
    }
    for source in sorted(path for path in episodes_dir.iterdir() if path.is_dir()):
        task_match = TASK_RE.match(source.name)
        if task_match is None or LEGACY_RE.match(source.name) is not None:
            continue
        task_id = int(task_match.group("task_id"))
        description = task_metadata.get(task_id)
        task_json = source / "task.json"
        if task_json.is_file():
            payload = json.loads(task_json.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("task_description") is not None:
                description = str(payload["task_description"])
        if description is None:
            raise ValueError(f"cannot determine task instruction for {source}")
        task_metadata[task_id] = description
        destination = episodes_dir / _task_dir_name(task_id, description)
        if source == destination:
            continue
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination} while moving {source}")
        task_moves.append((source, destination))

    for source in sorted(path for path in episodes_dir.iterdir() if path.is_dir()):
        match = LEGACY_RE.match(source.name)
        if match is None:
            continue
        task_id = int(match.group("task_id"))
        episode_idx = int(match.group("episode_idx"))
        status = match.group("status")
        record = records_by_key.get((task_id, episode_idx), {})
        description = str(record.get("task_description") or match.group("description").replace("_", " "))
        task_metadata.setdefault(task_id, description)
        destination = episodes_dir / _task_dir_name(task_id, description) / f"ep{episode_idx:03d}_{status}"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination} while moving {source}")
        episode_moves.append((source, destination))

    changed_records = 0
    for record in records:
        if not isinstance(record, dict) or "task_id" not in record or "episode_idx" not in record:
            continue
        task_id = int(record["task_id"])
        episode_idx = int(record["episode_idx"])
        status = "success" if bool(record.get("success")) else "failure"
        description = str(record.get("task_description") or f"task {task_id:02d}")
        new_prefix = f"episodes/{_task_dir_name(task_id, description)}/ep{episode_idx:03d}_{status}"
        task_metadata.setdefault(task_id, description)
        for key in ("video_path", "trace_path", "infer_path"):
            old_path = record.get(key)
            if not isinstance(old_path, str) or not old_path.startswith("episodes/"):
                continue
            file_name = Path(old_path).name
            new_path = f"{new_prefix}/{file_name}"
            if old_path != new_path:
                record[key] = new_path
                changed_records += 1

    if dry_run:
        return len(task_moves), len(episode_moves), changed_records

    for source, destination in task_moves:
        shutil.move(str(source), str(destination))
    for source, destination in episode_moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    for task_id, description in task_metadata.items():
        task_dir = episodes_dir / _task_dir_name(task_id, description)
        task_dir.mkdir(parents=True, exist_ok=True)
        _write_json(task_dir / "task.json", {"task_id": task_id, "task_description": description})
        (task_dir / "instruction.txt").write_text(description + "\n", encoding="utf-8")
    if changed_records:
        _write_json(episode_log_path, records)
    return len(task_moves), len(episode_moves), changed_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Root containing run directories")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_moves = 0
    total_task_moves = 0
    total_path_updates = 0
    run_dirs = sorted({path.parent for path in args.root.rglob("episode_log.json")})
    for run_dir in run_dirs:
        task_moves, moves, path_updates = _migrate_run(run_dir, dry_run=bool(args.dry_run))
        if task_moves or moves or path_updates:
            print(f"{run_dir}: task_dirs={task_moves}, episode_dirs={moves}, path_fields={path_updates}")
        total_task_moves += task_moves
        total_moves += moves
        total_path_updates += path_updates
    print(
        f"total: runs={len(run_dirs)}, task_dirs={total_task_moves}, "
        f"episode_dirs={total_moves}, path_fields={total_path_updates}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
