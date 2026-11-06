"""Atomic cross-worker prediction budget; failed calls remain charged."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sqlite3
import time


class CallBudget:
    def __init__(self, path: Path, limit: int):
        self.path, self.limit = Path(path), int(limit)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, budget INTEGER)')
            db.execute('INSERT OR IGNORE INTO config VALUES (1, ?)', (self.limit,))
            if db.execute('SELECT budget FROM config WHERE id=1').fetchone()[0] != self.limit:
                raise RuntimeError('Frozen call budget changed')
            db.execute('CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, pid INTEGER, '
                       'started REAL, identity TEXT, status TEXT, details TEXT)')

    def connect(self):
        return sqlite3.connect(self.path, timeout=120)

    def reserve(self, identity: dict) -> int:
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            count = db.execute('SELECT COUNT(*) FROM calls').fetchone()[0]
            if count >= self.limit:
                raise RuntimeError(f'Prediction budget exhausted: {count}/{self.limit}')
            result = db.execute('INSERT INTO calls VALUES (NULL,?,?,?,?,?)',
                                (os.getpid(), time.time(), json.dumps(identity, sort_keys=True), 'reserved', '{}'))
            return int(result.lastrowid)

    def finish(self, call_id: int, status: str, details: dict):
        with self.connect() as db:
            db.execute('UPDATE calls SET status=?, details=? WHERE id=?',
                       (status, json.dumps(details, sort_keys=True), call_id))

    def summary(self):
        with self.connect() as db:
            counts = dict(db.execute('SELECT status, COUNT(*) FROM calls GROUP BY status').fetchall())
        return {'limit': self.limit, 'charged_calls': sum(counts.values()), 'statuses': counts}


def accounted_infer(backend, skip, steps, *, budget: CallBudget | None, identity: dict):
    call = budget.reserve({**identity, 'skip_layers': list(skip), 'steps': steps}) if budget else None
    details = {'internal_d10_parity_forwards': 2 if backend.length not in backend.parity_lengths else 0,
               'internal_warmup_capture_forwards': 4 if (tuple(skip), steps) not in backend.graphs else 0,
               'requested_schedule_denoising_steps': steps}
    try:
        result = backend.infer(skip, steps)
    except BaseException as exc:
        if budget:
            budget.finish(call, 'failed', {**details, 'exception': type(exc).__name__})
        raise
    if budget:
        budget.finish(call, 'complete', details)
    return result
