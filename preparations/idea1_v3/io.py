"""Small atomic commits and independent worker journals, with no global ledger."""
import hashlib
import json
import os
from pathlib import Path
import time


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def jsonable(value):
    if hasattr(value,'tolist'): return value.tolist()
    if hasattr(value,'item'): return value.item()
    raise TypeError(type(value).__name__)


def atomic(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,default=jsonable,indent=2); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def append(path, row):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(dict(wall_timestamp=time.time(),**row),default=jsonable)+'\n')


def write_rows(path, rows):
    with Path(path).open('x') as stream:
        for row in rows: stream.write(json.dumps(row,default=jsonable)+'\n')
        stream.flush(); os.fsync(stream.fileno())
