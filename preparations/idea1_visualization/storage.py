"""Lossless streaming storage for full activation archives.

Compression runs in the analysis Python, which already provides zstandard,
without adding a dependency to or changing the model environment.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys


def save_compressed(path,arrays):
    import numpy as np
    path=Path(path)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    process=subprocess.Popen(['/usr/bin/python3','-m',__package__+'.storage',
                              'compress','--output',str(temporary)],stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    try:
        np.savez(process.stdin,**arrays)
        process.stdin.close()
        error=process.stderr.read().decode()
        code=process.wait()
        if code:
            raise RuntimeError(f'Activation compression failed ({code}): {error}')
        temporary.replace(path)
    finally:
        if process.poll() is None:
            process.terminate();process.wait()
        process.stderr.close()


@contextlib.contextmanager
def load_activations(path):
    """Open either original NPZ or Zstandard NPZ, preserving all native bits."""
    import numpy as np
    path=Path(path)
    if path.is_dir():
        path=path/('activations.npz.zst' if (path/'activations.npz.zst').exists() else 'activations.npz')
    if path.suffix=='.zst':
        import zstandard
        with path.open('rb') as stream:
            with zstandard.ZstdDecompressor().stream_reader(stream) as reader:
                buffer=io.BytesIO(reader.read())
        with buffer,np.load(buffer,allow_pickle=False) as arrays:
            yield arrays
    else:
        with np.load(path,allow_pickle=False) as arrays:
            yield arrays


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['compress'])
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    import zstandard
    with Path(args.output).open('wb') as destination:
        zstandard.ZstdCompressor(level=1).copy_stream(sys.stdin.buffer,destination)


if __name__=='__main__':
    main()
