"""Four outer NVTX scopes at execution/replay, not at graph construction."""
import argparse
from contextlib import contextmanager
import functools
import sys


def runtime_names(torch):
    original = torch.cuda.nvtx.range

    @contextmanager
    def scope(name, *args, **kwargs):
        if name in ("preprocess","encoder","vlm","AE"):
            label = name
        elif name.startswith("Input/Prepare["):
            label = "preprocess"
        else:
            label = None
        if label:
            with original(label):yield
        else:yield

    torch.cuda.nvtx.range = scope


def server(args, rest):
    import torch
    from preparations.idea2.progressive_kv import ProgressiveGraphs
    from results.idea2.simple_labels_20261026 import capture_native_dynamic as old
    old.simple_names = runtime_names
    graph_stages = {}
    prefix = ProgressiveGraphs._capture_prefix
    ae = ProgressiveGraphs._capture_ae
    replay = torch.cuda.CUDAGraph.replay

    @functools.wraps(prefix)
    def capture_prefix(self, *a, **kw):
        slot = prefix(self,*a,**kw)
        graph_stages[id(slot.vision_graph)] = "encoder"
        graph_stages[id(slot.prefix_graph)] = "vlm"
        return slot

    @functools.wraps(ae)
    def capture_ae(self, index):
        result = ae(self,index)
        graph_stages[id(self.slots[index].ae_graph)] = "AE"
        return result

    @functools.wraps(replay)
    def marked_replay(self):
        label = graph_stages.get(id(self))
        if label:
            with torch.cuda.nvtx.range(label):return replay(self)
        return replay(self)

    ProgressiveGraphs._capture_prefix = capture_prefix
    ProgressiveGraphs._capture_ae = capture_ae
    torch.cuda.CUDAGraph.replay = marked_replay
    old.server(args,rest)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--kind",choices=("native_jax","fresh_prepared","pipeline"),required=True)
    p.add_argument("--receipt",required=True)
    p.add_argument("--capture-warmup",dest="warmup",type=int,default=10)
    p.add_argument("--capture-count",dest="count",type=int,default=8)
    args,rest=p.parse_known_args()
    if args.kind=="native_jax":
        from results.idea2.simple_labels_20261026 import capture_native_dynamic as old
        old.simple_names=runtime_names
        old.native(args,rest)
    else:server(args,rest)


if __name__=="__main__":main()
