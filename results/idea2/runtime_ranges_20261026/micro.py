"""NVTX projection smoke: real graph nodes, runtime ranges, no model timing claim."""
import json
from pathlib import Path
import sys
import torch


def main():
    torch.cuda.set_device(0)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs = []
    owned = []
    for index in range(3):
        stream = streams[index == 2]
        with torch.cuda.stream(stream):
            a = torch.randn((1024,1024), device="cuda", dtype=torch.bfloat16)
            b = torch.randn_like(a)
            out = torch.empty_like(a)
            for _ in range(2):
                torch.mm(a,b,out=out)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(8):
                    torch.mm(a,b,out=out)
                torch.cuda._sleep(10_000_000)
        graphs.append(graph)
        owned.append((a,b,out))
    torch.cuda.synchronize()
    for _ in range(2):
        for index,graph in enumerate(graphs):
            with torch.cuda.stream(streams[index == 2]):
                graph.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        for _ in range(3):
            with torch.cuda.nvtx.range("preprocess"):
                sum(range(100))
            for index,label in enumerate(("encoder","vlm","AE")):
                with torch.cuda.stream(streams[index == 2]):
                    with torch.cuda.nvtx.range(label):
                        graphs[index].replay()
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    Path(sys.argv[1]).write_text(json.dumps(dict(status="passed",requests=3,
        expected_projected_ranges=dict(encoder=3,vlm=3,AE=3),
        scope="synthetic projection check only; sleep and GEMMs are not pi0.5 measurements"),indent=2)+"\n")


if __name__=="__main__":main()
