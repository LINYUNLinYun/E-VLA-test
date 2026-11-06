"""Repeat the synthetic input-copy probe with captured Green branch graphs."""
import gc
import json
from pathlib import Path
import time

from preparations.idea2.batch_resources import exclusive_gpu


def exercise(pair, torch):
    ae, prefix = pair.streams
    dedicated, caller = torch.cuda.Stream(), torch.cuda.current_stream()
    host = torch.arange(256, dtype=torch.float32)
    target = torch.empty_like(host, device="cuda:0")
    graphs = []
    for stream, cycles in ((ae, 100_000_000), (prefix, 30_000_000)):
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1000)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            torch.cuda._sleep(cycles)
        graphs.append(graph)
    for stream in (caller, prefix, dedicated):
        with torch.cuda.stream(stream):
            value = host.to("cuda:0")
        stream.synchronize()
    rows = []
    for repeat in range(3):
        for name, stream in (("caller", caller), ("prefix", prefix), ("dedicated", dedicated)):
            torch.cuda.synchronize()
            ae_end, prefix_end = torch.cuda.Event(), torch.cuda.Event()
            with torch.cuda.stream(ae):
                graphs[0].replay()
                ae_end.record(ae)
            with torch.cuda.stream(prefix):
                graphs[1].replay()
                prefix_end.record(prefix)
            begin = time.perf_counter()
            with torch.cuda.stream(stream):
                value = host.to("cuda:0")
            rows.append(dict(repeat=repeat, stream=name, host_copy_ms=(time.perf_counter()-begin)*1000,
                             ae_done=ae_end.query(), prefix_done=prefix_end.query()))
            torch.cuda.synchronize()
            assert torch.equal(value.cpu(), host)
    return rows


def main():
    output = Path(__file__).resolve().parent / "stage2_input_graph_stream_probe"
    output.mkdir(exist_ok=False)
    gpu = "GPU-599477fb-0507-40dc-14df-d491f7a7b202"
    result = dict(status="starting", synthetic=True, graph_work=True, model_performance=False)
    try:
        with exclusive_gpu(gpu, output):
            import torch
            from preparations.idea2.batch_bench import release_green_resources
            from preparations.idea2.green_context import GreenContextPair
            library = next(Path("results/idea2/batch_preparation_20261031/green_build").glob("*.so"))
            pair = GreenContextPair(library, first_sm_min=16)
            result.update(resources=pair.info, rows=exercise(pair, torch))
            gc.collect()
            resources = [pair]
            release_green_resources(resources, torch)
            result.update(status="passed", contexts_closed=True)
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
