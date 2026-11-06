"""Small GPU-only lifetime probe, not a model performance measurement."""
import gc
import json
from pathlib import Path

from preparations.idea2.batch_resources import exclusive_gpu


def main():
    output = Path(__file__).resolve().parent / "stage2_green_cleanup_probe"
    output.mkdir(exist_ok=False)
    gpu = "GPU-555cde36-5e41-bc44-0c69-ea47358fd703"
    result = dict(status="starting", synthetic=True, model_performance=False)
    try:
        with exclusive_gpu(gpu, output):
            import torch
            from preparations.idea2.batch_bench import release_green_resources
            from preparations.idea2.green_context import GreenContextPair
            library = next(Path("results/idea2/batch_preparation_20261031/green_build").glob("*.so"))
            pair = GreenContextPair(library, first_sm_min=16)
            def exercise():
                source = torch.ones((512, 512), device="cuda:0")
                with torch.cuda.stream(pair.streams[0]):
                    source.record_stream(pair.streams[0])
                    temporary = source @ source
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=pair.streams[0]):
                        target = temporary @ source
                    graph.replay()
                pair.streams[0].synchronize()
                assert target[0, 0].cpu().item() == 262144.0
                return source, temporary, graph, target
            caller_owned = exercise()
            result["resources"] = pair.info
            resources = [pair]
            # Real record_stream marks and a cuBLAS workspace are alive here.
            del caller_owned
            gc.collect()
            release_green_resources(resources, torch)
            assert not resources
            primary = torch.ones((16, 16), device="cuda:0")
            result["primary_after_context_close"] = float(primary.sum().cpu())
            assert result["primary_after_context_close"] == 256.0
            del primary, pair
            gc.collect()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
            result.update(status="passed", contexts_closed=True)
    except BaseException:
        import traceback
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
