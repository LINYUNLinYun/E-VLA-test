"""Trusted localhost pickle RPC, compatible with existing LIBERO episode client."""
import argparse
import pickle
import socket
import struct
import traceback

from .experiments import load_backend
from .policy import PipelinePolicy
from preparations.idea1_v2_1.runtime_client import recv_exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=9012)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-config", choices=("pi0_libero", "pi05_libero"), default="pi0_libero")
    parser.add_argument("--num-steps", type=int, choices=(1, 2, 5, 10), default=10,
                        help="Matched sampling stratum; compare speed only at the same step count")
    parser.add_argument("--submission-order", choices=("ae_first", "prefill_first"), default="prefill_first")
    parser.add_argument("--host-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true", help="Two-slot dynamic-input CUDA Graph replay (cold capture on first request)")
    parser.add_argument("--ae-priority", type=int, choices=(0, -1), default=0)
    parser.add_argument("--selective-overlap", choices=("all", "vision_only", "vlm_only"),
                        help="Experimental split Prefill graph scheduling; complete old KV is preserved")
    parser.add_argument("--hoist-schedule", action="store_true", help="Prepare fixed pi0.5 time conditioning per graph slot")
    parser.add_argument("--hoist-modulation", action="store_true", help="Also prepare original per-layer adaptive norm projections")
    parser.add_argument("--await-previous-prefix", action="store_true",
                        help="Wait for previous observation's complete KV before acquiring the old snapshot")
    args = parser.parse_args()
    if (args.selective_overlap or args.hoist_schedule or args.hoist_modulation) and not args.cuda_graphs:
        parser.error("Selective overlap and prepared schedule require --cuda-graphs")
    if (args.hoist_schedule or args.hoist_modulation) and args.model_config != "pi05_libero":
        parser.error("Prepared schedule currently supports pi05_libero only")
    backend = load_backend(args.checkpoint, args.device, model_config=args.model_config, num_steps=args.num_steps)
    backend.await_previous_prefix = args.await_previous_prefix
    if args.selective_overlap:
        from .cewp_segmented import SelectiveGraphs
        backend.graph_engine_class = SelectiveGraphs
        backend.selective_overlap = args.selective_overlap
    if args.hoist_schedule:
        from .cewp_preparation import PreparedScheduleCore
        backend.graph_core_class = PreparedScheduleCore
    if args.hoist_modulation:
        from .cewp_preparation import PreparedModulationCore
        backend.graph_core_class = PreparedModulationCore
    policy = PipelinePolicy(backend, submission_order=args.submission_order,
                            host_workers=args.host_workers, diagnostics=args.diagnostics,
                            cuda_graphs=args.cuda_graphs, ae_priority=args.ae_priority)
    try:
        serve(policy, args.port)
    finally:
        policy.close()


def serve(policy, port):
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(8)
        print("READY", flush=True)
        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    size = struct.unpack("!I", recv_exact(connection, 4))[0]
                    if size > 128 * 1024 * 1024:
                        raise ValueError("Oversized trusted-local request")
                    request = pickle.loads(recv_exact(connection, size))
                    operation = request.pop("op")
                    if operation == "health":
                        result = dict(protocol=policy.protocol)
                    elif operation == "configure":
                        result = dict(protocol=policy.configure(**request))
                    elif operation in ("reset", "drain"):
                        result = policy.reset()
                    elif operation == "shutdown":
                        result = policy.reset()
                        result["shutdown"] = True
                    elif operation == "predict":
                        result = policy.predict(**request)
                    else:
                        raise ValueError("Unknown operation")
                    result["ok"] = True
                except Exception:
                    result = dict(ok=False, error=traceback.format_exc())
                payload = pickle.dumps(result, protocol=5)
                connection.sendall(struct.pack("!I", len(payload)) + payload)
                if result.get("shutdown"):
                    return


if __name__ == "__main__":
    main()
