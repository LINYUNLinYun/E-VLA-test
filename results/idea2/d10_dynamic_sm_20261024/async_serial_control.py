"""Early-return control: serialize AE -> Vision -> VLM without blocking API on Prefill.

Results-only hooks reuse the frozen CEWP benchmark, Pool, Graph capture and oracle.
Both arms retain kind=vision_only; only the location of the same GPU wait changes.
"""
import hashlib
import os
from pathlib import Path
import shlex
import sys

from preparations.idea2 import cewp_bench as bench
from preparations.idea2 import cewp_segmented as segmented

BASE_GRAPHS, BASE_POLICY, BASE_RECORDER = segmented.SelectiveGraphs, bench.PipelinePolicy, bench.Recorder
BASE_WRITE = bench.write_results
BASE_SETTINGS = bench.settings
WRAPPER = Path(__file__).resolve()
SOURCE = WRAPPER.read_bytes()
SOURCE_SHA = hashlib.sha256(SOURCE).hexdigest()
MODULE = "results.idea2.d10_dynamic_sm_20261024.async_serial_control"
GATES = {"graph_async_serial": "ae_before_vision", "graph_vision_overlap": "ae_before_vlm"}


class ControlGraphs(BASE_GRAPHS):
    def build_prefix(self, pool, identity, prepared):
        gate = self.backend.prefix_dependency_gate
        pending = self.ae_pending
        if gate == "ae_before_vision" and pending:
            self.prefix_stream.wait_event(self.ae_done)
            self.ae_pending = False  # Original vision_only code must not wait a second time.
        self.last_prefix_dependency_gate = gate if pending else "bootstrap_no_ae"
        return super().build_prefix(pool, identity, prepared)


class ControlPolicy(BASE_POLICY):
    def __init__(self, backend, *, prefix_dependency_gate="ae_before_vlm", **kwargs):
        if (backend.selective_overlap != "vision_only" or not backend.await_previous_prefix
                or backend.num_steps != 10 or not kwargs.get("cuda_graphs")
                or getattr(getattr(backend, "graph_core_class", None), "__name__", "") != "PreparedModulationCore"
                or kwargs.get("ae_priority", 0) != 0 or kwargs.get("submission_order") != "ae_first"):
            raise ValueError("Control requires D10, --selective vision_only, --await-previous-prefix, Graph, AE-first, priority0")
        self.prefix_dependency_gate = prefix_dependency_gate
        super().__init__(backend, **kwargs)

    def configure(self, *args, prefix_dependency_gate=None, **kwargs):
        gate = self.prefix_dependency_gate if prefix_dependency_gate is None else prefix_dependency_gate
        if gate not in GATES.values():
            raise ValueError("Unknown Prefix dependency gate")
        owner = self.graphs
        result = super().configure(*args, **kwargs)  # Existing drain/reset; no extra GPU fence.
        if owner is not None and self.graphs is not owner:
            raise RuntimeError("Control must preserve the same Graph owner")
        bench.switch_selective_after_reset(self, "vision_only")
        self.prefix_dependency_gate = self.backend.prefix_dependency_gate = gate
        return result


class ControlRecorder(BASE_RECORDER):
    def predict(self, policy, *args, **kwargs):
        before = len(self.rows)
        response = super().predict(policy, *args, **kwargs)
        if len(self.rows) > before:
            row = self.rows[-1]
            row.update(prefix_dependency_gate=policy.graphs.last_prefix_dependency_gate,
                       early_return_contract="API waits for AE output, not current Prefill; next request awaits previous Prefix",
                       graph_engine_ordinal=self.capture_engines)
            if (self.capture_engines != 1 or row["steady_recapture"] != 0
                    or row["prefix_dependency_gate"] != policy.prefix_dependency_gate
                    or row["selective_kind"] != "vision_only"):
                raise RuntimeError("Control Graph/gate invariant failed")
        return response


def write_results(dest, result):
    snapshot = dest / "source" / "async_serial_control_wrapper.py"
    if not snapshot.exists():
        snapshot.write_bytes(SOURCE)
    unchanged = hashlib.sha256(WRAPPER.read_bytes()).hexdigest() == SOURCE_SHA
    result["wrapper"] = dict(path=str(WRAPPER), sha256=SOURCE_SHA, source_unchanged=unchanged,
        source_snapshot=str(snapshot), gates=GATES,
        scope="Same complete old KV/Graph owner/two slots; no Graph recapture or kernel changes. Control gates all Prefix staging/Vision/VLM after AE; both APIs return after AE output.")
    result["command"] = shlex.join([sys.executable, "-m", MODULE, *sys.argv[1:]])
    result["working_directory"] = os.getcwd()
    if not unchanged:
        result.update(status="excluded_wrapper_source_drift", completed=False, performance_eligible=False)
    return BASE_WRITE(dest, result)


def main():
    def settings(name):
        if name not in GATES:
            raise ValueError("This wrapper accepts graph_async_serial/graph_vision_overlap only")
        return BASE_SETTINGS(name)
    for name, gate in GATES.items():
        bench.MODES[name] = dict(cuda_graphs=True, submission_order="ae_first", prefix_dependency_gate=gate)
    segmented.SelectiveGraphs = ControlGraphs
    bench.settings = settings
    bench.PipelinePolicy, bench.Recorder, bench.write_results = ControlPolicy, ControlRecorder, write_results
    bench.main()


if __name__ == "__main__":
    main()
