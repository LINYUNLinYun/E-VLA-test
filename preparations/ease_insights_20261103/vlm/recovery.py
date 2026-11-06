"""Technical-only recovery; completed scientific outcomes are never retried."""
import argparse
import contextlib
import json
import math
from pathlib import Path
import time
import uuid
from preparations.idea1_v3_1.common import lock
from .common import ROOT, atomic_json, digest, read_manifest, sha, cases

SCIENTIFIC_FILES = ("identity.json", "result.json", "complete.json", "infer.jsonl", "telemetry.jsonl", "arrays.npz", "input_0000.npz")


def runtime_revision(manifest):
    files = [Path(__file__).parent/name for name in ("recovery.py", "worker_v2.py", "runner_v2.py")]
    value = dict(name="vlm-technical-receipt-v2", scientific_manifest_hash=manifest["manifest_hash"],
        purpose="Interpret exported schedule-node lists; retain earliest complete scientific outcome; never retry completed science",
        source_files={str(p.relative_to(ROOT)): sha(p) for p in files},
        model_backend_policy_changed=False, scientific_manifest_changed=False)
    value["revision_hash"] = digest(value)
    return value


def scheduled_nfe(nodes):
    if not isinstance(nodes, list) or len(nodes) not in (3,11):
        raise ValueError("Frozen backend must export a D2 or D10 time-node list")
    n = len(nodes)-1
    expected = [1-i/n for i in range(n+1)]
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x != y for x,y in zip(nodes,expected)):
        raise ValueError("Exported time nodes differ from the frozen native Euler schedule")
    return n


class CompletedEpisodeFinalizationError(RuntimeError):
    """The outcome is complete: fix its receipt, do not obtain another outcome."""


def run_with_retries(directory, max_attempts, run_episode, finalize, close, before_attempt=lambda: None):
    directory = Path(directory)
    def finish():
        try:
            return finalize()
        except Exception as exc:
            atomic_json(directory/"finalization_error.json", dict(error=repr(exc), time=time.time(),
                scientific_complete=True, retry_scientific_episode=False))
            raise CompletedEpisodeFinalizationError("Scientific episode complete; receipt repair required") from exc
    # Resume bookkeeping before considering the technical retry budget.
    if (directory/"complete.json").exists():
        return finish()
    errors = len(list(directory.parent.glob(directory.name+".invalid.*/technical_error.json"))) + int((directory/"technical_error.json").exists())
    for attempt in range(errors,max_attempts):
        if directory.exists():
            directory.rename(directory.with_name(directory.name+".invalid."+uuid.uuid4().hex))
        before_attempt()
        try:
            run_episode()
        except Exception as exc:
            if (directory/"complete.json").exists():
                atomic_json(directory/"finalization_error.json",dict(error=repr(exc),time=time.time(),
                    scientific_complete=True,retry_scientific_episode=False))
                raise CompletedEpisodeFinalizationError("Episode completed before cleanup failed; no scientific retry") from exc
            directory.mkdir(parents=True,exist_ok=True)
            atomic_json(directory/"technical_error.json",dict(error=repr(exc),attempt=attempt+1,time=time.time(),scientific_complete=False))
            close()
            if isinstance(exc,ValueError) or attempt+1 >= max_attempts:
                raise
            continue
        # Deliberately outside the retry exception handler.
        return finish()
    raise RuntimeError("Technical attempt budget exhausted")


def recover_case(output, condition, scene, manifest, finalize, already_locked=False):
    output = Path(output)
    canonical = output/condition/"episodes"/scene["suite"]/f"task{scene['task']:02d}"/f"init{scene['init']:02d}"
    context = contextlib.nullcontext() if already_locked else lock(str(canonical)+".lock")
    with context:
        # A prior v2 recovery is immutable; ordinary completed episodes have no ledger.
        ledger_path = canonical/"recovery_receipt.json"
        if ledger_path.exists():
            receipt = json.loads(ledger_path.read_text())
            if receipt["scientific_manifest_hash"] != manifest["manifest_hash"]:
                raise ValueError("Recovery belongs to another scientific manifest")
            for filename, expected in receipt["chosen_scientific_sha256"].items():
                if sha(canonical/filename) != expected:
                    raise ValueError("Recovered scientific data changed")
            return finalize(output,condition,scene,manifest)
        candidates = [canonical]
        candidates += list(canonical.parent.glob(canonical.name+".invalid.*"))
        candidates += list(canonical.parent.glob(canonical.name+".duplicate.*"))
        candidates = [p for p in candidates if p.is_dir() and (p/"complete.json").exists()]
        if not candidates:
            return None
        snapshots = []
        for directory in candidates:
            done = json.loads((directory/"complete.json").read_text())
            result = json.loads((directory/"result.json").read_text())
            if (not done["valid"] or done["manifest_hash"] != manifest["manifest_hash"] or not result["valid"] or
                result["manifest_hash"] != manifest["manifest_hash"] or result["condition"] != condition or result["case"] != scene):
                raise ValueError("Completed attempt has mismatched scientific identity; do not substitute another outcome")
            error_path = directory/"technical_error.json"
            error = json.loads(error_path.read_text()) if error_path.exists() else None
            snapshots.append(dict(original_path=str(directory.resolve()),complete_mtime_ns=(directory/"complete.json").stat().st_mtime_ns,
                result_mtime_ns=(directory/"result.json").stat().st_mtime_ns,
                scientific_sha256={name:sha(directory/name) for name in SCIENTIFIC_FILES if (directory/name).exists()},
                technical_error=error, technical_error_sha256=sha(error_path) if error_path.exists() else None,
                result_success=result["success"],actions=result["actions"],predictions=result["predictions"]))
        # Selection uses completion time only, never success, actions, or latency.
        snapshots.sort(key=lambda row:(row["complete_mtime_ns"],row["original_path"]))
        chosen = snapshots[0]
        chosen_path = Path(chosen["original_path"])
        revision = runtime_revision(manifest)
        ledger_dir = output/"technical_recovery"/condition/scene["suite"]/f"task{scene['task']:02d}"
        pending_path = ledger_dir/f"init{scene['init']:02d}_intent.json"
        intent = dict(scientific_manifest_hash=manifest["manifest_hash"],case=scene,condition=condition,
            rule="Earliest complete.json mtime_ns, with lexical original path only for exact timestamp ties; outcome ignored",
            candidates_before_recovery=snapshots,chosen_original_path=str(chosen_path),canonical_path=str(canonical.resolve()),
            runtime_revision=revision,scientific_inference_performed=False)
        atomic_json(pending_path,intent)
        # Validate and create only administrative receipt before moving anything.
        finalize(output,condition,scene,manifest,directory=chosen_path)
        renamed = []
        if chosen_path != canonical.resolve():
            if canonical.exists():
                duplicate = canonical.with_name(canonical.name+".duplicate."+uuid.uuid4().hex)
                canonical.rename(duplicate)
                renamed.append(dict(before=str(canonical.resolve()),after=str(duplicate.resolve()),classification="duplicate_completed_attempt_not_counted"))
            chosen_path.rename(canonical)
            renamed.append(dict(before=str(chosen_path),after=str(canonical.resolve()),classification="earliest_completed_scientific_result"))
        receipt = dict(intent,status="recovered",chosen_scientific_sha256=chosen["scientific_sha256"],renames=renamed,
            retained_for_analysis=1,duplicate_complete_attempts_not_counted=len(snapshots)-1,
            original_technical_errors="Preserved as receipt/finalization errors after valid science; not scientific failure",
            recovery_time=time.time())
        atomic_json(canonical/"recovery_receipt.json",receipt)
        atomic_json(ledger_dir/f"init{scene['init']:02d}_recovered.json",receipt)
        atomic_json(output/"technical_runtime_revisions"/(revision["revision_hash"]+".json"),revision)
        return finalize(output,condition,scene,manifest)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",required=True); parser.add_argument("--condition",required=True)
    args = parser.parse_args(); output = Path(args.output).resolve()
    manifest = read_manifest(output/"manifest.json",verify_sources=True)
    from .worker_v2 import finalize
    recovered = []
    for scene in cases(manifest,"discovery"):
        value = recover_case(output,args.condition,scene,manifest,finalize)
        if value is not None:
            recovered.append(dict(case=scene,success=value["success"],executed_steps=value["executed_steps"]))
    print(json.dumps(dict(recovered=recovered,scientific_inference_performed=False,
        runtime_revision=runtime_revision(manifest))))


if __name__ == "__main__": main()
