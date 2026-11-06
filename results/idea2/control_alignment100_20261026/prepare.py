"""Freeze the expanded grid and compatible reuse before GPU execution."""
import hashlib
import json
from pathlib import Path

from results.idea2.control_alignment_20261026.run_async import save, sources, SUITES

R = Path(__file__).resolve().parent
ROOT = R.parents[2]
PILOT = ROOT/"results/idea2/control_alignment_20261026"


def main():
    if (R/"manifest.json").exists():
        raise ValueError("Manifest already frozen")
    core = sources()
    reuse = []
    for stage in ("async_gate0", "async_gate1"):
        path = PILOT/stage/"execution.json"
        value = json.loads(path.read_text())
        if value["status"] != "passed" or not value["source_unchanged"] or value["server_exit_code"] != 0:
            raise ValueError("Pilot provenance failed")
        for name, digest in value["source_before"].items():
            if name.endswith("control_alignment_20261026/run_async.py"):
                archive = json.loads((PILOT/"source_archive.json").read_text())
                if digest != archive["sha256"]:
                    raise ValueError("Pilot launcher revision mismatch")
            elif core.get(name) != digest:
                raise ValueError("Pilot computational source changed: "+name)
        for c in value["cases"]:
            if c["status"] != "valid" or not c["result"]["valid"]:
                raise ValueError("Invalid pilot cell")
            reuse.append(dict(configuration=c["alignment"], suite=c["suite"], task=c["task"], init=c["init"],
                status="valid", success=c["result"]["success"], steps=c["result"]["steps"],
                directory=c["directory"], source_kind="pilot", gpu=3, source_execution=str(path), result=c["result"]))
    reused = {(x["configuration"],x["suite"],x["task"],x["init"]):x for x in reuse}
    if len(reused) != 16:
        raise ValueError("Expected sixteen unique pilot cells")
    rows = []
    chunks = []
    for task in range(10):
        pending = []
        for suite in SUITES:
            for init in range(3 if task%2 == 0 else 2):
                pair = []
                for mode in ("off", "control_steps"):
                    key = mode, suite, task, init
                    row = dict(configuration=mode, suite=suite, task=task, init=init,
                               status="pending", source_kind="new", case_key=f"{mode}/{suite}/{task}/{init}")
                    row.update(reused.get(key, {}))
                    rows.append(row)
                    pair.append(row)
                if any(r["status"] != "valid" for r in pair):
                    if any(r["status"] == "valid" for r in pair):
                        raise ValueError("Only whole paired pilot cases may be reused")
                    pending.append(f"{suite}/{task}/{init}")
        if pending:
            chunks.append(dict(chunk_id=task, case_keys=pending, episodes=2*len(pending), status="pending"))
    historical = json.loads((ROOT/"results/idea2/closedloop200_20261025/normalized_cases.json").read_text())["cases"]
    grid = {(r["suite"],r["task"],r["init"]) for r in rows}
    synchronous = [r for r in historical if r["configuration"] in ("fresh18", "sm16_tail0") and (r["suite"],r["task"],r["init"]) in grid]
    if len(synchronous) != 200 or any(r["status"] != "valid" for r in synchronous):
        raise ValueError("Incomplete synchronous historical reference")
    save(R/"synchronous_reference.json", dict(scope="same100 identities, original synchronous protocol; descriptive cross-protocol reference",
        cases=synchronous, successes={a:sum(r["success"] for r in synchronous if r["configuration"] == a) for a in ("fresh18", "sm16_tail0")}))
    save(R/"case_universe.json", rows)
    save(R/"pilot_reuse.json", dict(status="passed", cells=reuse, computational_sources_unchanged=True,
        launcher_exception="post-run environment-log whitelist only; measured version archived"))
    save(R/"manifest.json", dict(status="frozen", episodes=200, cases_per_configuration=100,
        configurations=["off", "control_steps"], pilot_reused_episodes=16, new_episodes=184,
        primary="paired control_steps minus off under the same continuous control protocol",
        model="full pi0.5 LIBERO BF16 B1 D10 H10 R5 lead3 AE16/Prefix112, next Encoder enabled",
        control="exact pilot client/server; only expanded case grid, device assignment and launch ordering",
        cases=[dict(suite=s,task=t,init=i) for s,t,i in sorted(grid)], chunks=chunks,
        case_universe_sha256=hashlib.sha256((R/"case_universe.json").read_bytes()).hexdigest(),
        source_sha256=core, gpu_candidates=[3,2], max_parallel_services=2,
        failure_policy="valid task and horizon failures retained; technical failure stops dispatcher for bounded diagnosis"))
    print(json.dumps(dict(episodes=len(rows), reused=len(reuse), new=184, chunks=len(chunks))))


if __name__ == "__main__":
    main()
