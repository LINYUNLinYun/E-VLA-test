"""Formal100 registration v2 adds final pre-CUDA idle guard; no scientific change."""
import argparse
import json
from pathlib import Path
from preparations.idea1_v3_2.transfer_protocol import digest,sha
from .policy_calvin_manifest import freeze,ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True);p.add_argument("--deadline-unix",type=float,required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    result=freeze(a.deadline_unix);result.pop("manifest_hash")
    for path in (Path(__file__),Path(__file__).with_name("policy_calvin_server.py")):
        result["source_files"][str(path.relative_to(ROOT))]=sha(path)
    result["allocation_safety"]="Initial GPU-idle guard plus immediate recheck before first native model.to(cuda), after CPU assembly; foreign allocation is explicit stop"
    result["server_entrypoint"]="preparations.idea1_v3_2_e4_supplement.policy_calvin_server"
    result["manifest_hash"]=digest(result)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(output),manifest_hash=result["manifest_hash"],cpu_only=True,gpu_used=False)))


if __name__=="__main__":main()
