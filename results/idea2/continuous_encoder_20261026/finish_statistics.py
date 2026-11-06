"""Wait on the fixed grid, then compute prospective statistics exactly once."""
import json
from pathlib import Path
import subprocess
import time

R=Path(__file__).resolve().parent
ROOT=R.parents[2]
P=ROOT/"reports/idea2_continuous_encoder_20261026"


def main():
    deadline=time.monotonic()+8*3600
    while True:
        path=R/"dispatch_execution.json"
        if path.exists():
            state=json.loads(path.read_text())
            if state["status"]=="passed":break
            if state["status"]!="running":raise RuntimeError("Dispatcher needs review; statistics not computed")
        if time.monotonic()>deadline:raise TimeoutError("Bounded statistics wait expired")
        time.sleep(30)
    P.mkdir(parents=True,exist_ok=True)
    command=["python3",str(R/"statistics.py"),"--input",str(R/"normalized_cases.json"),
             "--output",str(P/"statistics.json"),"--csv-dir",str(P)]
    result=subprocess.run(command,cwd=ROOT,check=False,capture_output=True,text=True)
    (R/"statistics_execution.json").write_text(json.dumps(dict(command=command,exit_code=result.returncode,
        stdout=result.stdout,stderr=result.stderr),indent=2)+"\n")
    if result.returncode:raise RuntimeError(result.stderr)
    value=json.loads((P/"statistics.json").read_text())
    if value["status"]!="complete":raise RuntimeError("Prospective complete-grid validation failed")


if __name__=="__main__":main()
