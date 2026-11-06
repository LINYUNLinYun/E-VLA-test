"""CPU-only terminal statistics; never launches model/simulator work."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

R=Path(__file__).resolve().parent
P=R.parents[2]/"reports/idea2_closedloop200_20261025"


def csv_write(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (P/name).open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def main():
    for i in range(1440):
        value=json.loads((R/"dispatch_execution.json").read_text())
        if value["status"]=="passed":break
        if value["status"]!="running":raise RuntimeError("Dispatcher not complete; final statistics suppressed")
        if i%10==0:print("Final statistics await all 800 cases",flush=True)
        time.sleep(30)
    else:raise TimeoutError("Finite 12-hour result wait exhausted")
    output=P/"statistics.json"
    command=[sys.executable,str(R/"statistics.py"),"--input",str(R/"normalized_cases.json"),"--output",str(output)]
    subprocess.run(command,check=True)
    result=json.loads(output.read_text());assert result["status"]=="complete"
    csv_write("success_rates.csv",result["configurations"])
    csv_write("paired_comparisons.csv",result["comparisons"])
    secondary=result["new192_sensitivity"];assert secondary["status"]=="complete"
    csv_write("new192_success_rates.csv",secondary["configurations"])
    csv_write("new192_paired_comparisons.csv",secondary["comparisons"])
    (R/"statistics_execution.json").write_text(json.dumps(dict(status="passed",command=command,exit_code=0,input_rows=result["received_rows"],gpu_launched=False),indent=2)+"\n")
    print("Final 200-case and new192 statistics complete",flush=True)


if __name__=="__main__":main()
