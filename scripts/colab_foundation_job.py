"""Launch an explicit research job and retain its exit status outside the kernel."""

import json
import os
from pathlib import Path
import subprocess

root = Path("/content/sae-foundation")
job = json.loads(Path("/content/foundation_job.json").read_text())
name = job["name"]
if not name.replace("_", "").isalnum():
    raise ValueError("Invalid job name")
status = root / f"{name}.status.json"
if status.exists():
    raise ValueError("Job name already used; inspect its status before resuming.")
runner = r"""
import json, os, subprocess, sys, time
from pathlib import Path
job = json.loads(sys.argv[1])
status = Path(sys.argv[2])
record = {"name": job["name"], "state": "running", "pid": os.getpid(), "started": time.time()}
status.write_text(json.dumps(record))
code = 1
try:
    for command in job["commands"]:
        print("RUN", command, flush=True)
        code = subprocess.call(command)
        if code:
            break
finally:
    record.update(state="complete" if code == 0 else "failed", exit_code=code, finished=time.time())
    status.write_text(json.dumps(record))
"""
env = dict(
    os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg"
)
with (root / f"{name}.log").open("w") as log:
    proc = subprocess.Popen(
        ["/content/sae-env/bin/python", "-u", "-c", runner, json.dumps(job), str(status)],
        cwd=root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print("JOB STARTED", name, proc.pid)
