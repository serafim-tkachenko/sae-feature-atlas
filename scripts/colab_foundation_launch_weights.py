"""Start the bounded resource download as a logged subprocess."""

import subprocess
from pathlib import Path

root = Path("/content/sae-foundation")
log = (root / "weights_download.log").open("w")
proc = subprocess.Popen(
    ["/content/sae-env/bin/python", "-u", "/content/colab_foundation_weights.py"],
    cwd=root,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("WEIGHTS_DOWNLOAD_PID", proc.pid)
