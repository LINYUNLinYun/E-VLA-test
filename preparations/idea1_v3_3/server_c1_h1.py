"""C1 scientific policy unchanged; corrected history-matched probe metadata."""
import hashlib
from pathlib import Path
from . import server_c1,cross_restart
from .cross_restart_history_h1 import check

cross_restart.check=check


class Policy(server_c1.Policy):
    def __init__(self,backend):
        super().__init__(backend)
        receipt=Path(__file__).resolve().parents[2]/'results/idea1_v3_3/bounded_probe_history_h1.json'
        self.protocol.update(probe_history_revision='cross-restart-fixed-new-h1',probe_history_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())


server_c1.server_r2.Policy=Policy
if __name__=='__main__':server_c1.server_r2.main()
