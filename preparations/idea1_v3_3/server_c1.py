"""Frozen original scientific paths, noise metadata r2, and strict E2 c gate."""
import hashlib
from pathlib import Path
from . import policy,server_r2
from .rules_c1 import gate
from .metadata_r2 import Policy as MetadataPolicy
policy.gate=gate


class Policy(MetadataPolicy):
    def __init__(self,backend):
        super().__init__(backend)
        receipt=Path(__file__).resolve().parents[2]/'results/idea1_v3_3/E2_calibration.json'
        self.protocol.update(E2_gate_revision='strict-c1',E2_c_operator='<',E2_c_quaternion='world EEF body xyzw',
            E2_c_reference='max pairwise shortest-arc angle over last5 completed control steps',
            E2_calibration_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())


server_r2.Policy=Policy
if __name__=='__main__': server_r2.main()
