"""Fresh safe E1 supplement server, never the frozen unsafe entrypoint."""
import os


def main():
    from preparations.idea1_v3.safety_runtime import install_in_model_process,kernel_digest
    install_in_model_process(os.environ['V3_QK_WORKER_RECEIPT'])
    from preparations.idea1_v3 import server
    from preparations.idea1_v3_supplement.policy import Policy
    class SafePolicy(Policy):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.protocol.update(execution_safety_revision='qk-row-bounds-v1',safety_kernel_sha256=kernel_digest(),safety_check_receipt=os.environ['V3_QK_WORKER_RECEIPT'])
    server.Policy=SafePolicy
    server.main()


if __name__=='__main__': main()
