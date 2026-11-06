"""Fresh child-server safety entrypoint for E1/E3 continuation only."""
import os


def main():
    from preparations.idea1_v3.safety_runtime import install_in_model_process,kernel_digest
    install_in_model_process(os.environ['V3_QK_WORKER_RECEIPT'])
    base=os.environ['V3_SAFE_BASE_SERVER']
    def annotate(policy):
        policy.protocol.update(execution_safety_revision='qk-row-bounds-v1',safety_kernel_sha256=kernel_digest(),
            safety_check_receipt=os.environ['V3_QK_WORKER_RECEIPT'])
    if base=='preparations.idea1_v3.server':
        from preparations.idea1_v3 import server
        class SafePolicy(server.Policy):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs); annotate(self)
        server.Policy=SafePolicy
        server.main()
    elif base=='preparations.idea1_v3.branch_server':
        from preparations.idea1_v3 import history,branch_server
        class SafeHistory(history.HistoryPolicy):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs); annotate(self)
        history.HistoryPolicy=SafeHistory
        branch_server.main()
    else: raise ValueError('Only explicit E1/E3 base servers may use safe continuation')

if __name__=='__main__': main()
