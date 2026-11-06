"""Fresh-process-only recovery entrypoint for qk-row-bounds-v1."""
import os

if __name__=='__main__':
    from preparations.idea1_v3.safe_qk import install
    install(os.environ['V3_QK_SAFETY_RECEIPT'])
    from preparations.idea1_v3 import server
    class BoundedPolicy(server.Policy):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.protocol['execution_safety_revision']='qk-row-bounds-v1'
            self.protocol['safety_check_receipt']=os.environ['V3_QK_SAFETY_RECEIPT']
    server.Policy=BoundedPolicy
    server.main()
