from preparations.idea1_v3_4.backend import Backend as OriginalBackend
from preparations.idea1_v3.policy import V14


class Backend(OriginalBackend):
    reuse_allowed=True

    def prepare_hybrid(self,obs,instruction,full,cutoff=16):
        if not self.reuse_allowed:
            c,t=self.prepare(obs,instruction,V14)
            t.update(current_kv_layers=18,old_kv_layers=0)
            return c,t
        return super().prepare_hybrid(obs,instruction,full,cutoff)
