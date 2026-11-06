from preparations.idea1_v3_4.policy import Policy as OriginalPolicy


def reuse_this_round(config,draft_rounds):
    return not config.get('first_only',False) or draft_rounds==0


class Policy(OriginalPolicy):
    def predict(self,obs,instruction,suite,seed):
        allowed=reuse_this_round(self.config,self.draft_rounds)
        self.backend.reuse_allowed=allowed
        result=super().predict(obs,instruction,suite,seed)
        if result['route']['path']!='full':
            if not allowed: result['route'].update(path='v14',cutoff=None)
            result['route']['old_kv_used']=allowed
        return result
