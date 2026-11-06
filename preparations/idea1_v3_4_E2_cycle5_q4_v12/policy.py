from preparations.idea1_v3_4_E2_cycle5_spaced.policy import Policy as Parent
from .model import Selector

class Policy(Parent):
    def __init__(self,b):self.b=b;self.selector=Selector(b);self.config={};self.reset()
    def configure(self,config):
        assert config['method']=='jacobian' and config['budget']==12 and config['projections'] in (1,4)
        self.selector.q=config['projections']
        return super().configure(config)
    def predict(self,*args,**kwargs):
        r=super().predict(*args,**kwargs);full=r['route']['path']=='full'
        r['route'].update(configured_projections=self.config['projections'],steady_vjp_count=self.config['projections'] if full else 0,selector_forward_count=int(full),scoring_mode='original_residual',score_aggregation='projection_rms',projection_seed=340610)
        return r
