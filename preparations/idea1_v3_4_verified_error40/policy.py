from preparations.idea1_v3_4_verified_paths.policy import Policy as Parent
from .model import CompensatedSelector

class Policy(Parent):
    def __init__(self,b):
        super().__init__(b);self.selector.clear();self.selector=CompensatedSelector(b,self)
    def configure(self,config):
        assert config['selection']=='jacobian' and config['projections']==4 and config['compensation']=='extrapolate'
        return super().configure(config)
    def predict(self,*args,**kwargs):
        r=super().predict(*args,**kwargs);r['route'].update(self.selector.record)
        if r['route']['path']=='full' and self.selector.error_rms is not None:r['arrays']['compensation_error_rms']=self.selector.error_rms.copy()
        return r
