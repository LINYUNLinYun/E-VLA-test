from preparations.idea1_v3_4_E2_cycle3.policy import Policy as Parent
from .model import Selector

class Policy(Parent):
    def __init__(self,b):self.b=b;self.selector=Selector(b);self.config={};self.reset()
    def predict(self,*args,**kwargs):
        result=super().predict(*args,**kwargs)
        result['route'].update(jacobian_model_steps=10,jacobian_output_steps=[1,2,3,4,5],jacobian_full_layers=18)
        return result
