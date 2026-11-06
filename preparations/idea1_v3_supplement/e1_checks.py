"""CPU route contract and strict old-policy differential checks."""
import numpy as np
from preparations.idea1_v3.checks import Backend
from preparations.idea1_v3.policy import Policy as Old
from preparations.idea1_v3_supplement.policy import Policy,acceptance


class Fake(Backend):
    def __init__(self): super().__init__(); self.mode='normal'; self.steps=[]
    def sample(self,c,seed,steps):
        self.steps.append(steps)
        actions,raw,t=super().sample(c,seed,steps)
        if self.mode=='draft_switch' and len(self.visited[-1][1]): raw[2:,6]=1; actions=raw[:,:7].copy()
        return actions,raw,t
    def verify(self,*args):
        hats,t=super().verify(*args)
        if self.mode=='draft_switch': hats[:,:,6]=-1
        if self.mode=='verify_switch': hats[0,2,6]=1
        if self.mode=='confirmed': hats[:,2:,6]=1
        return hats,t


def check():
    config=dict(experiment='E1',draft_steps=2,tau=.10,period=2)
    old=Old(config,Fake()); new=Policy(config,Fake())
    # Fixed fake sequence covers accepted, spatial reject, both switch paths/pending.
    for i,mode in enumerate(['normal','normal','normal','normal','draft_switch','normal','verify_switch','normal']):
        old.backend.mode=new.backend.mode=mode
        old.backend.reject=new.backend.reject=i==7
        a=old.predict(i,'instruction','libero_goal',123); b=new.predict(i,'instruction','libero_goal',123)
        np.testing.assert_array_equal(a['actions'],b['actions']); np.testing.assert_array_equal(a['raw'],b['raw'])
        assert a['execute_length']==b['execute_length'] and a['counts']==b['counts']
        for k,v in a['route'].items(): assert b['route'][k]==v
        assert old.pending_gripper==new.pending_gripper and old.draft_rounds==new.draft_rounds
        old.acknowledge(a['actions'][:a['execute_length']]); new.acknowledge(b['actions'][:b['execute_length']])
    def make(rule='strict',period=None): return Policy(dict(config,period=period,gripper_rule=rule,reject_steps=2),Fake())
    def predict(p,i=1): return p.predict(i,'instruction','libero_goal',123)
    def ack(p,r): p.acknowledge(r['actions'][:r['execute_length']])
    p=make(); r=predict(p); assert r['route']['full_steps']==10; ack(p,r)
    for i in range(10): r=predict(p,i+2); assert r['route']['path']=='v14_verified'; ack(p,r)
    assert p.backend.steps==[10]+[2]*10
    assert all(np.all(k==1) for k,_ in p.backend.fulls)
    p.backend.reject=True; r=predict(p,20)
    assert r['route']['full_reason']=='zero_accept' and r['route']['full_steps']==2 and r['route']['accepted_length']==0
    assert p.step==55 and p.full['state']==20 and r['execute_length']==5
    ack(p,r); assert p.step==60
    p=make(period=2); ack(p,predict(p)); ack(p,predict(p)); ack(p,predict(p)); r=predict(p)
    assert r['route']['full_reason']=='periodic' and r['route']['full_steps']==10
    p=make(); ack(p,predict(p)); p.backend.mode='draft_switch'; r=predict(p)
    assert r['execute_length']==2 and p.pending_gripper; ack(p,r); r=predict(p)
    assert r['route']['full_reason']=='gripper_window' and r['route']['full_steps']==2
    p=make(); ack(p,predict(p)); p.backend.mode='verify_switch'; r=predict(p)
    assert r['route']['full_reason']=='gripper' and r['route']['full_steps']==2
    for rule in ('confirmed_disagreement','off'):
        p=make(rule); ack(p,predict(p)); p.backend.mode='confirmed'; r=predict(p)
        assert r['execute_length']==(2 if rule=='confirmed_disagreement' else 5) and not p.pending_gripper
    draft=np.zeros((50,32)); draft[:,6]=-1; hats=np.stack([draft.copy(),draft.copy()]); hats[0,0,6]=1
    assert acceptance(draft,hats,.1,-1,'confirmed_disagreement')[0]==5
    hats[1,0,6]=1; assert acceptance(draft,hats,.1,-1,'confirmed_disagreement')[0]==0
    p.reset(); assert p.full is None and p.pending is None
    print('PASS strict old differential; none/P counters; all Full reason D; independent KV; confirmed/off; rejection/ack/reset')


if __name__=='__main__': check()
