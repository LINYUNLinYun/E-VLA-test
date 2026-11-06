"""Validated scientific schedules; no model/framework dependency."""
import math

MODEL_ALIASES = {'xr0':'xiaomi-robotic0','dbcogact':'db-cogact','π0':'pi0'}

def normalize_schedule(model_id, schedule=None):
    model_id = MODEL_ALIASES.get(model_id, model_id)
    if model_id not in ('pi0','xiaomi-robotic0','db-cogact'):
        raise ValueError(f'Unknown model {model_id}')
    if schedule is None:
        schedule = dict(kind='native', m=None, num_steps=5 if model_id=='xiaomi-robotic0' else 10)
    if set(schedule) - {'kind','m','num_steps'}:
        raise ValueError('Unknown schedule fields')
    kind, m, n = schedule.get('kind'), schedule.get('m'), schedule.get('num_steps')
    if isinstance(n,bool) or not isinstance(n,int) or n <= 0:
        raise ValueError('num_steps must be a positive integer')
    if kind in ('flow','ddim_custom'):
        if isinstance(m,bool) or not isinstance(m,(float,int)) or not math.isfinite(m) or not 0 < m < 1 or n != 2:
            raise ValueError('Custom schedules require 0<m<1 and exactly two updates')
        if (kind=='ddim_custom') != (model_id=='db-cogact'):
            raise ValueError('Flow and DDIM schedules cannot be interchanged')
        m=float(m)
    elif kind=='native':
        if m is not None:
            raise ValueError('Native schedule has no m')
        if n not in ((2,10) if model_id=='db-cogact' else ((2,5) if model_id=='xiaomi-robotic0' else (2,10))):
            raise ValueError('Unsupported native control')
    else:
        raise ValueError('Unknown schedule kind')
    return dict(kind=kind,m=m,num_steps=n)

def flow_schedule(model_id, m=None, num_steps=2):
    if model_id not in ('pi0','xiaomi-robotic0'):
        raise ValueError('Not a flow model')
    q = [1.,float(m),0.] if m is not None else [1-i/num_steps for i in range(num_steps+1)]
    if m is not None and not 0 < m < 1:
        raise ValueError('0<m<1 required')
    nodes = q if model_id=='pi0' else [1-v for v in q]
    return dict(q=q,timesteps=nodes[:-1],nodes=nodes,dt=[b-a for a,b in zip(nodes,nodes[1:])],NFE=len(nodes)-1)

def ddim_indices(t_hi,m):
    mid=round(float(m)*t_hi)
    if not 0 < mid < t_hi:
        raise ValueError('Rounded intermediate DDIM index must be strictly interior')
    return [t_hi,mid]
