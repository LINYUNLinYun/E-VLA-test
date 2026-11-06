"""Causal routing features from completed control commands only."""
import numpy as np
from preparations.idea1_v3_3.rules_c1 import gate as old_gate,pose_angle


def feature_after_observe(kalman,executed,prior_forecast):
    z=np.clip(np.asarray(executed,float)[:3],-1,1)
    return dict(innovation_rms=float(np.linalg.norm(kalman.innovation)/np.sqrt(3)),
                forecast_error_rms=float(np.linalg.norm(z-prior_forecast)/np.sqrt(3)))


def metrics(commands,quaternions,history):
    if len(commands)<5 or len(quaternions)<5 or len(history)<5:return dict(ready=False)
    a=np.asarray(commands[-5:],float);norm=np.linalg.norm(a[:,:3],axis=1);turn=[]
    for i in range(4):
        if max(norm[i:i+2])<1e-4:turn.append(0.)
        elif min(norm[i:i+2])<1e-4:turn.append(180.)
        else:turn.append(float(np.degrees(np.arccos(np.clip(np.dot(a[i,:3],a[i+1,:3])/(norm[i]*norm[i+1]),-1,1)))))
    h=history[-5:]
    return dict(ready=True,a_turn_degrees=max(turn),b_command_change=float(np.linalg.norm(np.diff(a[:,:3],axis=0),axis=1).max()),
        c_world_pose_degrees=pose_angle(quaternions[-5:]),innovation_rms_max5=max(x['innovation_rms'] for x in h),
        forecast_error_rms_max5=max(x['forecast_error_rms'] for x in h),
        forecast_error_relative_max5=max(x['forecast_error_rms'] for x in h)/max(.02,float(np.sqrt(np.mean(a[:,:3]**2)))),
        rotation_command_change=float(np.linalg.norm(np.diff(a[:,3:6],axis=0),axis=1).max()),gripper_switch=bool(np.any(np.diff(a[:,6]<0))))


def common_blockers(commands,quaternions):
    if len(commands)<5 or len(quaternions)<5:return ['history']
    a=np.asarray(commands[-5:],float)
    if a.shape!=(5,7) or not np.isfinite(a).all():return ['commands_invalid']
    blocked=[]
    if np.any(np.diff(a[:,6]<0)):blocked.append('gripper')
    if np.linalg.norm(np.diff(a[:,3:6],axis=0),axis=1).max()>.05:blocked.append('rotation_command')
    return blocked


def decide(config,commands,quaternions,history):
    if config['kind']=='abc':
        allowed,blocked=old_gate(commands,quaternions,config['angle'],config['change'],config['pose'])
        return allowed,blocked,dict(runtime_new_metric_collection=False)
    if config['kind']=='alternate':
        allowed=len(commands)>=5
        return allowed,[] if allowed else ['history'],dict(runtime_new_metric_collection=False)
    blocked=common_blockers(commands,quaternions)
    if 'commands_invalid' in blocked:return False,blocked,dict(ready=False)
    if len(commands)<5 or len(quaternions)<5 or len(history)<5:return False,list(dict.fromkeys(blocked+['history'])),dict(ready=False)
    a=np.asarray(commands[-5:],float);error=max(x['forecast_error_rms'] for x in history[-5:]);pose=pose_angle(quaternions[-5:])
    value=error if config['metric']=='forecast_error_rms_max5' else error/max(.02,float(np.sqrt(np.mean(a[:,:3]**2))))
    m=dict(ready=True,forecast_error_rms_max5=error,c_world_pose_degrees=pose,metric=config['metric'],metric_value=value)
    if pose is None:blocked.append('c_invalid')
    elif pose>=config['pose']:blocked.append('c_pose')
    if not np.isfinite(value):blocked.append('metric_invalid')
    elif value>config['threshold']:blocked.append(config['metric'])
    return not blocked,blocked,m
