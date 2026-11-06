"""Lazy LIBERO adapter: captures online snapshots and preserves the existing rollout."""
import sys
from .core import REPO

def run(args):
    sys.path.insert(0,str(REPO/'scripts/spec'))
    sys.path.insert(0,str(REPO/'scripts/experiment6'))
    import spec_client_libero as client
    import trajectory_replay as replay
    history=[]
    snapshot=replay.snapshot
    def capture(*a,**kw):
        record=snapshot(*a,**kw)
        raw=a[0].env
        if record['step']==0:
            controller=raw.robots[0].controller
            controller_info={name:client.np.asarray(getattr(controller,name)).tolist()
                             for name in ('input_min','input_max','output_min','output_max')}
            record['controller']=controller_info
            record['control_frequency_hz']=float(raw.control_freq)
            expected=client.np.asarray([.05,.05,.05,.5,.5,.5])
            if (float(raw.control_freq)!=20 or
                not client.np.allclose(controller_info['output_max'],expected,rtol=0,atol=1e-10) or
                not client.np.allclose(controller_info['output_min'],-expected,rtol=0,atol=1e-10) or
                not client.np.allclose(controller_info['input_max'],1,rtol=0,atol=1e-10) or
                not client.np.allclose(controller_info['input_min'],-1,rtol=0,atol=1e-10)):
                raise RuntimeError('Simulator controller differs from frozen command units/20Hz')
        # Legacy snapshots convert contact failures into False. For online routing,
        # unresolved evidence must remain unknown so that Full handles the round.
        for name in record['primary_objects']:
            try:
                gripper=raw.robots[0].gripper;obj=raw.get_object(name)
                record['robot_contacts'][name]=bool(raw.check_contact(gripper,obj))
                record['robot_grasps'][name]=bool(raw._check_grasp(gripper=gripper,object_geoms=obj))
            except Exception:
                record['robot_contacts'][name]=None;record['robot_grasps'][name]=None
        if record['step']==0:
            history.clear()
        # No semantic_phase labels or future trajectory enter the policy.
        history.append(record.copy())
        return record
    replay.snapshot=capture
    original=client._websocket_client_policy.WebsocketClientPolicy
    class OnlineClient(original):
        def infer(self,element):
            if not history:
                raise RuntimeError('Online initial state is missing')
            element=dict(element,__v2_states__=[dict(r) for r in history],__v2_episode_attempt__=args.episode_attempt,
                         __v2_global_step__=history[-1]['step'])
            result=super().infer(element)
            if result.get('accepted_prefix_len')!=5 or result['policy_timing'].get('denoise_steps')!=3:
                raise RuntimeError('Server violates frozen D3/5')
            return result
    client._websocket_client_policy.WebsocketClientPolicy=OnlineClient
    original_record=client._make_infer_record
    def record(*a,**kw):
        result=original_record(*a,**kw)
        result['idea1_v2']=dict(kw['policy_timing'])
        return result
    client._make_infer_record=record
    video=client.imageio.mimwrite
    def encode(path,frames,**kw):
        kw.update(codec='libx264',pixelformat='yuv420p',macro_block_size=2,
                  ffmpeg_params=['-crf','20','-movflags','+faststart'])
        return video(path,frames,**kw)
    client.imageio.mimwrite=encode
    root=args.output/'closed_loop'/args.condition/args.suite/('task%02d'%args.task)/('init%02d'%args.init)/args.episode_attempt
    if root.exists():
        raise FileExistsError('Episode attempt already exists')
    root.parent.mkdir(parents=True,exist_ok=True)
    client.eval_libero(client.Args(host='127.0.0.1',port=args.port,replan_steps=5,task_suite_name=args.suite,
                                  num_steps_wait=10,num_trials_per_task=1,episode_index_start=args.init,
                                  episode_index_end=args.init+1,record_semantic_telemetry=True,task=str(args.task),
                                  video_out_path=str(root.parent),run_name=root.name,seed=7))
