"""E3 client wrapper validates the native OSC clipping bounds before execution."""
import numpy as np


def run(*args,**kwargs):
    from libero.libero import envs
    from preparations.idea1_v3 import branch_episode
    original=envs.OffScreenRenderEnv
    def validated_env(*args,**kwargs):
        env=original(*args,**kwargs)
        controller=env.env.robots[0].controller
        if not np.allclose(controller.input_min,-1.) or not np.allclose(controller.input_max,1.):
            env.close()
            raise ValueError('E3 history clipping requires verified native OSC [-1,1] input bounds')
        return env
    envs.OffScreenRenderEnv=validated_env
    try: return branch_episode.run(*args,**kwargs)
    finally: envs.OffScreenRenderEnv=original
