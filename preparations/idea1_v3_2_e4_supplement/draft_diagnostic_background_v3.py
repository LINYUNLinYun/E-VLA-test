"""Version3 background intervention: original strict gates with native MuJoCo texture upload."""
import numpy as np
from .libero_plus_env import ORIGINAL
from .draft_diagnostic_inputs import texture_ownership,background_geom_allowed,observations
from .draft_diagnostic_protocol import hashes
from .draft_diagnostic_texture_upload import upload_texture


def background_intervention(env,case):
    """Only copy matching named background texture pixels from mother BDDL model."""
    from libero.libero.envs.env_wrapper import ControlEnv
    import mujoco
    bddl=ORIGINAL/'libero/libero/bddl_files'/case['mother_suite']/(case['mother_task']+'.bddl')
    mother=ControlEnv(str(bddl),use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,hard_reset=True)
    try:
        model=env.env.sim.model;source=mother.sim.model
        def names(m):
            return {mujoco.mj_id2name(m._model,mujoco.mjtObj.mjOBJ_TEXTURE,i):i for i in range(m.ntex)}
        originals=names(source);changed=[];skipped=[]
        for name,index in names(model).items():
            if name not in originals or not name or not (name.startswith(('texplane','tex-wall','tex-table','tex-living_room_table','tex-coffee','tex-study','tex-kitchen')) or name=='textable'):
                continue
            ownership=texture_ownership(model,index)
            uses=[g for material in ownership for g in material['geoms']]
            if not uses or not all(background_geom_allowed(g['geom_name']) and g['static_body'] for g in uses):
                skipped.append(dict(name=name,reason='texture is unused or shared with non-arena geom',ownership=ownership));continue
            other=originals[name];shape=(int(model.tex_height[index]),int(model.tex_width[index]),int(model.tex_nchannel[index]))
            other_shape=(int(source.tex_height[other]),int(source.tex_width[other]),int(source.tex_nchannel[other]))
            if shape!=other_shape or int(model.tex_type[index])!=int(source.tex_type[other]):
                skipped.append(dict(name=name,reason='texture shape/type mismatch',target_shape=shape,source_shape=other_shape));continue
            size=int(np.prod(shape));start=int(model.tex_adr[index]);source_start=int(source.tex_adr[other])
            target=model.tex_data.reshape(-1)[start:start+size];pixels=source.tex_data.reshape(-1)[source_start:source_start+size]
            if not np.array_equal(target,pixels):
                pixel_count=int(np.sum(target!=pixels));before_hash=hashes(dict(texture=target))['texture']
                target[:]=pixels;upload_texture(env.env.sim,index)
                changed.append(dict(texture_name=name,texture_id=index,source_texture_id=other,shape=shape,
                    changed_channels=pixel_count,before_sha256=before_hash,after_sha256=hashes(dict(texture=target))['texture'],ownership=ownership))
        return observations(env),dict(kind='matching_named_background_texture_bytes_from_mother_BDDL_model',source_bddl=str(bddl),
            changed_textures=changed,skipped_textures=skipped,material_parameters_unchanged=True,
            intervention_nonempty=bool(changed),limitation='Texture-only removal; other material coefficients remain actual Plus values. No whole-scene reload.')
    finally:mother.close()


