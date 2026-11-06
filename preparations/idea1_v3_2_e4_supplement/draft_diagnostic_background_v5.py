"""Version5 background intervention: original strict gates with native MuJoCo texture upload."""
import numpy as np
from .libero_plus_env import ORIGINAL
from .draft_diagnostic_inputs import texture_ownership,background_geom_allowed,observations
from .draft_diagnostic_protocol import hashes
from .draft_diagnostic_texture_upload import upload_texture
from .draft_diagnostic_arena_proof import ownership_allowed
from .draft_diagnostic_texture_auxiliary import auxiliary_resample


def background_intervention(env,case):
    """Only copy matching named background texture pixels from mother BDDL model."""
    if case["mother_suite"]!="libero_10" or not case["name"].endswith("_table_8"):
        raise ValueError("Auxiliary v5 is registered only for original fixed case252")
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
            if not ownership_allowed(env,name,ownership):
                skipped.append(dict(name=name,reason='texture is unused or shared with non-arena geom',ownership=ownership));continue
            other=originals[name];shape=(int(model.tex_height[index]),int(model.tex_width[index]),int(model.tex_nchannel[index]))
            other_shape=(int(source.tex_height[other]),int(source.tex_width[other]),int(source.tex_nchannel[other]))
            same_type=int(model.tex_type[index])==int(source.tex_type[other])
            source_size=int(np.prod(other_shape));source_start=int(source.tex_adr[other])
            pixels=source.tex_data.reshape(-1)[source_start:source_start+source_size].reshape(other_shape)
            approximate=False
            if shape!=other_shape or not same_type:
                if name=='tex-living_room_table' and same_type and int(model.tex_type[index])==0 and shape==(512,512,3) and other_shape==(1024,1024,3):
                    pixels=auxiliary_resample(pixels,shape,int(model.tex_type[index]));approximate=True
                else:
                    skipped.append(dict(name=name,reason='unregistered texture shape/type mismatch',target_shape=shape,source_shape=other_shape));continue
            size=int(np.prod(shape));start=int(model.tex_adr[index])
            target=model.tex_data.reshape(-1)[start:start+size];pixels=np.ascontiguousarray(pixels).reshape(-1)
            if not np.array_equal(target,pixels):
                pixel_count=int(np.sum(target!=pixels));before_hash=hashes(dict(texture=target))['texture']
                target[:]=pixels;upload_texture(env.env.sim,index)
                changed.append(dict(texture_name=name,texture_id=index,source_texture_id=other,shape=shape,source_shape=other_shape,
                    approximate_mother_appearance=approximate,resampling='PIL LANCZOS1024-to512 RGB2D' if approximate else None,
                    changed_channels=pixel_count,before_sha256=before_hash,after_sha256=hashes(dict(texture=target))['texture'],ownership=ownership))
        return observations(env),dict(kind='matching_named_background_texture_bytes_from_mother_BDDL_model',source_bddl=str(bddl),
            changed_textures=changed,skipped_textures=skipped,material_parameters_unchanged=True,
            intervention_nonempty=bool(changed),limitation='Auxiliary approximate mother appearance at actual target resolution; not exact mother texture restoration. Only registered RGB2D1024-to512 resampling; material/physical properties unchanged.')
    finally:mother.close()




