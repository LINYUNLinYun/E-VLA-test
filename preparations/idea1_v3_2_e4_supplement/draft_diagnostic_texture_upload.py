"""MuJoCo texture upload using native model, bypassing the broken wrapper helper."""
def upload_texture(sim,texture_id,mujoco_module=None):
    if mujoco_module is None:
        import mujoco as mujoco_module
    context=sim._render_context_offscreen
    if context is None:raise RuntimeError('An existing offscreen render context is required')
    native=sim.model._model
    if context.model._model is not native:raise RuntimeError('Render context does not own the target native model')
    if not 0<=int(texture_id)<int(sim.model.ntex):raise ValueError('Texture ID out of model range')
    context.gl_ctx.make_current()
    mujoco_module.mjr_uploadTexture(native,context.con,int(texture_id))
