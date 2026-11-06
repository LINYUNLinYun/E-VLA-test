"""Exact XML+runtime proof for case252's unnamed static living-room table visual."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
from .libero_plus_env import PLUS
from .draft_diagnostic_inputs import background_geom_allowed

EXPECTED=PLUS/'libero/libero/assets/scenes/liv_table_Cobblestone01_NRM_6K.xml'


def xml_proof(path):
    path=Path(path)
    if path.resolve()!=EXPECTED.resolve():return None
    tree=ET.parse(path);root=tree.getroot()
    parent=root.find("./worldbody/body[@name='living_room_table']")
    body=parent.find("./body[@name='living_room_table_col']") if parent is not None else None
    material=root.find("./asset/material[@name='living_room_table']")
    mesh=root.find("./asset/mesh[@name='living_room_table_vis']")
    texture=root.find("./asset/texture[@name='tex-living_room_table']")
    if body is None or material is None or mesh is None or texture is None:return None
    if parent.findall('./joint') or body.findall('./joint'):return None
    if material.get('texture')!='tex-living_room_table':return None
    if mesh.get('file')!='living_room_table/visual/living_room_table_vis.msh':return None
    matches=[g for g in body.findall('./geom') if g.get('name') is None and
        all(g.get(k)==v for k,v in dict(type='mesh',mesh='living_room_table_vis',material='living_room_table',group='1',contype='0',conaffinity='0').items())]
    if len(matches)!=1:return None
    meshpath=path.parent/mesh.get('file')
    if not meshpath.is_file():raise RuntimeError('Proven arena mesh asset missing')
    return dict(xml_path=str(path),xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        geom_xml=ET.tostring(matches[0],encoding='unicode'),mesh_path=str(meshpath),mesh_sha256=hashlib.sha256(meshpath.read_bytes()).hexdigest(),
        material='living_room_table',texture='tex-living_room_table',body_chain=['living_room_table_col','living_room_table'])


def ownership_allowed(env,texture_name,ownership):
    import mujoco
    model=env.env.sim.model;uses=[(mat,g) for mat in ownership for g in mat['geoms']]
    if not uses:return False
    for material,geom in uses:
        if not geom['static_body']:return False
        if background_geom_allowed(geom['geom_name']):continue
        if not (geom['geom_name'] is None and geom['body_name']=='living_room_table_col' and
                material['material_name']=='living_room_table' and texture_name=='tex-living_room_table'):
            return False
        index=geom['geom_id'];body=geom['body_id'];parent=int(model.body_parentid[body]);mesh_id=int(model.geom_dataid[index])
        if not (model.body_id2name(parent)=='living_room_table' and
            int(model.geom_type[index])==int(mujoco.mjtGeom.mjGEOM_MESH) and mesh_id>=0 and
            mujoco.mj_id2name(model._model,mujoco.mjtObj.mjOBJ_MESH,mesh_id)=='living_room_table_vis' and
            int(model.geom_group[index])==1 and int(model.geom_contype[index])==0 and int(model.geom_conaffinity[index])==0):
            return False
        arena=Path(env.env.env._arena_xml)
        if not arena.is_absolute():arena=PLUS/'libero/libero/assets'/arena
        proof=xml_proof(arena)
        if proof is None:return False
        geom['exact_arena_xml_proof']=proof
    return True
