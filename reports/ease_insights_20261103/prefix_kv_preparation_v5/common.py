"""CPU preparation definitions only;no runnable study or quality manifest."""
from preparations.ease_insights_20261103.camera_paths_v4.common import FULL
PATTERNS={'center128_d2':'center_columns','checker128_d2':'checkerboard'}
CONDITIONS=('full_d10','dense_d2',*PATTERNS)


def conditions():
    return [dict(id=cid,model='pi0',mode='full' if cid=='full_d10' else 'periodic',compensation=None,
        period=2,H=50,R=5,D=10 if cid=='full_d10' else 2,refresh_D=10,skip=[],ae_skip=[],
        vision_kept=[list(FULL),list(FULL)],fresh_cameras=[0,1],vision_cache='none',
        prefix_pattern=PATTERNS.get(cid,'all')) for cid in CONDITIONS]


def validate_condition(config):
    if config not in conditions():raise ValueError('Not a prospective fixed prefix allocation')
