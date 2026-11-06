"""CPU checks that scientific expansion never redispatches source init6 keys."""
import json
from pathlib import Path
import tempfile
import types

from preparations.idea1_v3.advance import build
from preparations.idea1_v3.cli import make_manifest


def main():
    with tempfile.TemporaryDirectory() as temp:
        source=Path(temp)/'initial'; source.mkdir()
        manifest=make_manifest(types.SimpleNamespace(stage='E4',config=None,execution='A',expand=False,shards=1))
        (source/'manifest.json').write_text(json.dumps(manifest))
        for row in manifest['tasks']:
            directory=source/'episodes'/'E4'/row['condition']/'A'/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"
            directory.mkdir(parents=True)
            (directory/'complete.json').write_text(json.dumps(dict(key=row['key'],valid=True)))
            (directory/'result.json').write_text(json.dumps(dict(row,valid=True,success=False,actions=5)))
        args=types.SimpleNamespace(command='e4-expand',source=source,shards=3)
        expansion=build(args)
        assert len(expansion['tasks'])==160 and {x['init'] for x in expansion['tasks']}=={7,8,9,10}
        assert not ({x['key'] for x in expansion['tasks']}&{x['key'] for x in manifest['tasks']})
        assert {x['owner'] for x in expansion['tasks']}=={0,1,2}
        source=Path(temp)/'calibration'; source.mkdir()
        calibration=make_manifest(types.SimpleNamespace(stage='calibration',config=None,execution='A',expand=False,shards=1))
        (source/'manifest.json').write_text(json.dumps(calibration))
        for row in calibration['tasks']:
            c=row['config']; directory=source/'episodes'/'E1'/row['condition']/'A'/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"
            directory.mkdir(parents=True)
            best=(c['draft_steps']==2 and c['tau']==.1 and c['period']==4) or (c['draft_steps']==10 and c['tau']==.15 and c['period']==2)
            (directory/'complete.json').write_text(json.dumps(dict(key=row['key'],valid=True)))
            (directory/'result.json').write_text(json.dumps(dict(row,valid=True,success=best,actions=5)))
            (directory/'infer.jsonl').write_text(json.dumps(dict(timing=dict(policy_wall_ms=10)))+'\n')
        args=types.SimpleNamespace(command='e1-prepare',source=source,shards=2,parent_attempt=None)
        expansion=build(args)
        assert len(expansion['conditions'])==3 and len(expansion['tasks'])==120
        assert {x['init'] for x in expansion['tasks']}=={6}
        args.command='e3-prepare'; history=build(args)
        assert len(history['tasks'])==40 and history['conditions'][0]['config']['tau']==.1
    print('advance E4 new160/no source key overlap; E1 reference+best dedup; E3 frozen D2 parent: OK')

if __name__=='__main__': main()
