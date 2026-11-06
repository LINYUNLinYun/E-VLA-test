"""Nonempty physical-artifact regression for raw ledger vs display directory identity."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from common import ROOT, SUITES, atomic_json, planned, sha
import quick_stage_scope_v2 as scope
import runtime_analysis_quick_v2 as analysis
from benchmark_revision_analysis import install_analysis_route


class DirectoryTests(unittest.TestCase):
    def test_real_build_load_route_audit_relative_absolute_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'attempt'; root.mkdir()
            manifest=dict(conditions=[dict(id=f'c{i}',model='pi0',extension=False,
                schedule=dict(kind='flow',m=.5,num_steps=2),mask='full') for i in range(54)],
                protocol_hashes={'pi0':'protocol'},manifest_hash='canonical',
                native_action_limits=dict(limits={suite:520 for suite in SUITES}))
            atomic_json(root/'condition_manifest.json',manifest)
            atomic_json(root/'protocols/pi0.json',dict(H=2,R=1))
            entries=[]
            for n,row in enumerate(r for r in planned(manifest) if r['init'] in (6,7,8)):
                token=f't{n}'; directory=root/'episodes'/row['condition']/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"/token
                result=dict(model=row['model'],suite=row['suite'],task=row['task'],init=row['init'],success=n%2==0,
                    valid=True,kind='formal',video='video.mp4',actions=1,frames=2,H=2,R=1,directory=str(directory))
                atomic_json(directory/'result.json',result)
                validated=dict(result,episode_key=row['key'],video_verified=True,
                    artifacts={'result.json':sha(directory/'result.json'),'video.mp4':'saved-video-sha'})
                atomic_json(directory/'validated.json',validated)
                entry=dict(row,status='valid',token=token,pid=123,start_ticks=456,success=result['success'],
                    directory=os.path.relpath(directory,ROOT) if n==0 else str(directory),
                    validated_sha256=sha(directory/'validated.json'))
                entries.append(entry)
            snapshot=root/'saved.json'; atomic_json(snapshot,dict(entries=entries))
            receipt=root/'drain.json'
            atomic_json(receipt,dict(schema=1,receipt_type='quick_scope_formal_drain_validation',attempt=str(root),
                drain_snapshot=dict(path='saved.json',sha256=sha(snapshot),bytes=snapshot.stat().st_size,
                    entries=6480,valid_rows=6480,reserved_rows=0,canonical_ledger_lock='episode_ledger.lock'),
                frozen_manifest=dict(sha256=sha(root/'condition_manifest.json')),
                quick_scope=dict(inits=[6,7,8],valid_keys=6480),preserved_supplemental_scope=dict(valid_rows_outside_quick_scope=0),
                formal_sessions=[dict(model=model,shard=shard,session_id=n,exit_code=0) for n,(model,shard) in enumerate((('pi0',0),('pi0',1),('db-cogact',0),('db-cogact',1)))],
                process_and_lease_checks=dict(all_eight_worker_server_pids_absent_after_sessions_terminated=True,
                    all_four_model_shard_locks_nonblocking_acquired_then_released=True,
                    all_four_gpu_locks_nonblocking_acquired_then_released=True)))
            view=Path(temp)/'view'; view.mkdir()
            (view/'condition_manifest.json').symlink_to(root/'condition_manifest.json')
            (view/'episode_ledger.json').symlink_to(snapshot)
            (view/'episodes').symlink_to(root/'episodes',target_is_directory=True)
            (view/'protocols').symlink_to(root/'protocols',target_is_directory=True)
            unrelated=Path(temp)/'unrelated/a/b/c/d/e/f'; unrelated.mkdir(parents=True)
            cwd=Path.cwd()
            try:
                os.chdir(unrelated)
                built=scope.build(root,snapshot,receipt)
                first=entries[0]; second=entries[1]
                self.assertEqual(built['valid_records'][first['key']]['directory'],first['directory'])
                self.assertFalse(Path(first['directory']).is_absolute())
                self.assertTrue(Path(second['directory']).is_absolute())
                atomic_json(view/'quick_stage_scope.json',built)
                scope.load_scope(view)
                original=install_analysis_route(analysis,view,root)
                try:
                    completion,valid=analysis.audit(view,strict=False,check_outputs=False)
                    self.assertEqual(len(valid),6480)
                    self.assertFalse(completion['complete'])  # No benchmark/offline fixtures: not a false seal.
                    self.assertTrue(valid[first['key']]['directory'].startswith(str(view/'episodes')))
                    self.assertEqual(Path(valid[first['key']]['directory']).resolve(),Path(built['valid_records'][first['key']]['resolved_directory']))
                    self.assertEqual(Path(valid[second['key']]['directory']).resolve(),Path(second['directory']))
                    (view/'episodes').unlink(); (view/'episodes').symlink_to(Path(temp)/'wrong',target_is_directory=True)
                    with self.assertRaises(ValueError): analysis.audit(view,strict=False,check_outputs=False)
                    (view/'episodes').unlink(); (view/'episodes').symlink_to(root/'episodes',target_is_directory=True)
                finally:
                    analysis.audit=original
                original_record=dict(built['valid_records'][first['key']])
                for change in ({'resolved_directory':str(root/'episodes/wrong-token')},{'pid':999},
                               {'directory':str(root/'episodes/wrong-token')}):
                    built['valid_records'][first['key']]={**original_record,**change}
                    atomic_json(view/'quick_stage_scope.json',built)
                    with self.assertRaises(ValueError): scope.load_scope(view)
            finally:
                os.chdir(cwd)


if __name__=='__main__': unittest.main()
