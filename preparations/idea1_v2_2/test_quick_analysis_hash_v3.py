import csv
import itertools
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from common import ROOT,atomic_json,sha
import runtime_analysis_quick_v2 as old
import runtime_analysis_quick_v3 as new


class HashPerformanceTests(unittest.TestCase):
    def test_table_byte_identical_receipt_hash_once_not_per_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); atomic_json(root/'analysis_input_receipt.json',{'sealed':'source'})
            rows=[dict(index=i,value=i/3) for i in range(200)]
            with patch.object(old,'sha',wraps=sha) as before:
                old.table(root/'old.csv',rows)
            with patch.object(new,'sha',wraps=sha) as after:
                new.table(root/'new.csv',rows)
            self.assertEqual(before.call_count,200); self.assertEqual(after.call_count,1)
            self.assertEqual((root/'old.csv').read_bytes(),(root/'new.csv').read_bytes())

    def test_cached_input_hash_once_then_final_recheck_and_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'offline.json'; atomic_json(path,dict(actions=[[1,2,3]]))
            with patch.object(new,'sha',wraps=sha) as calls:
                snapshot=new.HashSnapshot()
                for _ in range(1000): self.assertEqual(snapshot(path),sha(path))
                self.assertEqual(calls.call_count,1)
                snapshot.verify(); self.assertEqual(calls.call_count,2)
                atomic_json(path,dict(actions=[[4,5,6]]))
                with self.assertRaisesRegex(ValueError,'Sealed analysis input changed'): snapshot.verify()

    def test_actual_saved_csv_summary_values_rows_and_hash_counts_identical(self):
        actual=ROOT/'policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v2_v2_20261019T0917Z/action_error_by_offset_phase.csv'
        if actual.exists():
            with actual.open(newline='') as stream: records=list(itertools.islice(csv.DictReader(stream),12))
            for row in records:
                for key in ('point','offset'): row[key]=int(row[key])
                for key in ('first_five','execution_prefix'): row[key]=row[key]=='True'
                for key in ('translation_l2','rotation_l2','gripper_direction_mismatch','gripper_absolute',
                            *[a+s for a in ('x','y','z','rotation_x','rotation_y','rotation_z') for s in ('_signed','_absolute')]):
                    row[key]=float(row[key])
        else:
            records=[dict(condition='c',reference='full',phase='unknown',point=0,offset=1,first_five=True,execution_prefix=True,
                **{key:.25 for key in ('translation_l2','rotation_l2','gripper_direction_mismatch','gripper_absolute',
                   *[a+s for a in ('x','y','z','rotation_x','rotation_y','rotation_z') for s in ('_signed','_absolute')])})]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); filename=root/'action_error_by_offset_phase.csv'
            with filename.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
            results=[]
            for module in (old,new):
                source=Path(module.__file__).read_text()
                marker='    error_csv_sha256=' if module is new else '    metrics='
                start=source.index(marker); stop=source.index('    failure=[]',start)
                block=textwrap.dedent(source[start:stop]); calls=[]
                def tracked(path): calls.append(Path(path)); return sha(path)
                namespace=dict(root=root,errors=records,np=np,sha=tracked,table=module.table)
                with patch.object(module,'sha',side_effect=tracked):
                    if module is new: namespace['sealed_sha']=new.HashSnapshot()
                    exec(compile(block,module.__file__+':summary-regression','exec'),namespace)
                    initial=len(calls)
                    if module is new: namespace['sealed_sha'].verify()
                results.append((namespace['summaries'],(root/'action_error_summary.csv').read_bytes(),initial,len(calls)))
            self.assertEqual(results[0][0],results[1][0]); self.assertEqual(results[0][1],results[1][1])
            self.assertEqual(results[0][2],len(results[0][0]))
            self.assertEqual(results[1][2],1); self.assertEqual(results[1][3],2)
            self.assertGreater(len(results[1][0]),1)


if __name__=='__main__': unittest.main()
