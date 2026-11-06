"""Synthetic SQLite/interval tests only; imports no torch and uses no GPU."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from preparations.idea2.layer_trace import (analyze, classify_labels, compare_signatures,
                                            is_request, overlap_metrics, section_group)


PID=1<<24
TID=PID+7


def kernel(begin,end,frame,stream,green=0,context=1):
    return dict(start=begin,end=end,frame_id=str(frame),process=PID,device=0,
                stream=stream,green_context=green,context=context)


class Intervals(unittest.TestCase):
    def test_cross_frame_and_green_contexts(self):
        # Overlapping AE nodes must be unioned, not summed. Green IDs differ
        # while both groups use the same physical GPU.
        ae=[kernel(0,10_000_000,1,10,1),kernel(2_000_000,8_000_000,1,10,1)]
        pre=[kernel(3_000_000,6_000_000,1,11,2),kernel(7_000_000,12_000_000,2,11,2)]
        value=overlap_metrics(ae,pre)
        self.assertEqual(value['left_busy_ms'],10)
        self.assertEqual(value['right_busy_ms'],8)
        self.assertEqual(value['kernel_union_ms'],12)
        self.assertEqual(value['overlap_ms'],6)
        self.assertEqual(value['same_frame_overlap_ms'],3)
        self.assertEqual(value['cross_frame_overlap_ms'],3)

    def test_same_and_cross_subsets_can_coexist(self):
        value=overlap_metrics([kernel(0,10,1,1)],
                              [kernel(0,10,1,2),kernel(0,10,2,3)])
        self.assertEqual(value['overlap_ms'],10/1e6)
        self.assertEqual(value['same_and_cross_simultaneous_ms'],10/1e6)
        self.assertEqual(sum(x['overlap_ms'] for x in value['frame_pairs']),20/1e6)

    def test_different_devices_and_same_stream_do_not_count(self):
        other=kernel(0,10,1,2);other['device']=1
        value=overlap_metrics([kernel(0,10,1,1)],[other,kernel(0,10,1,1)])
        self.assertEqual(value['overlap_ms'],0)
        self.assertEqual(value['same_stream_interval_overlap_ms'],10/1e6)

    def test_labels_keep_iteration_and_exclude_projections(self):
        item=classify_labels(['AE/layer_03/q_proj','AE/layer_03/Attention',
                              'AE/iteration_2','AE/Denoising[same_frame_KV]'])
        self.assertEqual((item['branch'],item['iteration'],item['layer'],item['operation']),('AE',2,3,'Q'))
        item=classify_labels(['Prefill/VLM/layer_03/Attention','Prefill/VLM'])
        self.assertEqual(item['operation'],'Attention_non_projection')


class SQLiteMapping(unittest.TestCase):
    def fixture(self,path,orphan=False,cycle=False):
        db=sqlite3.connect(path)
        db.executescript('''
        create table StringIds(id integer,value text);
        create table NVTX_EVENTS(start integer,end integer,globalTid integer,text text,textId integer);
        create table CUDA_GRAPH_NODE_EVENTS(start integer,end integer,globalTid integer,graphNodeId integer,originalGraphNodeId integer);
        create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer,nameId integer);
        create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,deviceId integer,contextId integer,greenContextId integer,streamId integer,correlationId integer,globalPid integer,demangledName integer,graphNodeId integer);
        ''')
        db.executemany('insert into StringIds values(?,?)',[(1,'cudaGraphLaunch'),(2,'tiny_gemm')])
        ranges=[(-100,-10,TID,'AE/Denoising[same_frame_KV]',None),
                (-90,-20,TID,'AE/iteration_1',None),(-80,-30,TID,'AE/layer_00/q_proj',None),
                (90,150,TID,'layer/continuous/overlap/frame_0/slot_0/gen_1',None),
                (100,120,TID,'submit/ae/frame=7/epoch=2/slot=0/gen=1',None)]
        db.executemany('insert into NVTX_EVENTS values(?,?,?,?,?)',ranges)
        # Clone creation entries must NOT shadow their ancestry links.
        db.executemany('insert into CUDA_GRAPH_NODE_EVENTS values(?,?,?,?,?)',[
            (-50,-50,TID,10,None),(-5,-5,TID,20,None),(-5,-5,TID,20,10),
            (-1,-1,TID,30,None),(-1,-1,TID,30,20)])
        if cycle:db.execute('insert into CUDA_GRAPH_NODE_EVENTS values(?,?,?,?,?)',(-1,-1,TID,10,30))
        db.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?,?)',(105,110,TID,99,1))
        db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?,?,?)',
                   (200,210,0,1,3,8,99,PID,2,999 if orphan else 30))
        db.commit();db.close()

    def test_recursive_clone_and_submission_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.sqlite';self.fixture(path)
            report,rows=analyze(path)
        self.assertTrue(report['attribution']['all_graph_stages_mapped'])
        self.assertEqual(rows[0]['original_graph_node'],10)
        self.assertEqual(rows[0]['operation'],'Q')
        self.assertEqual(rows[0]['iteration'],1)
        # Logical submission frame 7 wins over outer host input frame 0.
        self.assertEqual(rows[0]['frame'],7)
        self.assertIn('frame_0',rows[0]['host_request'])
        self.assertEqual(report['requests'][0]['stages'][0]['nonkernel_gap_ms'],0)

    def test_orphan_and_cycle_are_reported(self):
        for option,expected in [('orphan','missing_original_graph_node'),('cycle','graph_ancestry_cycle')]:
            with self.subTest(option=option),tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'trace.sqlite';self.fixture(path,**{option:True})
                report,rows=analyze(path)
            self.assertFalse(report['attribution']['all_graph_stages_mapped'])
            self.assertEqual(len(report['unmapped']),1)
            self.assertIn(expected,rows[0]['mapping_errors'])

    def test_crossframe_next_encoder_keeps_submission_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.sqlite';self.fixture(path)
            db=sqlite3.connect(path)
            host='crossframe/continuous/overlap/frame_0/slot_0/gen_1'
            db.execute("update NVTX_EVENTS set text=? where text like 'layer/%'",(host,))
            db.executemany('insert into NVTX_EVENTS values(?,?,?,?,?)',[
                (-200,-110,TID,'Prefill/VisionEncoder',None),
                (120,140,TID,'submit/vision/frame=8/epoch=2/slot=1/gen=2',None)])
            db.execute('insert into CUDA_GRAPH_NODE_EVENTS values(?,?,?,?,?)',
                       (-150,-150,TID,40,None))
            db.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?,?)',
                       (125,130,TID,100,1))
            db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?,?,?)',
                       (205,215,0,1,0,9,100,PID,2,40))
            db.commit();db.close()
            report,rows=analyze(path)
        ae,vision=rows
        self.assertTrue(report['attribution']['all_graph_stages_mapped'])
        self.assertEqual({r['group'] for r in rows},{'crossframe/continuous/overlap'})
        self.assertEqual({r['host_request'] for r in rows},{host})
        self.assertEqual((ae['frame'],ae['epoch'],ae['slot'],ae['generation']),(7,2,0,1))
        self.assertEqual((vision['frame'],vision['epoch'],vision['slot'],vision['generation']),(8,2,1,2))
        self.assertNotEqual(ae['request_id'],vision['request_id'])
        self.assertEqual(len(report['requests']),2)
        overlap=overlap_metrics([ae],[vision])
        self.assertEqual(overlap['same_frame_overlap_ms'],0)
        self.assertEqual(overlap['cross_frame_overlap_ms'],5/1e6)

    def test_request_groups_preserve_runner_and_mode(self):
        for runner in ('layer','crossframe','cewp'):
            for phase in ('continuous','paired'):
                for mode in ('serial','overlap'):
                    group=f'{runner}/{phase}/{mode}'
                    label=f'{group}/frame_3/slot_1/gen_2'
                    with self.subTest(label=label):
                        self.assertTrue(is_request(label))
                        self.assertEqual(section_group(label),group)
        self.assertFalse(is_request('submit/vision/frame=3/epoch=0/slot=1/gen=2'))

    def test_outside_request_encoder_group_inference_is_explicit(self):
        for matching,ambiguous in ((True,False),(False,False),(False,True)):
            with self.subTest(matching=matching,ambiguous=ambiguous),tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'trace.sqlite';self.fixture(path)
                db=sqlite3.connect(path)
                db.execute("update NVTX_EVENTS set text=replace(text,'layer/','crossframe/') where text like 'layer/%'")
                frame=7 if matching else 8
                db.executemany('insert into NVTX_EVENTS values(?,?,?,?,?)',[
                    (-200,-110,TID,'Prefill/VisionEncoder',None),
                    (70,80,TID,f'submit/vision/frame={frame}/epoch=2/slot=0/gen=1',None)])
                db.execute('insert into CUDA_GRAPH_NODE_EVENTS values(?,?,?,?,?)',(-150,-150,TID,40,None))
                db.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?,?)',(75,76,TID,100,1))
                db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?,?,?)',
                           (190,195,0,1,0,9,100,PID,2,40))
                if ambiguous:
                    db.executemany('insert into NVTX_EVENTS values(?,?,?,?,?)',[
                        (300,340,TID,'crossframe/continuous/serial/frame_9/slot_1/gen_1',None),
                        (310,330,TID,'submit/ae/frame=9/epoch=2/slot=1/gen=1',None)])
                    db.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?,?)',(315,320,TID,101,1))
                    db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?,?,?)',
                               (350,360,0,1,0,8,101,PID,2,30))
                db.commit();db.close()
                report,rows=analyze(path)
            vision=next(r for r in rows if r['submit_kind']=='vision')
            ae=next(r for r in rows if r['submit_kind']=='ae')
            self.assertIsNone(vision['host_request'])
            self.assertTrue(vision['host_section_missing'])
            self.assertEqual(vision['source_group'],'unscoped_submit')
            if ambiguous:
                self.assertEqual(vision['group'],'unscoped_submit')
                self.assertEqual(report['attribution']['unresolved_unscoped_submissions'],1)
            else:
                self.assertEqual(vision['group'],'crossframe/continuous/overlap')
                self.assertEqual(vision['group_assignment'],
                                 'matching_submit_identity' if matching else 'unique_crossframe_process_group')
                self.assertEqual(vision['request_id']==ae['request_id'],matching)
                self.assertEqual(report['attribution']['unresolved_unscoped_submissions'],0)

    def test_legacy_d10_cannot_be_compared_with_d2(self):
        with self.assertRaisesRegex(ValueError,'iteration counts'):
            compare_signatures(dict(iterations_seen=list(range(1,11))),dict(iterations_seen=[1,2]))


if __name__=='__main__':
    unittest.main()
