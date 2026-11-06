"""CPU-only synthetic trace proof; never substitutes for a collected profile."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('progressive_analysis',HERE.parent/'analyze_traces.py')
analyzer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)
PID=1<<24
TID=PID+7
MS=1_000_000


def fixture(path):
    if path.exists():path.unlink()
    db=sqlite3.connect(path)
    db.executescript('''
    create table StringIds(id integer,value text);
    create table NVTX_EVENTS(start integer,end integer,globalTid integer,text text,textId integer);
    create table CUDA_GRAPH_NODE_EVENTS(start integer,end integer,globalTid integer,graphNodeId integer,originalGraphNodeId integer);
    create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer,nameId integer);
    create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,deviceId integer,contextId integer,greenContextId integer,streamId integer,correlationId integer,globalPid integer,demangledName integer,graphNodeId integer);
    create table CUPTI_ACTIVITY_KIND_CUDA_EVENT(timestamp integer,deviceId integer,contextId integer,greenContextId integer,streamId integer,correlationId integer,globalPid integer,eventId integer,eventSyncId integer);
    ''')
    db.executemany('insert into StringIds values(?,?)',[(1,'cudaGraphLaunch_v10000'),(2,'gemm'),(3,'_publish'),(4,'_snapshot_override'),(5,'_select'),(6,'cudaEventRecord')])
    labels=[(-100,-90,'Prefill/VisionEncoder+Projector+Metadata'),(-80,-70,'Prefill/VLM[release_KV_frontier]'),
            (-60,-40,'AE/D10[per_round_frozen_frontier]'),(-56,-41,'AE/iteration_1/mixed_KV'),
            (13,15,'progressive/request_1/lead_3/slot_1'),(25,27,'progressive/request_2/lead_3/slot_0')]
    db.executemany('insert into NVTX_EVENTS values(?,?,?,?,?)',[(a*MS,b*MS,TID,s,None) for a,b,s in labels])
    for node,time in [(10,-95),(20,-75),(21,-74),(30,-58),(31,-55),(32,-54)]:
        db.execute('insert into CUDA_GRAPH_NODE_EVENTS values(?,?,?,?,?)',(time*MS,time*MS,TID,node,None))
        db.execute('insert into CUDA_GRAPH_NODE_EVENTS values(?,?,?,?,?)',(-1,-1,TID,node+100,node))
    def launch(corr,cpu,stream,nodes):
        db.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?,?)',(int(cpu*MS),int((cpu+.01)*MS),TID,corr,1))
        for node,begin,end,name in nodes:
            db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?,?,?)',(int(begin*MS),int(end*MS),0,1,0,stream,corr,PID,name,node+100))
    launch(1,2,1,[(10,20,25,2)])
    launch(2,4,2,[(20,25,30,2)])
    launch(3,10,1,[(10,100,110,2)])
    launch(4,12,2,[(20,150,179,2),(21,179,180,3)])
    launch(5,14,3,[(30,160,161,4),(31,161,162,5),(32,162,200,2)])
    launch(6,16,1,[(10,185,210,2)])
    launch(7,24,2,[(20,220,249,2),(21,249,250,3)])
    launch(8,26,3,[(30,230,231,4),(31,231,232,5),(32,232,270,2)])
    def event(corr,cpu,gpu,stream):
        db.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?,?)',(int(cpu*MS),int((cpu+.001)*MS),TID,corr,6))
        db.execute('insert into CUPTI_ACTIVITY_KIND_CUDA_EVENT values(?,?,?,?,?,?,?,?,?)',(int(gpu*MS),0,1,0,stream,corr,PID,corr,corr))
    event(101,13.9,160,3)
    event(102,14.1,200.1,3)
    event(103,14.2,200.2,3)
    event(104,15.9,185,1)
    event(105,16.1,210.1,1)
    event(106,25.9,230,3)
    event(107,26.1,270.1,3)
    event(108,26.2,270.2,3)
    db.commit();db.close()
    return {'args':{'profile':True,'profile_lead':3},'rows':[
        dict(phase='profile',request=1,lead=3,slot=1,source_id=1,next_encoder=True,frontiers=[3]*10,variant='encoder',ae_ms=40.2,next_vision_ms=25.1,ae_finish_minus_next_encoder_ms=-9.9),
        dict(phase='profile',request=2,lead=3,slot=0,source_id=2,next_encoder=False,frontiers=[6]*10,variant='encoder',ae_ms=40.2)]}


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.path=HERE/'synthetic.sqlite'
        self.profile=fixture(self.path)

    def test_actual_intersections_helpers_and_unclipped_tails(self):
        report=analyzer.analyze(self.path,HERE/'synthetic_output',self.profile,{'synthetic':True,'gpu_launched':False})
        first,second=report['requests']
        self.assertEqual(first['AE_x_currentVLM_ms'],20)
        self.assertEqual(first['AE_x_nextVision_ms'],15)
        self.assertEqual(first['next_encoder_finish_minus_AE_finish_kernel_ms'],10)
        self.assertEqual(first['ae_busy_ms'],40)
        self.assertGreater(first['graph_kernels_after_host_NVTX'],0)
        self.assertIsNone(second['AE_x_nextVision_ms'])
        self.assertEqual(report['attribution']['helper_kernels_by_stage']['AE']['_select'],2)
        self.assertEqual(report['attribution']['helper_kernels_by_stage']['VLM']['_publish'],2)
        check=report['crosschecks'][0]
        self.assertAlmostEqual(check['next_encoder_finish_minus_AE_finish_nsys_event_ms'],9.9)
        self.assertAlmostEqual(check['event_delta_residual_ms'],0)

    def test_missing_profile_does_not_invent_lookahead(self):
        report=analyzer.analyze(self.path,HERE/'no_profile_output',None,{'synthetic':True})
        self.assertTrue(all(row['AE_x_nextVision_ms'] is None for row in report['requests']))

    def test_mismatched_source_sequence_rejected(self):
        self.profile['rows'][1]['source_id']=3
        with self.assertRaisesRegex(ValueError,'source IDs'):
            analyzer.analyze(self.path,HERE/'invalid_source_output',self.profile)

    def test_missing_node_ancestry_rejected(self):
        with sqlite3.connect(self.path) as db:
            db.execute('update CUPTI_ACTIVITY_KIND_KERNEL set graphNodeId=9999 where correlationId=5')
        with self.assertRaisesRegex(ValueError,'attribution incomplete'):
            analyzer.analyze(self.path,HERE/'invalid_node_output',self.profile)

    def test_explicit_submission_identity_and_ahead_flag(self):
        labels=[(10,'submit/vision/frame=1/slot=1/gen=9'),(12,'submit/vlm/frame=1/slot=1/gen=9'),
                (16,'submit/vision/frame=2/slot=0/gen=8/ahead'),(24,'submit/vlm/frame=2/slot=0/gen=8')]
        with sqlite3.connect(self.path) as db:
            db.executemany('insert into NVTX_EVENTS values(?,?,?,?,?)',
                [(int((t-.1)*MS),int((t+.1)*MS),TID,label,None) for t,label in labels])
        self.profile['rows'][0]['generation']=9
        self.profile['rows'][1]['generation']=8
        report=analyzer.analyze(self.path,HERE/'explicit_submit_output',self.profile,{'synthetic':True})
        self.assertEqual(report['requests'][0]['AE_x_nextVision_ms'],15)
        with sqlite3.connect(self.path) as db:
            db.execute("update NVTX_EVENTS set text=replace(text,'/ahead','') where text like 'submit/%'")
        with self.assertRaisesRegex(ValueError,'lacks /ahead'):
            analyzer.analyze(self.path,HERE/'invalid_ahead_output',self.profile)

    def test_green_id_proves_resource_pair_not_primary_context(self):
        with sqlite3.connect(self.path) as db:
            db.executescript('''
            create table TARGET_INFO_CUDA_CONTEXT_INFO(processId integer,deviceId integer,contextId integer,parentContextId integer,isGreenContext integer,numMultiprocessors integer);
            ''')
            db.executemany('insert into TARGET_INFO_CUDA_CONTEXT_INFO values(?,?,?,?,?,?)',
                [(1,0,1,0,0,0),(1,0,2,1,1,16),(1,0,3,1,1,112)])
            db.execute('update CUPTI_ACTIVITY_KIND_KERNEL set greenContextId=case when streamId=3 then 2 else 3 end')
        self.profile['resource_adapter']={'ae_sm_min':16,'actual_resources':[{'sm_counts':[16,112],'device_sm_count':128}]}
        report=analyzer.analyze(self.path,HERE/'green_context_output',self.profile,{'synthetic':True})
        audit=report['resource_audit']
        self.assertTrue(audit['passed'])
        self.assertEqual({r['kernel_primary_context_id'] for r in audit['assignments']},{1})
        self.assertEqual({r['actual_eligible_sms'] for r in audit['assignments'] if r['stage']=='AE'},{16})
        self.assertEqual({r['actual_eligible_sms'] for r in audit['assignments'] if r['stage']!='AE'},{112})
        with sqlite3.connect(self.path) as db:
            db.execute('update CUPTI_ACTIVITY_KIND_KERNEL set greenContextId=1 where streamId=3')
        report=analyzer.analyze(self.path,HERE/'wrong_green_output',self.profile,{'synthetic':True})
        self.assertFalse(report['resource_audit']['passed'])

    def test_kernel_tail_disagreement_with_profile_is_retained(self):
        with sqlite3.connect(self.path) as db:
            db.execute('update CUPTI_ACTIVITY_KIND_KERNEL set end=? where correlationId=5 and demangledName=2',(175*MS,))
        self.profile['rows'][0]['ae_finish_after_vlm_ms']=20.2
        report=analyzer.analyze(self.path,HERE/'tail_disagreement_output',self.profile,{'synthetic':True})
        first=report['requests'][0]
        self.assertTrue(first['AE_kernel_tail_before_or_at_VLM_end'])
        self.assertEqual(first['ae_finish_minus_currentVLM_finish_kernel_ms'],-5)
        self.assertEqual(first['ae_finish_minus_currentVLM_finish_profile_event_ms'],20.2)


if __name__=='__main__':
    unittest.main()
