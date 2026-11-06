import pickle
import socket
import struct
import threading
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .transfer_envs import (pose,align_pose_to_source,state8_from_pose,physical_delta,
    calvin_absolute_action,simpler_native_action,pusht_state8,pusht_native_action,clip_libero_action)
from .transfer_client import rpc,receive,validate_raw,noise_seed


class TransferMathTests(unittest.TestCase):
    def setUp(self):
        self.source=pose([-.66,0,.912],Rotation.from_euler('xyz',[.2,-.1,.4]).as_matrix())
        self.target=pose([-.34,-.46,.24],Rotation.from_euler('xyz',[-.3,.1,.2]).as_matrix())
        self.eef=pose([.1,.2,.6],Rotation.from_euler('xyz',[.4,.2,-.3]).as_matrix())
        self.reference=dict(robot_base_world_matrix=self.source.tolist(),reference_state8=[-.21,-.01,1.17,-3.14,0,.08,.0387,-.0387])

    def test_base_alignment_roundtrip(self):
        aligned=align_pose_to_source(self.eef,self.target,self.source)
        restored=self.target.dot(np.linalg.inv(self.source)).dot(aligned)
        np.testing.assert_allclose(restored,self.eef,atol=1e-12)
        state=state8_from_pose(self.eef,self.target,self.reference,.06)
        np.testing.assert_allclose(state[-2:],[.03,-.03],atol=1e-8)
        np.testing.assert_allclose(Rotation.from_rotvec(state[3:6]).as_matrix(),aligned[:3,:3],atol=1e-6)

    def test_absolute_command_uses_actual_pose_and_left_rotation(self):
        action=np.array([1,-1,.5,.2,-.4,.3,1])
        command,audit=calvin_absolute_action(action,self.eef,self.target,self.reference)
        dp,dr,_,_=physical_delta(action,self.source,self.target)
        np.testing.assert_allclose(command[0],self.eef[:3,3]+dp,atol=1e-12)
        expected=Rotation.from_rotvec(dr).as_matrix().dot(self.eef[:3,:3])
        np.testing.assert_allclose(Rotation.from_quat(command[1]).as_matrix(),expected,atol=1e-12)
        self.assertEqual(command[2],-1)
        zero,_=calvin_absolute_action(np.zeros(7),self.eef,self.target,self.reference)
        np.testing.assert_allclose(zero[0],self.eef[:3,3],atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(zero[1]).as_matrix(),self.eef[:3,:3],atol=1e-12)

    def test_clips_and_physical_units(self):
        action=np.array([2,-2,.5,2,-2,.2,2])
        clipped,audit=clip_libero_action(action)
        np.testing.assert_equal(clipped,[1,-1,.5,1,-1,.2,1])
        self.assertEqual(audit['clipped_components'],[0,1,3,4,6])
        native,_=simpler_native_action(action,self.reference)
        np.testing.assert_allclose(self.source[:3,:3].dot(native[:3]),[.05,-.05,.025],atol=1e-12)
        np.testing.assert_allclose(self.source[:3,:3].dot(native[3:6]),[.5,-.5,.1],atol=1e-12)
        with self.assertRaises(ValueError): clip_libero_action([np.nan]*7)

    def test_pusht_projection_is_declared_xy_only(self):
        state=pusht_state8([256,256],self.reference)
        np.testing.assert_allclose(state,self.reference['reference_state8'],atol=1e-6)
        moved=pusht_state8([512,0],self.reference)
        np.testing.assert_allclose(moved[:2]-state[:2],[.3,-.3],atol=1e-6)
        np.testing.assert_allclose(moved[2:],state[2:])
        action,audit=pusht_native_action([1,-1,1,1,1,1,1],[500,10])
        np.testing.assert_equal(action,[512,0])
        self.assertEqual(audit['ignored_channels'],[2,3,4,5,6])

    def test_seed_has_no_mode_dependency(self):
        self.assertEqual(noise_seed('calvin',0,1),noise_seed('calvin',0,1))
        self.assertNotEqual(noise_seed('calvin',0,1),noise_seed('calvin',0,2))

    def test_legacy_pickle_rpc_roundtrip_preserves_upright_rgb(self):
        image=np.zeros((5,7,3),dtype=np.uint8);image[0,0]=[1,2,3];image[-1,-1]=[7,8,9]
        raw=dict(base_rgb=image,wrist_rgb=np.zeros_like(image),state8=np.arange(8,dtype=np.float32))
        validate_raw(raw)
        with socket.socket() as server:
            server.bind(('127.0.0.1',0));server.listen(1)
            port=server.getsockname()[1]
            def handle():
                connection,_=server.accept()
                with connection:
                    n=struct.unpack('!I',receive(connection,4))[0]
                    request=pickle.loads(receive(connection,n))
                    result=dict(ok=True,actions=[[float(x) for x in request['raw']['state8'][:7]]]*50,
                        first=request['raw']['base_rgb'][0,0].tolist(),last=request['raw']['base_rgb'][-1,-1].tolist())
                    payload=pickle.dumps(result,protocol=4)
                    connection.sendall(struct.pack('!I',len(payload))+payload)
            thread=threading.Thread(target=handle);thread.start()
            response=rpc(port,dict(op='predict',mode='full_d10',raw=raw,instruction='test',seed=7))
            thread.join(2)
        self.assertEqual(response['first'],[1,2,3]);self.assertEqual(response['last'],[7,8,9])
        self.assertEqual(np.asarray(response['actions']).shape,(50,7))


if __name__=='__main__': unittest.main()
