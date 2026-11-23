"""A retained chart is not a retained slit, a new observation timestamp or an endpoint."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'task'))
from same_bite_world import SameBiteWorld, SimWorld, steady


class SameBiteTest(unittest.TestCase):
    def setUp(self):
        self.w=w=SameBiteWorld.__new__(SameBiteWorld)
        w.tick,w._fix_epoch,w._fix_max_age=11,2,6
        w.cfg={'cut':{'length_m':.009,'vision_axis_tolerance_deg':5}}
        w.event_log=[]
        w.image=np.zeros((16,16,3),np.uint8)
        w._fix=dict(tick=10,epoch=2,progress=.005,front=np.zeros(3),normal=np.array([0.,0,1.]),
                    direction=np.array([1.,0,0]))
        self.edges={k:{1,2,3,4,5,6} for k in range(6)}
        w._anchor_stamp=(10,2,[1,2],[2,4],self.edges)
        self.est=dict(tissue=dict(valid=True,normal=[0.,0,1.]),scissors=dict(valid=True),
                      cut=dict(front_valid=True,mode='slit',front=[.0003,0,0],direction=[1.,0,0]))

    def confirm(self):
        with patch.object(SimWorld,'_update_fix'),patch('same_bite_world.tracks',return_value=self.edges):
            self.w._update_fix(self.est)
        return self.w._fix_valid()

    def test_fresh_geometry_must_be_checked_again_next_tick(self):
        self.assertTrue(self.confirm())
        self.assertEqual(self.w._fix['tick'],10)
        self.w.tick+=1
        self.assertFalse(self.w._fix_valid())
        self.est['cut']['mode']='memory'
        self.assertFalse(self.confirm())

    def test_no_carry_over_closure_camera_change_expiry_or_endpoint(self):
        self.w.awaiting_new_slit=True
        self.assertFalse(self.confirm())
        self.w.awaiting_new_slit=False
        self.w._fix_epoch+=1
        self.assertFalse(self.confirm())
        self.w._fix_epoch-=1
        self.w.tick=17
        self.assertFalse(self.confirm())
        self.w.tick=11
        self.w._fix['progress']=.009
        self.assertFalse(self.confirm())

    def test_front_motion_shutter_and_one_lost_bank_refuse(self):
        self.est['cut']['front']=[.0006,0,0]
        self.assertFalse(self.confirm())
        self.est['cut']['front']=[0.,0,0]
        self.est['scissors']['valid']=False
        self.assertFalse(self.confirm())
        self.assertFalse(steady(self.edges,{**self.edges,3:{10,11,12,13,14,15}}))
