"""Real Lua tests for the independent distal EndoWrist interface (no simulator required)."""
from pathlib import Path
import unittest

try:
    from .test_lua_safety import LuaRuntime, ROOT, FIXTURE
except ImportError:
    from test_lua_safety import LuaRuntime, ROOT, FIXTURE


class ArticulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = LuaRuntime()
        files = (FIXTURE, Path(__file__).with_name('articulation_fixture.lua'),
                 ROOT / 'src/endoscope/rcm_controller.lua',
                 ROOT / 'src/instruments/capability_guard.lua',
                 ROOT / 'src/instruments/articulation.lua',
                 ROOT / 'src/tissue/tissue.lua')
        cls.source = '\n'.join(p.read_text(encoding='utf-8') for p in files)

    def lua(self, body, before_init=''):
        self.runtime.run(self.source + '\n' + before_init + '\nsysCall_init(); auto=false\n' + body)

    def test_independent_dofs_do_not_change_grip_or_upstream(self):
        self.lua("""
            assert(setGrip(1,0.2));local writes=#stub.writeLog
            local shaft=sim.getObjectMatrix(arms[1].shaft,-1)
            assert(setWrist(1,0.3,0,0));eq(stub.q.a1_roll,0.3);eq(stub.q.a1_pitch,0);eq(stub.q.a1_yaw,0)
            assert(setWrist(1,0.3,-0.2,0));eq(stub.q.a1_roll,0.3);eq(stub.q.a1_pitch,-0.2);eq(stub.q.a1_yaw,0)
            assert(setWrist(1,0.3,-0.2,0.4));eq(stub.q.a1_yaw,0.4)
            eq(stub.q.a1_j1,0.2);eq(stub.q.a1_j2,0.2)
            for k=writes+1,#stub.writeLog do assert(stub.writeLog[k][1]=='a1_roll' or
                stub.writeLog[k][1]=='a1_pitch' or stub.writeLog[k][1]=='a1_yaw') end
            local after=sim.getObjectMatrix(arms[1].shaft,-1)
            for k=1,12 do eq(shaft[k],after[k]) end
            local s=getWristState(1);assert(s.available and s.units=='rad')
            eq(s.roll,0.3);eq(s.pitch,-0.2);eq(s.yaw,0.4);eq(s.grip,0.2);eq(s.jaw,0.4)
            assert(not auto and frozen)
        """)

    def test_sign_mapping_and_actual_readback(self):
        self.lua("""
            assert(setWrist(1,0.1,0.2,-0.3))
            eq(stub.q.a1_pitch,-0.2);eq(getWristState(1).pitch,0.2)
            stub.q.a1_yaw=0.12;eq(getWristState(1).yaw,0.12)
            local s=getWristState(1);s.limits.pitch[2]=99
            eq(getWristState(1).limits.pitch[2],1.4)
        """, "INSTRUMENTS.a1.articulation.pitch_sign=-1")

    def test_invalid_inputs_and_scene_limits_are_atomic(self):
        self.lua("""
            auto=true;frozen=false
            for _,bad in ipairs({0/0,math.huge,-math.huge,'bad',{},99}) do
                local n=stub.writes
                assert(not setWrist(1,bad,0,0));assert(not setWrist(1,0,bad,0));assert(not setWrist(1,0,0,bad))
                assert(stub.writes==n);assert(auto and not frozen)
            end
            assert(not setWrist(1,nil,0,0))
            stub.intervals.a1_yaw={-0.1,0.2}
            local n=stub.writes
            assert(not setWrist(1,0.3,0.2,0.15));assert(stub.writes==n)
            eq(stub.q.a1_roll,0);eq(stub.q.a1_pitch,0);eq(stub.q.a1_yaw,0)
        """)

    def test_legacy_or_invalid_metadata_fails_closed(self):
        self.lua("""
            assert(not getWristState(1).available);assert(not setWrist(1,0,0,0))
            assert(not getWristState(2).available);assert(not setWrist(2,0,0,0))
            assert(not getWristState(3).available);assert(not setWrist(3,0,0,0))
            assert(not getWristState(99).available)
        """, """
            INSTRUMENTS.a1.articulation=nil
            INSTRUMENTS.a2.articulation.yaw_joint='missing_joint'
            INSTRUMENTS.a3.articulation.roll_joint='a3_j1'
        """)

    def test_mid_path_collision_rolls_back_all_wrist_joints(self):
        self.lua("""
            assert(setWrist(1,0.1,0.05,0))
            stub.blockWrist=function(a,b) return a=='a1_tool' and stub.q.a1_yaw>0.075 and stub.q.a1_yaw<0.125 end
            assert(not setWrist(1,0.3,-0.2,0.2), 'endpoint-only gate missed intervening obstacle')
            eq(stub.q.a1_roll,0.1);eq(stub.q.a1_pitch,0.05);eq(stub.q.a1_yaw,0)
            assert(arms[1].collisionBlocks==1);assert(getWristState(1).last_error=='wrist collision')
            assert(not auto and frozen)
        """)

    def test_setter_exception_and_readback_mismatch_roll_back(self):
        self.lua("""
            assert(setWrist(1,0.1,0.05,-0.1))
            stub.throwJoint='a1_pitch'
            assert(not setWrist(1,0.2,0.1,0.2))
            eq(stub.q.a1_roll,0.1);eq(stub.q.a1_pitch,0.05);eq(stub.q.a1_yaw,-0.1)
            stub.readbackJoint='a1_yaw'
            assert(not setWrist(1,0.2,0.1,0.2))
            eq(stub.q.a1_roll,0.1);eq(stub.q.a1_pitch,0.05);eq(stub.q.a1_yaw,-0.1)
        """)

    def test_wrist_motion_keeps_grip_tissue_chain_intact(self):
        self.lua("""
            sheet(0,0.008,false);assert(setGrip(2,0.5))
            assert(setWrist(2,0.2,0.1,0.3));assert(tissueStats().cut==0, 'wrist motion caused tissue cut')
            assert(setWrist(2,0,0,0));assert(setGrip(2,0));assert(tissueStats().cut>0)
            sheet(0,0.008,false);assert(tissueGrasp(1)>0)
            local before=tissueNodes()
            assert(setWrist(1,0,0,0.2));sysCall_actuation()
            local after=tissueNodes();assert(after[2]>before[2], 'grasped nodes did not follow TCP')
            assert(tissueStats().grasped.a1>0)
            assert(setGrip(1,0.1));assert(tissueStats().grasped.a1==nil)
        """)

    def test_prograsp_slide_depends_only_on_grip(self):
        self.lua("""
            assert(setGrip(3,0.3));local slide=stub.q.a3_slide
            assert(setWrist(3,0.2,-0.1,0.4));eq(stub.q.a3_slide,slide)
            eq(stub.q.a3_j1,0.3);eq(stub.q.a3_j2,0.3)
            assert(setGrip(3,0.1));eq(stub.q.a3_slide,-0.0021*0.2)
            local s=getWristState(3);eq(s.roll,0.2);eq(s.pitch,-0.1);eq(s.yaw,0.4)
        """)

    def test_home_and_cleanup_restore_initial_wrist_not_zero(self):
        self.lua("""
            assert(setWrist(1,0.4,0.3,0.2));assert(homeRCM())
            eq(stub.q.a1_roll,0.12);eq(stub.q.a1_pitch,-0.1);eq(stub.q.a1_yaw,0.08)
            assert(not auto and frozen)
            assert(setWrist(1,0.3,0.2,0.1));sysCall_cleanup()
            eq(stub.q.a1_roll,0.12);eq(stub.q.a1_pitch,-0.1);eq(stub.q.a1_yaw,0.08)
        """, "stub.q.a1_roll=0.12;stub.q.a1_pitch=-0.1;stub.q.a1_yaw=0.08")

    def test_home_wrist_failure_restores_entire_group_and_holds(self):
        self.lua("""
            for i=1,3 do assert(setWrist(i,0.2,0.1,0.2)) end
            stub.q.a1_insert=0.02
            auto=true;frozen=false
            stub.blockWrist=function(a,b) return a=='a2_tool' and stub.q.a2_yaw<0.15 end
            assert(not homeRCM())
            for i=1,3 do local s=getWristState(i);eq(s.roll,0.2);eq(s.pitch,0.1);eq(s.yaw,0.2) end
            eq(stub.q.a1_insert,0.02);assert(not auto and frozen)
        """)

    def test_home_base_failure_restores_wrists_and_rcm_group(self):
        self.lua("""
            for i=1,3 do assert(setWrist(i,0.2,0.1,0.2)) end
            stub.q.a1_insert=0.02;arms[1].lastTarget={0,0,0.02}
            stub.q.a2_insert=0.03;arms[2].lastTarget={0,0,0.03}
            -- Wrist home is safe; a later shaft insertion hits the obstacle.
            stub.blockWrist=function(a,b) return a=='a2_tool' and stub.q.a2_insert>0.035 end
            assert(not homeRCM())
            for i=1,3 do local s=getWristState(i);eq(s.roll,0.2);eq(s.pitch,0.1);eq(s.yaw,0.2) end
            eq(stub.q.a1_insert,0.02);eq(stub.q.a2_insert,0.03)
            eq(arms[1].lastTarget[3],0.02);eq(arms[2].lastTarget[3],0.03)
            assert(not auto and frozen)
        """)


if __name__ == '__main__':
    unittest.main()