"""Execute the production Lua controller/guards/tissue with a deterministic sim stub.

Run: python -m unittest discover -s scripts/tests -p test_lua_safety.py -v
No Python packages are required. Set LUA_LIBRARY to a Lua 5.3/5.4 shared library
(including CoppeliaSim's lua.dll), or LUA_EXECUTABLE to a Lua interpreter. Common
system libraries/interpreters and COPPELIASIM_ROOT/HOME are also detected. Missing
Lua is a test ERROR, never a silent pass or skip. These are not mesh/IK scene tests.
"""
import ctypes
import ctypes.util
import os
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name('lua_safety_fixture.lua')
SOURCES = (
    ROOT / 'src/endoscope/rcm_controller.lua',
    ROOT / 'src/instruments/capability_guard.lua',
    ROOT / 'src/tissue/tissue.lua',
)


class LuaRuntime:
    def __init__(self):
        self.lib = None
        self.executable = None
        errors = []
        candidates = []
        explicit_library = os.environ.get('LUA_LIBRARY')
        explicit_executable = os.environ.get('LUA_EXECUTABLE')
        if explicit_library:
            candidates.append(explicit_library)
        elif not explicit_executable:
            for var in ('COPPELIASIM_ROOT', 'COPPELIASIM_HOME'):
                if os.environ.get(var):
                    folder = Path(os.environ[var])
                    candidates.extend(str(folder / name) for name in ('lua.dll', 'liblua.so', 'liblua.dylib'))
            for name in ('lua5.4', 'lua54', 'lua5.3', 'lua53', 'lua'):
                found = ctypes.util.find_library(name)
                if found:
                    candidates.append(found)
            if os.name == 'nt':
                base = Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'CoppeliaRobotics'
                candidates.extend(str(p) for p in base.glob('CoppeliaSim*/lua.dll'))
        for candidate in dict.fromkeys(candidates):
            try:
                lib = ctypes.CDLL(candidate)
                lib.luaL_newstate.argtypes = []
                lib.luaL_newstate.restype = ctypes.c_void_p
                lib.luaL_openlibs.argtypes = [ctypes.c_void_p]
                lib.luaL_openlibs.restype = None
                lib.luaL_loadbufferx.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                               ctypes.c_char_p, ctypes.c_char_p]
                lib.luaL_loadbufferx.restype = ctypes.c_int
                lib.lua_pcallk.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_ssize_t, ctypes.c_void_p]
                lib.lua_pcallk.restype = ctypes.c_int
                lib.lua_tolstring.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
                lib.lua_tolstring.restype = ctypes.c_void_p
                lib.lua_close.argtypes = [ctypes.c_void_p]
                lib.lua_close.restype = None
                self.lib = lib
                self.description = candidate
                return
            except (OSError, AttributeError) as exc:
                errors.append(f'{candidate}: {exc}')
        if not explicit_library:
            names = [explicit_executable] if explicit_executable else ['lua5.4', 'lua54', 'lua5.3', 'lua53', 'lua']
            for name in names:
                executable = shutil.which(name)
                if executable:
                    self.executable = executable
                    self.description = executable
                    return
        raise RuntimeError('Lua not found. Install Lua or set LUA_LIBRARY / LUA_EXECUTABLE.\n' + '\n'.join(errors))

    def run(self, source):
        if self.executable:
            result = subprocess.run([self.executable, '-'], input=source, encoding='utf-8',
                                    capture_output=True, timeout=30)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            return
        lib = self.lib
        state = lib.luaL_newstate()
        if not state:
            raise RuntimeError('luaL_newstate failed')
        try:
            lib.luaL_openlibs(state)
            data = source.encode('utf-8')
            status = lib.luaL_loadbufferx(state, data, len(data), b'@lua_safety', None)
            if status == 0:
                status = lib.lua_pcallk(state, 0, 0, 0, 0, None)
            if status:
                size = ctypes.c_size_t()
                text = lib.lua_tolstring(state, -1, ctypes.byref(size))
                message = ctypes.string_at(text, size.value).decode('utf-8', errors='replace') if text else 'Lua error'
                raise AssertionError(message)
        finally:
            lib.lua_close(state)


class LuaSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = LuaRuntime()
        print(f'Lua safety runtime: {cls.runtime.description}', flush=True)
        # Match the scene's concatenation order, including the unmodified capability wrapper.
        cls.source = '\n'.join(p.read_text(encoding='utf-8') for p in (FIXTURE, *SOURCES))

    def lua(self, body):
        self.runtime.run(self.source + '\nsysCall_init(); auto=false\n' + body)

    def test_invalid_grip_requests_are_atomic(self):
        self.lua("""
            assert(setGrip(1,0.3))
            for _,q in ipairs({0/0,math.huge,-math.huge,'bad',{},0.8,-0.2}) do
                local calls=stub.writes
                assert(not setGrip(1,q))
                assert(stub.writes==calls, 'invalid input moved a joint')
                eq(stub.q.a1_j1,0.3);eq(stub.q.a1_j2,0.3);eq(arms[1].gripAngle,0.3)
            end
            assert(not setGrip(1,nil));assert(not setGrip(99,0))
            stub.intervals.a1_j2={0,0.2}
            local calls=stub.writes
            assert(not setGrip(1,0.25));assert(stub.writes==calls, 'second jaw interval checked too late')
            eq(arms[1].gripAngle,0.3)
        """)

    def test_readback_and_collision_rollback(self):
        self.lua("""
            stub.quantize=true
            assert(setGrip(1,0.234));eq(arms[1].gripAngle,0.23)
            eq(getRCMStats()[1].grip,0.23)
            stub.quantize=false
            stub.blocked.a1_obstacle=true
            assert(not setGrip(1,0.4))
            eq(stub.q.a1_j1,0.23);eq(stub.q.a1_j2,0.23);eq(arms[1].gripAngle,0.23)
            assert(arms[1].collisionBlocks==1)
            stub.blocked.a1_obstacle=nil
            stub.jaw2offset=0.01
            assert(not setGrip(1,0.4), 'asymmetric readback must fail')
            eq(stub.q.a1_j1,0.23);eq(stub.q.a1_j2,0.23)
        """)

    def test_init_reads_actual_grip(self):
        self.runtime.run(self.source + """
            stub.q.a1_j1=0.27;stub.q.a1_j2=0.27
            sysCall_init();eq(getRCMStats()[1].grip,0.27)
        """)

    def test_hinge_wrapper_rejects_bad_input_and_rolls_back(self):
        self.lua("""
            assert(setGrip(3,0.3));local slide=stub.q.a3_slide
            for _,q in ipairs({'bad',{},0/0,math.huge,0.8,-0.2}) do
                local calls=stub.writes
                assert(not setGrip(3,q));eq(stub.q.a3_slide,slide);assert(stub.writes==calls)
            end
            assert(not setGrip(3,0.8));eq(stub.q.a3_slide,slide)
            stub.blocked.a3_obstacle=true
            assert(not setGrip(3,0));eq(stub.q.a3_slide,slide);eq(arms[3].gripAngle,0.3)
        """)

    def test_hinge_guard_without_tissue_wrapper(self):
        source = '\n'.join(p.read_text(encoding='utf-8') for p in (FIXTURE, *SOURCES[:2]))
        self.runtime.run(source + """
            sysCall_init()
            for _,q in ipairs({'bad',{},0/0,math.huge,0.8,-0.2}) do
                local calls=stub.writes
                assert(not setGrip(3,q));assert(stub.writes==calls)
            end
            assert(not setGrip(3,nil))
        """)

    def test_rcm_nonfinite_targets_do_not_mutate_state(self):
        self.lua("""
            auto=true;frozen=true
            for _,q in ipairs({0/0,math.huge,-math.huge,'bad',{}}) do
                local calls=stub.writes
                assert(not setRCMTarget(1,q,0,0.04))
                assert(not setRCMTarget(1,0,q,0.04))
                assert(not setRCMTarget(1,0,0,q))
                assert(stub.writes==calls);assert(auto and frozen)
            end
            assert(not setRCMTarget(1,0,0,nil))
        """)

    def test_partial_closing_sweeps_both_blades_not_centre(self):
        self.lua("""
            sheet(0.35,0.008,false)
            assert(setGrip(2,0.5));assert(tissueStats().cut==0)
            assert(setGrip(2,0.2));assert(tissueStats().cut>0, 'positive blade sweep missed')
            sheet(-0.35,0.008,false)
            assert(setGrip(2,0.5));assert(setGrip(2,0.2));assert(tissueStats().cut>0, 'negative blade sweep missed')
            sheet(0,0.008,false)
            assert(setGrip(2,0.5));assert(setGrip(2,0.2))
            assert(tissueStats().cut==0, 'still-open centre gap was cut')
            assert(setGrip(2,0));assert(tissueStats().cut>0, 'complete closure lost centre cut')
        """)

    def test_small_steps_below_arming_threshold_continue(self):
        self.lua("""
            sheet(0.05,0.008,false)
            assert(setGrip(2,0.5));assert(setGrip(2,0.14));assert(tissueStats().cut==0)
            for k=1,1400 do assert(setGrip(2,0.14*(1-k/1400))) end
            assert(tissueStats().cut>0, 'sub-milliradian remaining closure did not cut')
            local events=tissueStats().cut_events
            eq(events[1].from_angle,0.5);eq(events[1].to_angle,0.14)
            eq(events[2].from_angle,0.14);eq(events[2].to_angle,0.1399)
            local count=#events
            assert(setGrip(2,0));assert(#tissueStats().cut_events==count, 'no-op cut')
        """)

    def test_arming_cycle_reopening_and_completion(self):
        self.lua("""
            sheet(0,0.008,false)
            assert(setGrip(2,0.1));assert(setGrip(2,0));assert(tissueStats().cut==0, 'unarmed cut')
            assert(setGrip(2,0.5));assert(setGrip(2,0.08));assert(setGrip(2,0.1));assert(setGrip(2,0))
            assert(tissueStats().cut==0, 'subthreshold reopening must cancel cycle')
            assert(setGrip(2,0.15));assert(setGrip(2,0));assert(tissueStats().cut>0)
            local events=#tissueStats().cut_events
            assert(setGrip(2,0.1));assert(setGrip(2,0));assert(#tissueStats().cut_events==events, 'completed cycle stayed armed')
        """)

    def test_full_and_incremental_closure_agree(self):
        self.lua("""
            for _,coplanar in ipairs({false,true}) do
                for _,angle in ipairs({-0.45,-0.25,0,0.05,0.25,0.45}) do
                    sheet(angle,0.008,coplanar);assert(setGrip(2,0.5));assert(setGrip(2,0))
                    local direct=tissueConstraints();assert(tissueStats().cut>0)
                    sheet(angle,0.008,coplanar);assert(setGrip(2,0.5))
                    for k=1,50 do assert(setGrip(2,0.5*(1-k/50))) end
                    local stepped=tissueConstraints()
                    for i=1,#direct do assert(direct[i]==stepped[i], 'step-dependent sweep') end
                end
            end
            sheet(0.3,0.02,false);assert(setGrip(2,0.5));assert(setGrip(2,0))
            assert(tissueStats().cut==0, 'cut beyond blade tips')
        """)

    def test_rejected_closure_preserves_armed_cycle_and_tissue(self):
        self.lua("""
            sheet(0,0.008,false);assert(setGrip(2,0.5));assert(setGrip(2,0.1))
            local events=#tissueStats().cut_events
            stub.blocked.a2_obstacle=true
            assert(not setGrip(2,0));assert(tissueStats().cut==0);assert(#tissueStats().cut_events==events)
            eq(arms[2].gripAngle,0.1)
            stub.blocked.a2_obstacle=nil
            assert(setGrip(2,0));assert(tissueStats().cut>0)
            sheet(0,0.008,false);assert(setGrip(2,0.1));assert(setGrip(2,0))
            assert(tissueStats().cut==0, 'new tissue inherited old cycle')
        """)

    def test_sweep_uses_actual_old_and_new_readback(self):
        self.lua("""
            sheet(0,0.008,false);stub.quantize=true
            assert(setGrip(2,0.504));assert(setGrip(2,0.144))
            local events=tissueStats().cut_events
            eq(events[1].from_angle,0.5);eq(events[1].to_angle,0.14)
            local count=#events
            assert(setGrip(2,0.139));assert(#tissueStats().cut_events==count, 'unchanged actual jaws cut tissue')
            assert(setGrip(2,0.004));assert(tissueStats().cut>0)
            eq(arms[2].gripAngle,0)
        """)

    def test_grasp_requires_actual_closed_jaws_and_opening_releases(self):
        self.lua("""
            sheet(0,0.008,false)
            assert(setGrip(1,0.5));assert(tissueGrasp(1)==0)
            assert(setGrip(1,0));assert(tissueGrasp(1)>0)
            assert(tissueStats().grasped.a1>0)
            stub.blocked.a1_obstacle=true
            assert(not setGrip(1,0.2));assert(tissueStats().grasped.a1>0, 'rejected opening released tissue')
            stub.blocked.a1_obstacle=nil
            assert(setGrip(1,0.001));assert(tissueStats().grasped.a1==nil)
            assert(setGrip(1,0));stub.q.a1_j2=0.3
            assert(tissueGrasp(1)==0, 'cached angle or only first jaw used for grasp')
            stub.q.a1_j2=0;assert(tissueGrasp(1)>0)
            stub.q.a1_j2=0.1;sysCall_actuation()
            assert(tissueStats().grasped.a1==nil, 'external opening did not release on step')
            assert(setGrip(1,-0.1));assert(tissueGrasp(1)==0, 'crossed jaws treated as closed')
        """)

    def test_capabilities_still_gate_tissue_operations(self):
        self.lua("""
            sheet(0,0.008,false)
            assert(tissueGrasp(2)==-1)
            assert(setGrip(1,0.5));assert(setGrip(1,0));assert(tissueStats().cut==0)
            assert(tissueGrasp(1)>0);assert(tissueRelease(1)>0);assert(tissueRelease(1)==0)
        """)

    def test_home_success_and_group_rollback(self):
        self.lua("""
            assert(homeRCM()==true)
            for i,a in ipairs(arms) do
                stub.q[a.insertion]=0.02+i*0.001
                stub.q[a.handles[1]]=0.1*i
                a.lastTarget={i,2*i,stub.q[a.insertion]}
                a.converged=true;a.targetError=0.01*i;a.drift=0.002*i;a.distance=0.3*i;a.maxDrift=0.004*i
                sim.setObjectPosition(a.projection,-1,{i,i+1,i+2})
            end
            auto=true;frozen=true
            stub.blocked.a2_obstacle=true
            assert(homeRCM()==false)
            assert(auto and frozen)
            for i,a in ipairs(arms) do
                eq(stub.q[a.insertion],0.02+i*0.001);eq(stub.q[a.handles[1]],0.1*i)
                eq(a.lastTarget[1],i);eq(a.lastTarget[2],2*i);eq(a.lastTarget[3],stub.q[a.insertion])
                assert(a.converged);eq(a.targetError,0.01*i);eq(a.drift,0.002*i)
                eq(a.distance,0.3*i);eq(a.maxDrift,0.004*i)
                local p=sim.getObjectPosition(a.projection,-1);eq(p[1],i);eq(p[2],i+1);eq(p[3],i+2)
            end
            assert(arms[2].collisionBlocks==1, 'failure diagnostics should survive rollback')
            stub.blocked.a2_obstacle=nil
            assert(homeRCM()==true);assert(not auto and not frozen)
            for _,a in ipairs(arms) do eq(stub.q[a.insertion],a.insertion_home);eq(a.lastTarget[1],0) end
        """)

    def test_home_ik_failure_also_rolls_back_group(self):
        self.lua("""
            stub.q.a1_insert=0.02;arms[1].lastTarget={0,0,0.02}
            stub.q.a2_joint=0.12;stub.q.a2_insert=0.03;arms[2].lastTarget={1,0,0.03}
            -- Shaft position fixed away from the reachable target: exercise the real solver rejection.
            stub.positions.a2_shaft={1,0,0}
            assert(homeRCM()==false)
            eq(stub.q.a1_insert,0.02);eq(arms[1].lastTarget[3],0.02)
            eq(stub.q.a2_joint,0.12);eq(stub.q.a2_insert,0.03);eq(arms[2].lastTarget[1],1)
            assert(arms[2].failures==1);assert(arms[2].collisionBlocks==0)
        """)

    def test_tissue_explicit_nodes_follow_geometry(self):
        self.lua("""
            -- Curved node geometry (the prostate-proxy ROI path): the 3x2 grid is bent around a
            -- cylinder of radius r, so the i-chord differs from the flat `spacing` pitch.
            local r=0.01
            local nodes={}
            for j=0,1 do for i=0,2 do
                local a=(i-1)*0.3
                nodes[j*3+i+1]={r*math.sin(a),j*0.001,r*(1-math.cos(a))}
            end end
            tissueCreate{nu=3,nv=2,nodes=nodes,spacing=0.001,pretension=0.1,anchor_sides=false,
                         iterations=0,damping=0}
            eq(#T.x,6)
            eq(T.x[1][1],nodes[1][1]);eq(T.x[6][3],nodes[6][3])
            local p0,p1=T.x[1],T.x[2]
            local chord=math.sqrt((p1[1]-p0[1])^2+(p1[2]-p0[2])^2+(p1[3]-p0[3])^2)
            assert(math.abs(chord-2*r*math.sin(0.15))<1e-12)
            assert(chord>0.001, 'curved chord must differ from the flat pitch to discriminate')
            assert(math.abs(T.springs[1].rest-chord*0.9)<1e-12,
                   'rest must follow the supplied geometry incl. pretension')
            assert(T.pinned[4]~=nil and T.pinned[1]==nil,'far-row anchoring unchanged')
        """)


if __name__ == '__main__':
    unittest.main()