"""Execute the folded-foreground slide-guide extension under the sim stub.

The staged chunk is built by seam_slide_world.staged_source itself (fixture + controller
sources + ruler_print.lua + surface_seam.lua as the parent), so the injection validation
and the exact projection call site are exercised, not re-implemented here.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'task'))

from test_lua_safety import FIXTURE, LEGACY_SOURCES as SOURCES, LuaRuntime  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PARENT = (ROOT / 'src/tissue/ruler_print.lua', ROOT / 'src/tissue/surface_seam.lua')
SPEC = ('{nu=24,nv=14,spacing=0.001,origin={0,0,0},u={1,0,0},v={0,1,0},'
        'pretension=0.05,anchor_sides=false,anchor_from_row=3,grasp_radius=0.0025}')


class SeamSlideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = LuaRuntime()
        print(f'Lua safety runtime: {cls.runtime.description}', flush=True)
        from seam_slide_world import staged_source
        cls.staged_source = staticmethod(staged_source)
        parent = '\n'.join(p.read_text(encoding='utf-8') for p in (FIXTURE, *SOURCES, *PARENT))
        cls.parent = parent
        cls.source = staged_source(parent)

    def lua(self, body):
        self.runtime.run(self.source + '\nsysCall_init(); auto=false\n' + body)

    def test_projection_call_site_is_unique_and_inside_solve(self):
        src = self.source
        self.assertEqual(src.count('\n        projectSlide()\n'), 1)
        tail = '        end\n    end\n    local m=0\n    for _,s in ipairs(T.springs) do'
        self.assertNotIn(tail, src)
        decl = src.index('local function tissueStep()')
        call = src.index('\n        projectSlide()\n')
        strain = src.index('    local m=0', decl)
        self.assertGreater(call, decl)
        self.assertLess(call, strain)

    def test_guide_clamps_normal_component_only(self):
        self.lua("""
            tissueCreate(%s)
            local rp=seamRailPins()
            eq(#rp,130)
            local n=0
            for j=0,12 do for _,i in ipairs({6,7,8,9,10,13,14,15,16,17}) do n=n+1;eq(rp[n],j*24+i) end end
            local k=6*24+8+1
            local p0={T.x[k][1],T.x[k][2],T.x[k][3]}
            T.x[k]={p0[1],p0[2]+0.0008,p0[3]+0.0004}
            projectSlide()
            eq(T.x[k][1],p0[1]);eq(T.x[k][2],p0[2]+0.0008);eq(T.x[k][3],p0[3]+0.0001)
            T.x[k]={p0[1],p0[2]-0.0008,p0[3]-0.0004}
            projectSlide()
            eq(T.x[k][1],p0[1]);eq(T.x[k][2],p0[2]-0.0008);eq(T.x[k][3],p0[3]-0.0001)
        """ % SPEC)

    def test_adjacent_central_and_grasp_pinned_points_skipped(self):
        self.lua("""
            tissueCreate(%s)
            local kc=6*24+11+1
            local c0={T.x[kc][1],T.x[kc][2],T.x[kc][3]}
            T.x[kc]={c0[1],c0[2],c0[3]+0.0004}
            projectSlide()
            eq(T.x[kc][3],c0[3]+0.0004)
            local kg=6*24+8+1
            T.pinned[kg]={arm=1,rel={0,0,0}}
            local g0={T.x[kg][1],T.x[kg][2],T.x[kg][3]}
            T.x[kg]={g0[1],g0[2],g0[3]+0.0004}
            projectSlide()
            eq(T.x[kg][3],g0[3]+0.0004)
            T.pinned[kg]=nil
        """ % SPEC)

    def test_parent_pins_connectivity_and_tear_threshold_unchanged(self):
        self.lua("""
            local spec=%s
            createSeamSlide(spec)
            local bp={};local bn=0
            for k,p in pairs(T.pinned) do bp[k]=p.fixed~=nil and 'f' or 'a';bn=bn+1 end
            local bs=T.springs
            local tear=T.p.tear_ratio
            tissueCreate(spec)
            local np=0
            for k,p in pairs(T.pinned) do
                np=np+1;assert(bp[k],'pin added by slide fixture')
                assert((p.fixed~=nil)==(bp[k]=='f'),'pin type changed')
            end
            eq(np,bn)
            eq(#T.springs,#bs)
            for i,s in ipairs(T.springs) do
                eq(s.a,bs[i].a);eq(s.b,bs[i].b);eq(s.rest,bs[i].rest);assert(s.alive==bs[i].alive)
            end
            eq(T.p.tear_ratio,tear)
            eq(tissueStats().cut,0);eq(T.torn,0);eq(T.seamReleases,0);eq(surfaceSeamReleases(),0)
        """ % SPEC)

    def test_invalid_configuration_refused_before_rpc(self):
        import seam_slide_world as slw
        import surface_seam_world as ssw
        truth = {'cut': {'control_source': 'truth'}}
        vision = {'cut': {'control_source': 'vision'}}
        called = []
        with patch.object(ssw.SurfaceSeamWorld, '__init__', lambda self, *a, **k: called.append(1)):
            for cfg, mode in ((truth, 'cut_only'), (vision, 'full')):
                with self.assertRaises(ValueError):
                    slw.SeamSlideWorld(None, None, SimpleNamespace(), cfg, mode=mode)
        self.assertFalse(called, 'world construction reached the sim before validation')


if __name__ == '__main__':
    unittest.main()
