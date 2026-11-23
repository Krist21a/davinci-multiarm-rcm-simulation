"""Execute the support-rail tissue extension under the same sim stub as test_lua_safety.

Chunk order mirrors SeamRailWorld staging: fixture + controller sources + ruler_print.lua
+ surface_seam.lua + 'local railLeft=N' + seam_rail.lua. The seam parent baseline is
captured through the chunk's own createSeam binding, so nothing is re-implemented here.
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
RAIL = ROOT / 'src/tissue/seam_rail.lua'
SPEC = ('{nu=24,nv=14,spacing=0.001,origin={0,0,0},u={1,0,0},v={0,1,0},'
        'pretension=0.05,anchor_sides=false,anchor_from_row=3,grasp_radius=0.0025}')

RAIL_BODY = """
    local spec=%s
    createSeam(spec)
    local base={x=T.x,springs=T.springs,tri=T.tri,pinned=T.pinned}
    local bp={}
    for k,p in pairs(T.pinned) do bp[k]=p.fixed~=nil and 'f' or 'a' end
    local np=0;for k in pairs(T.pinned) do np=np+1 end
    tissueCreate(spec)
    eq(#T.x,#base.x)
    for k=1,#T.x do for c=1,3 do eq(T.x[k][c],base.x[k][c]) end end
    eq(#T.springs,#base.springs)
    for idx,s in ipairs(T.springs) do
        local o=base.springs[idx]
        eq(s.a,o.a);eq(s.b,o.b);eq(s.rest,o.rest);assert(s.alive==o.alive)
    end
    eq(#T.tri,#base.tri)
    for i,tr in ipairs(T.tri) do for q=1,7 do eq(tr[q],base.tri[i][q]) end end
    local added=0
    for k,p in pairs(T.pinned) do
        if bp[k] then
            assert((p.fixed~=nil)==(bp[k]=='f'),'existing pin type changed')
        else
            added=added+1
            assert(p.fixed~=nil,'rail pin is not a fixed anchor')
            local i=(k-1)%%24;local j=math.floor((k-1)/24)
            assert(j>=6 and j<=12 and (i==%d or i==%d),'rail pin outside the registered lane')
        end
    end
    eq(added,14)
    local now=0;for k in pairs(T.pinned) do now=now+1 end
    eq(now,np+14)
    local rp=seamRailPins()
    eq(#rp,14)
    local n=0
    for j=6,12 do for _,i in ipairs({%d,%d}) do n=n+1;eq(rp[n],j*24+i) end end
    eq(tissueStats().cut,0);eq(T.torn,0);eq(T.seamReleases,0);eq(surfaceSeamReleases(),0)
    local faces=surfaceSeamFaces();eq(#faces,26);for i=1,26 do eq(faces[i],1) end
"""


class SeamRailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = LuaRuntime()
        print(f'Lua safety runtime: {cls.runtime.description}', flush=True)
        parent = '\n'.join(p.read_text(encoding='utf-8') for p in (FIXTURE, *SOURCES, *PARENT))
        rail = RAIL.read_text(encoding='utf-8')
        cls.src = {r: parent + f'\nlocal railLeft={r}\n' + rail for r in (8, 9)}

    def lua(self, body, source):
        self.runtime.run(source + '\nsysCall_init(); auto=false\n' + body)

    def test_rail_outer_pins_and_unchanged_coupon(self):
        self.lua(RAIL_BODY % (SPEC, 8, 15, 8, 15), self.src[8])

    def test_rail_inner_pins_and_unchanged_coupon(self):
        self.lua(RAIL_BODY % (SPEC, 9, 14, 9, 14), self.src[9])

    def test_refuses_non_coupon_specs(self):
        self.lua("""
            local base={nu=24,nv=14,spacing=0.001,origin={0,0,0},u={1,0,0},v={0,1,0}}
            local s1={anchor_sides=true};for k,v in pairs(base) do s1[k]=v end
            assert(not pcall(tissueCreate,s1),'anchor_sides spec accepted')
            local s2={material_bands={}};for k,v in pairs(base) do s2[k]=v end
            assert(not pcall(tissueCreate,s2),'material_bands spec accepted')
            local s3={nodes={{0,0,0}}};for k,v in pairs(base) do s3[k]=v end
            assert(not pcall(tissueCreate,s3),'nodes spec accepted')
        """, self.src[8])

    def test_invalid_configuration_refused_before_rpc(self):
        import seam_rail_world as srw
        import surface_seam_world as ssw
        truth = {'cut': {'control_source': 'truth'}}
        vision = {'cut': {'control_source': 'vision'}}
        called = []
        with patch.object(ssw.SurfaceSeamWorld, '__init__', lambda self, *a, **k: called.append(1)):
            for args, cfg, mode in (
                    (SimpleNamespace(rail='outer'), truth, 'cut_only'),
                    (SimpleNamespace(rail='outer'), vision, 'full'),
                    (SimpleNamespace(rail='sideways'), truth, 'full'),
                    (SimpleNamespace(), truth, 'full')):
                with self.assertRaises(ValueError):
                    srw.SeamRailWorld(None, None, args, cfg, mode=mode)
        self.assertFalse(called, 'world construction reached the sim before validation')

    def test_rail_left_mapping(self):
        import seam_rail_world as srw
        import surface_seam_world as ssw
        cfg = {'cut': {'control_source': 'truth'}}
        with patch.object(ssw.SurfaceSeamWorld, '__init__', lambda self, *a, **k: None):
            w = srw.SeamRailWorld(None, None, SimpleNamespace(rail='outer'), cfg, mode='full')
            self.assertEqual(w.rail_left, 8)
            w = srw.SeamRailWorld(None, None, SimpleNamespace(rail='inner'), cfg, mode='full')
            self.assertEqual(w.rail_left, 9)


if __name__ == '__main__':
    unittest.main()
