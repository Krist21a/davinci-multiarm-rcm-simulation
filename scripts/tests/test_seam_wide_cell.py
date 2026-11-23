"""Exercise the wide central-cell staged source under the sim stub.

Chunk built through seam_wide_cell_world.wide_source itself: fixture + controller
sources + ruler_print.lua + surface_seam.lua as parent (then slide injection, grid
widening and the certificate print append). The narrow parent baseline is captured
through the chunk's own createSeamSlide binding.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'task'))

from test_lua_safety import FIXTURE, LEGACY_SOURCES as SOURCES, LuaRuntime  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PARENT = (ROOT / 'src/tissue/ruler_print.lua', ROOT / 'src/tissue/surface_seam.lua')
SPEC = ('{nu=24,nv=14,spacing=0.001,origin={0,0,0},u={1,0,0},v={0,1,0},'
        'pretension=0.08,anchor_sides=false,anchor_from_row=3,grasp_radius=0.0025}')


class WideCellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = LuaRuntime()
        print(f'Lua safety runtime: {cls.runtime.description}', flush=True)
        from seam_wide_cell_world import wide_source
        cls.wide_source = staticmethod(wide_source)
        cls.parent = '\n'.join(p.read_text(encoding='utf-8') for p in (FIXTURE, *SOURCES, *PARENT))
        cls.source = wide_source(cls.parent)

    def lua(self, body):
        self.runtime.run(self.source + '\nsysCall_init(); auto=false\n' + body)

    def test_wide_grid_coordinates(self):
        self.lua("""
            tissueCreate(%s)
            local h=0.001
            for j=0,13 do for i=0,23 do
                local x=T.x[j*24+i+1]
                local want=(i<=11) and (-11.5+i*(9/11))*h or (2.5+(i-12)*(9/11))*h
                eq(x[1],want)
                eq(x[2],j*h)
                eq(x[3],0)
                eq(x[1],-T.x[j*24+(23-i)+1][1])
            end end
            eq(T.x[12][1],-0.0025)   -- (11,0) centre boundary
            eq(T.x[13][1],0.0025)    -- (12,0)
            eq(T.x[1][1],-0.0115)    -- (0,0) outer boundary
            eq(T.x[24][1],0.0115)    -- (23,0)
        """ % SPEC)

    def test_coupon_mechanics_and_parent_invariants(self):
        self.lua("""
            local spec=%s
            createSeamSlide(spec)
            local pp=T.p
            local npin=0;for k in pairs(T.pinned) do npin=npin+1 end
            tissueCreate(spec)
            local f=surfaceSeamFaces()
            eq(#f,26);for i=1,26 do eq(f[i],1) end
            eq(tissueStats().cut,0);eq(T.torn,0);eq(T.seamReleases,0);eq(surfaceSeamReleases(),0)
            local trans=nil
            for k,s in ipairs(T.springs) do
                if (s.a==12 and s.b==13) or (s.a==13 and s.b==12) then trans=s.rest end
            end
            eq(trans,0.005*(1-spec.pretension))
            for _,key in ipairs({'cut_radius','grasp_radius','tear_ratio','iterations','damping'}) do
                eq(T.p[key],pp[key])
            end
            local now=0;for k in pairs(T.pinned) do now=now+1 end
            eq(now,npin)
            local rp=seamRailPins()
            eq(#rp,130)
            local n=0
            for j=0,12 do for _,i in ipairs({6,7,8,9,10,13,14,15,16,17}) do n=n+1;eq(rp[n],j*24+i) end end
            local worst=0
            for n2=1,#T.tri do
                local i=math.floor((n2-1)/2)%%23
                if i~=11 then
                    local tr=T.tri[n2]
                    local a,b,c=T.x[tr[1]],T.x[tr[2]],T.x[tr[3]]
                    for _,e in ipairs({{a,b},{b,c},{a,c}}) do
                        local d=math.sqrt((e[1][1]-e[2][1])^2+(e[1][2]-e[2][2])^2+(e[1][3]-e[2][3])^2)
                        if d>worst then worst=d end
                    end
                end
            end
            assert(worst<=math.sqrt(2)*0.001,'non-central edge '..worst)
        """ % SPEC)

    def test_parent_chain_unchanged_outside_anchor(self):
        from seam_slide_world import staged_source as slide_source
        from seam_wide_cell_world import _ANCHOR, _WIDE
        printed = (ROOT / 'src/tissue/cell_certificate_print.lua').read_text(encoding='utf-8')
        slide = slide_source(self.parent)
        wide = self.source
        self.assertEqual(self.parent.count(_ANCHOR), 1)
        self.assertTrue(wide.endswith('\n' + printed))
        core = wide[:-(len('\n' + printed))]
        self.assertEqual(core.replace('vmul(spec.u,' + _WIDE + ')', _ANCHOR), slide)
        self.assertEqual(core.count('(i-(nu-1)/2)*h'), 1, 'only the doc comment needle may remain')


if __name__ == '__main__':
    unittest.main()
