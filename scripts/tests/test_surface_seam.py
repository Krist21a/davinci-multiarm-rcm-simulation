"""Execute the surface-seam tissue extension under the same sim stub as test_lua_safety.

The seam chunk is FIXTURE + controller sources + ruler_print.lua + surface_seam.lua, the
order SurfaceSeamWorld stages them. Baseline comparisons call the chunk's own captured
createPrinted/createBaseSheet wrappers, so no extension logic is re-implemented here.
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
SEAM = (ROOT / 'src/tissue/ruler_print.lua', ROOT / 'src/tissue/surface_seam.lua')
SPEC = ('{nu=24,nv=14,spacing=0.001,origin={0,0,0},u={1,0,0},v={0,1,0},'
        'pretension=0.05,anchor_sides=false,anchor_from_row=3,grasp_radius=0.0025}')


class SurfaceSeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = LuaRuntime()
        print(f'Lua safety runtime: {cls.runtime.description}', flush=True)
        cls.seam = '\n'.join(p.read_text(encoding='utf-8') for p in (FIXTURE, *SOURCES, *SEAM))
        cls.base = '\n'.join(p.read_text(encoding='utf-8') for p in (FIXTURE, *SOURCES))

    def lua(self, body, source=None):
        self.runtime.run((source or self.seam) + '\nsysCall_init(); auto=false\n' + body)

    def test_surviving_state_matches_printed_baseline(self):
        self.lua("""
            local spec=%s
            createPrinted(spec)
            local base={x=T.x,tri=T.tri,pinned=T.pinned,springs={}}
            for _,s in ipairs(T.springs) do
                base.springs[math.min(s.a,s.b)..'_'..math.max(s.a,s.b)]=s.rest
            end
            local base_pairs={}
            for i,tr in ipairs(T.tri) do
                base_pairs[i]={}
                for q=4,6 do
                    local s=T.springs[tr[q]]
                    base_pairs[i][q]=math.min(s.a,s.b)..'_'..math.max(s.a,s.b)
                end
            end
            tissueCreate(spec)
            eq(#T.x,#base.x)
            for k=1,#T.x do for c=1,3 do eq(T.x[k][c],base.x[k][c]) end end
            eq(#T.tri,#base.tri)
            for i,tr in ipairs(T.tri) do
                for q=1,3 do eq(tr[q],base.tri[i][q]) end
                eq(tr[7],base.tri[i][7])
                for q=4,6 do
                    local s=T.springs[tr[q]]
                    assert(math.min(s.a,s.b)..'_'..math.max(s.a,s.b)==base_pairs[i][q])
                end
            end
            local pn,bp=0,0
            for k in pairs(T.pinned) do pn=pn+1;assert(base.pinned[k]~=nil) end
            for k in pairs(base.pinned) do bp=bp+1;assert(T.pinned[k]~=nil) end
            eq(pn,bp)
            for _,s in ipairs(T.springs) do
                eq(s.rest,base.springs[math.min(s.a,s.b)..'_'..math.max(s.a,s.b)])
            end
        """ % SPEC)

    def test_exactly_13_removed_27_crossing_no_precut(self):
        self.lua("""
            local spec=%s
            createPrinted(spec)
            local before=#T.springs
            tissueCreate(spec)
            eq(before-#T.springs,13)
            local crossing,transverse,diagonal=0,0,0
            for k,s in ipairs(T.springs) do
                local ca,cb=(s.a-1)%%24,(s.b-1)%%24
                if (ca==11 and cb==12) or (ca==12 and cb==11) then
                    assert(s.alive)
                    crossing=crossing+1
                    local ra,rb=math.floor((s.a-1)/24),math.floor((s.b-1)/24)
                    if ra==rb then transverse=transverse+1 else diagonal=diagonal+1 end
                end
            end
            eq(crossing,27);eq(transverse,14);eq(diagonal,13)
            eq(T.seamReleases,0);eq(surfaceSeamReleases(),0);eq(tissueStats().cut,0)
            local faces=surfaceSeamFaces()
            eq(#faces,26)
            for i=1,26 do eq(faces[i],1) end
        """ % SPEC)

    def test_orphan_release_is_face_supported_and_counted_once(self):
        self.lua("""
            tissueCreate(%s)
            -- Kill one non-crossing edge of a face: the face dies but its centre bonds stay
            -- alive, so the render pass has to release the orphaned ones itself.
            local function dropFace(tr,q) T.springs[tr[q]].alive=false end
            local function bond(a,b)
                for k,s in ipairs(T.springs) do
                    if (s.a==a and s.b==b) or (s.a==b and s.b==a) then return k end
                end
            end
            -- Cell column 11, row 0 is T.tri[23]={k00,k10,k11,s1,s2,s3} and
            -- T.tri[24]={k00,k11,k01,s3,s5,s4}; bonds (12,13) transverse row 0,
            -- (12,37) positive diagonal, (36,37) transverse row 1 shared with cell (11,1).
            local trans0,diag0,trans1=bond(12,13),bond(12,37),bond(36,37)
            dropFace(T.tri[23],5);render()    -- face 1 dead via its column-12 edge
            assert(not T.springs[trans0].alive,'sole-face crossing survived its face')
            eq(T.seamReleases,1)
            assert(T.springs[diag0].alive,'crossing died while a face still lives')
            dropFace(T.tri[24],6);render()    -- face 2 dead via its column-11 edge
            assert(not T.springs[diag0].alive,'orphan crossing kept after last face')
            eq(T.seamReleases,2)
            assert(T.springs[trans1].alive,'neighbour crossing lost shared support')
            render()
            eq(T.seamReleases,2)
            local faces=surfaceSeamFaces()
            eq(faces[1],0);eq(faces[2],0);eq(faces[3],1)
        """ % SPEC)

    def test_unmodified_baseline_keeps_the_extra_diagonal(self):
        self.lua("""
            local spec=%s
            for _,create in ipairs({createBaseSheet,createPrinted}) do
                create(spec)
                local reverse=0
                for k,s in ipairs(T.springs) do
                    local ca,cb=(s.a-1)%%24,(s.b-1)%%24
                    if (ca==11 and cb==12) or (ca==12 and cb==11) then
                        local ra,rb=math.floor((s.a-1)/24),math.floor((s.b-1)/24)
                        local r11,r12=(ca==11) and ra or rb,(ca==12) and ra or rb
                        if r11==r12+1 then reverse=reverse+1 end
                    end
                end
                eq(reverse,13)
            end
        """ % SPEC)

    def test_vision_control_source_is_refused(self):
        import surface_seam_world as ssw
        with patch.object(ssw.SimWorld, '__init__', lambda self, *a, **k: setattr(self, 'vision', True)):
            with self.assertRaises(ValueError):
                ssw.SurfaceSeamWorld(None, None, None, None)

    def test_truth_is_seam_truth_and_reads_zero_without_sheet(self):
        import surface_seam_world as ssw
        with patch.object(ssw.SimWorld, '__init__', lambda self, *a, **k: setattr(self, 'vision', False)):
            w = ssw.SurfaceSeamWorld(None, None, None, None)
        self.assertIsInstance(w.truth, ssw.SeamTruth)
        self.assertIs(w.truth.world, w)
        truth = ssw.SeamTruth(SimpleNamespace(sheet_origin=None))
        self.assertEqual(truth.cut_length(), 0.0)
        self.assertEqual(truth.reads['cut_length'], 1)


if __name__ == '__main__':
    unittest.main()
