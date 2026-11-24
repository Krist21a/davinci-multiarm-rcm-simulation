"""Canonical src/tissue/tissue.lua tests: surface seam, guide band, ruler print, rest readback.

These exercise the CURRENT tissue source (the legacy fixtures in test_lua_safety.py run the
frozen src/tissue/legacy/tissue_v1.lua). The chain is the real controller + current tissue
with the same deterministic sim stub. No Python packages are required beyond the repo's.

Run: python -m unittest discover -s scripts/tests -p test_canonical_tissue.py -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lua_safety import FIXTURE, SOURCES, LuaRuntime  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Shared spec helpers: the flat 24x14 coupon at the origin, u along x, v along y (normal z).
LUA_HELPERS = r'''
local function create(extra)
    local spec={nu=24,nv=14,spacing=0.001,origin={0,0,0},u={1,0,0},v={0,1,0},
                pretension=0.08,anchor_from_row=3,grasp_radius=0.0025}
    for k,v in pairs(extra or {}) do spec[k]=v end
    tissueCreate(spec)
end
local function spring(a,b)
    for _,s in ipairs(T.springs) do
        if (s.a==a and s.b==b) or (s.a==b and s.b==a) then return s end
    end
end
local function kill(a,b) local s=spring(a,b);assert(s,'no spring '..a..','..b);s.alive=false end
'''


class CanonicalTissueTests(unittest.TestCase):
    """LuaRuntime proofs for the canonical development options."""

    @classmethod
    def setUpClass(cls):
        cls.runtime = LuaRuntime()
        print(f'canonical tissue runtime: {cls.runtime.description}', flush=True)
        # Fixture stub + real controller sources + CURRENT tissue.lua.
        cls.source = '\n'.join(p.read_text(encoding='utf-8')
                               for p in (FIXTURE, *SOURCES))

    def lua(self, body):
        self.runtime.run(self.source + '\nsysCall_init(); auto=false\n' + LUA_HELPERS + body)

    def test_reverse_diagonal_present_only_when_seam_disabled(self):
        self.lua(r'''
            create()
            for j=0,12 do
                assert(spring(j*24+13,(j+1)*24+12),'default sheet lost the reverse diagonal '..j)
            end
            create{seam_mode='surface'}
            for j=0,12 do
                assert(not spring(j*24+13,(j+1)*24+12),'surface sheet kept the reverse diagonal '..j)
                assert(spring(j*24+12,(j+1)*24+13),'surface sheet lost the face diagonal '..j)
            end
        ''')

    def test_seam_rejects_noncanonical_sheets(self):
        self.lua(r'''
            local function refused(extra)
                local ok=pcall(create,extra)
                assert(not ok,'accepted a noncanonical spec')
            end
            refused{seam_mode='surface',nu=2,nv=2}
            refused{seam_mode='surface',nodes={}}
            refused{seam_mode='surface',material_bands={}}
            refused{seam_mode='holes'}
            refused{guide_band_m=-0.001}
            refused{guide_band_m=0.0001,nu=2,nv=2}
            refused{ruler_print=true}                       -- requires surface mode
            refused{ruler_print='yes',seam_mode='surface'}
        ''')

    def test_seam_faces_and_shared_bridges(self):
        self.lua(r'''
            create{seam_mode='surface'}
            local f=surfaceSeamFaces()
            assert(#f==26,'expected 26 face flags, got '..#f)
            for _,v in ipairs(f) do assert(v==1,'a real face is dead at creation') end
            local n=0
            for _,refs in pairs(T.seamRefs) do
                n=n+1
                assert(#refs==1 or #refs==2,'crossing bond with '..#refs..' faces')
            end
            assert(n==27,'expected 27 shared bridges, got '..n)
            assert(surfaceSeamReleases()==0)
            assert(tissueStats().seam_releases==0)
        ''')

    def test_read_only_paths_do_not_release_orphans(self):
        self.lua(r'''
            create{seam_mode='surface'}
            -- Orphan row 0's transverse bond (12,13) without physics: kill cell 0's
            -- left face edge s2 (13,37). Its only incident central face is now dead.
            kill(13,37)
            local f=surfaceSeamFaces()
            assert(f[1]==0 and f[2]==1,'face flag did not follow the dead edge')
            local meshes=stub.mesh
            surfaceSeamFaces();surfaceSeamReleases();tissueStats();tissueConstraints();tissueNodes()
            assert(stub.mesh==meshes,'a read path rebuilt the rendered mesh')
            assert(surfaceSeamReleases()==0,'a read path released an orphan')
            assert(spring(12,13).alive,'the orphaned bond was removed without physics')
            -- render() itself must leave the orphan and every physical flag untouched.
            render()
            assert(surfaceSeamReleases()==0,'render released an orphan')
            assert(spring(12,13).alive,'render removed the orphaned bond')
            assert(surfaceSeamFaces()[1]==0,'render changed a face flag')
        ''')

    def test_physical_step_releases_orphans_without_cut_count(self):
        self.lua(r'''
            -- iterations=0 keeps the solver from relaxing the imposed stretch before
            -- the tear check; the +50 mm offset then tears the incident face edges.
            create{seam_mode='surface',iterations=0}
            T.x[37]={T.x[37][1],T.x[37][2],T.x[37][3]+0.05}
            T.prev[37]=T.x[37]
            tissueStep()
            local st=tissueStats()
            assert(st.torn>=1,'the stretched face edges did not tear')
            assert(st.seam_releases>=1,'the orphaned centre bonds were not released')
            assert(st.seam_releases==surfaceSeamReleases())
            assert(st.cut==0,'orphan releases were counted as direct cuts')
            assert(not spring(12,13).alive,'orphaned transverse bond is still load bearing')
            local f=surfaceSeamFaces()
            assert(f[1]==0,'dead face still flagged')
        ''')

    def test_cut_path_calls_release_before_render(self):
        self.lua(r'''
            -- Sheet parked off the blade line: the closure sweeps nothing, but the
            -- cut path still releases the externally orphaned centre bond.
            create{seam_mode='surface',origin={0.1,0.1,0.1}}
            kill(13,37)                              -- externally orphaned transverse bond
            assert(tissueCut(2,0.5,0.2)>=0)          -- a real closure event, cut or not
            assert(surfaceSeamReleases()==1,'cut path did not release the orphan')
            assert(not spring(12,13).alive)
            assert(tissueStats().seam_releases==1)
        ''')

    def test_guide_skips_pins_and_clamps_only_normal(self):
        self.lua(r'''
            create{seam_mode='surface',guide_band_m=0.0001,iterations=1}
            -- Unpinned lane node (row 5, col 7): normal offset clamped, tangential left free.
            local k=5*24+8
            local p={T.x[k][1],T.x[k][2],T.x[k][3]}
            T.x[k]={p[1]+0.002,p[2],p[3]+0.005}
            T.prev[k]=T.x[k]
            -- Pinned lane node (row 5, col 8): the pin snap target holds it 5 mm off plane.
            local k2=5*24+9
            local pinned_z=T.x[k2][3]+0.005
            T.pinned[k2]={fixed={T.x[k2][1],T.x[k2][2],pinned_z}}
            tissueStep()
            local q=T.x[k]
            assert(math.abs(q[3]-p[3])<=0.0001+1e-12,'normal component not clamped to the band')
            assert(math.abs(q[1]-p[1])>0.0001,'tangential component was clamped too')
            assert(T.x[k2][3]==pinned_z,'the guide moved a pinned node')
            -- The lane covers exactly rows 0..12 x columns 6..10 and 13..17.
            local ids={}
            for _,kk in ipairs(guideKeys) do ids[kk]=true end
            local n=0
            for j=0,12 do
                for _,i in ipairs({6,7,8,9,10,13,14,15,16,17}) do
                    assert(ids[j*24+i+1],'missing lane node '..j..','..i)
                    n=n+1
                end
            end
            assert(n==130)
        ''')

    def test_guide_rejects_missing_origin(self):
        self.lua(r'''
            local ok=pcall(tissueCreate,{nu=24,nv=14,spacing=0.001,u={1,0,0},v={0,1,0},
                                         guide_band_m=0.0001})
            assert(not ok,'guide accepted a missing origin')
        ''')

    def test_transverse_rest_lengths_are_measured(self):
        self.lua(r'''
            create()
            local r=tissueTransverseRestLengths()
            assert(#r==14,'expected 14 rests, got '..#r)
            for _,v in ipairs(r) do
                assert(v>0 and math.abs(v-0.001*(1-0.08))<1e-12,'nominal rest mismatch')
            end
            create{seam_mode='surface'}
            local r2=tissueTransverseRestLengths()
            assert(#r2==14)
            for _,v in ipairs(r2) do assert(v>0) end
        ''')

    def test_ruler_print_marks_exact_cell_ordinals(self):
        self.lua(r'''
            create{seam_mode='surface',ruler_print=true}
            assert(#T.colors==27,'expected 1 tissue + 26 marker colours')
            -- Centre-column cells (i=11 left, i=12 right), row j: colour index 2+side*13+j.
            assert(T.tri[2*(0*23+11)+1][7]==15)     -- i=11, odd tri -> side 1, j=0
            assert(T.tri[2*(0*23+11)+2][7]==2)      -- i=11, even tri -> side 0, j=0
            assert(T.tri[2*(0*23+12)+1][7]==15)     -- i=12 -> side 1
            assert(T.tri[2*(12*23+11)+1][7]==2+13+12)  -- row 12 left cell
            assert(T.tri[2*(12*23+12)+2][7]==2+13+12)  -- row 12 right cell
            -- An outer cell keeps the tissue colour.
            assert(T.tri[2*(5*23+3)+1][7]==1)
            -- Marker colour formula sample: side 0 row 3 -> {r,235,k}
            local c=T.colors[2+3]
            assert(c[2]==235/255,'left cell green channel differs from the ruler formula')
        ''')

    def test_indexed_print_marks_full_face_material_addresses(self):
        self.lua(r'''
            create{seam_mode='surface',indexed_print=true}
            assert(#T.colors==105,'expected 1 tissue + 104 address colours, got '..#T.colors)
            for j=0,12 do
                -- Centre cell i=11 splits lanes 4/3 across the shared diagonal.
                assert(T.tri[2*(j*23+11)+1][7]==2+8*j+4,'centre odd lane '..j)
                assert(T.tri[2*(j*23+11)+2][7]==2+8*j+3,'centre even lane '..j)
                -- Outboard lanes run single-tri-wide on the left, two-wide on the right.
                assert(T.tri[2*(j*23+8)+1][7]==2+8*j+0,'left outboard lane0 '..j)
                assert(T.tri[2*(j*23+9)+1][7]==2+8*j+1,'left outboard lane1 '..j)
                assert(T.tri[2*(j*23+13)+1][7]==2+8*j+6,'right lane6 '..j)
                assert(T.tri[2*(j*23+14)+1][7]==2+8*j+7,'right lane7 '..j)
            end
            -- Row 7: every lane emits exactly the registered address formula.
            for k=0,7 do
                local c=T.colors[2+8*7+k]
                assert(c[1]==(42+13*7)/255 and c[2]==(50+21*k)/255
                       and c[3]==(42+25*((3*7+5*k)%8))/255,'emission formula k='..k)
            end
            -- Conflict, non-boolean and non-surface requests fail BEFORE the live
            -- sheet is removed: T must be the same table afterwards.
            local live=T
            for _,extra in ipairs({{seam_mode='surface',ruler_print=true,indexed_print=true},
                                   {seam_mode='surface',indexed_print='yes'},
                                   {indexed_print=true}}) do
                assert(not pcall(create,extra),'accepted '..tostring(extra.indexed_print))
                assert(T==live,'a refused spec destroyed the live sheet')
            end
        ''')


if __name__ == '__main__':
    unittest.main()
