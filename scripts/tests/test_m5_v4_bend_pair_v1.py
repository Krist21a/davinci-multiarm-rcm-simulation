"""Exercise the staged PBD projection against the canonical Lua mesh, not a mock law."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/task'),str(ROOT/'scripts/tests')]
from m5_v4_bend_staging import stage
from test_lua_safety import FIXTURE, SOURCES, LuaRuntime


class BendPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lua = LuaRuntime()
        cls.src = stage('\n'.join(f.read_text(encoding='utf8') for f in (FIXTURE,*SOURCES)))+"""
sysCall_init(); auto=false
tissueCreate({origin={0,0,0},u={1,0,0},v={0,1,0},nu=24,nv=14,spacing=.001,
    seam_mode='surface',guide_band_m=.0001,pretension=.05})
local function compare(a,b)
    eq(#a,#b)
    for k=1,#a do eq(a[k],b[k]) end
end
"""

    def test_partial_face_disables_hinge_and_preserves_labels_and_pins(self):
        self.lua.run(self.src+"""
local j,i=8,11
local a,b=T.tri[2*((j-1)*23+i)+2],T.tri[2*(j*23+i)+1]
for _,s in ipairs(T.springs) do s.alive=false end
for _,f in ipairs({a,b}) do for k=4,6 do T.springs[f[k]].alive=true end end
local d=b[3]
T.x[d][3]=.0004
T.pinned[d]={fixed=v3(T.x[d])}
local before=tissueNodes()
local labels=tissueConstraints()
bendTransverse()
compare(labels,tissueConstraints())
eq(T.x[d][3],.0004)
assert(math.abs(T.x[a[1]][3])>1e-10,'live non-flat hinge did not act')
T.springs[a[4]].alive=false
assert(not faceLive(a) and faceLive(b))
local broken=tissueConstraints()
local stationary=tissueNodes()
bendTransverse()
compare(stationary,tissueNodes())
compare(broken,tissueConstraints())
""")

    def test_split_halves_have_no_bending_coupling(self):
        self.lua.run(self.src+"""
for _,s in ipairs(T.springs) do
    if (((s.a-1)%24<12)~=((s.b-1)%24<12)) then s.alive=false end
end
local faces=surfaceSeamFaces()
for _,live in ipairs(faces) do eq(live,0) end
local labels=tissueConstraints()
local k=8*24+12
T.x[k][3]=.0004
local before=tissueNodes()
bendTransverse()
assert(T.x[k][3]<.0004,'left bank not exercised')
local after=tissueNodes()
for id=1,#T.x do
    if (id-1)%24>=12 then
        for c=1,3 do eq(after[3*(id-1)+c],before[3*(id-1)+c]) end
    end
end
compare(labels,tissueConstraints())
compare(faces,surfaceSeamFaces())
""")
