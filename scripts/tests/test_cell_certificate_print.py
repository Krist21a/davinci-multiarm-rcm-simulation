"""Exercise the passive cell-certificate print overlay under the sim stub.

Chunk order mirrors staging: fixture + controller sources + ruler_print.lua +
surface_seam.lua + cell_certificate_print.lua. A spying drawPrinted closure captures
the temporary layout mid-render; the real renderer is never invoked.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_lua_safety import FIXTURE, LEGACY_SOURCES as SOURCES, LuaRuntime  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CHUNK = (ROOT / 'src/tissue/ruler_print.lua', ROOT / 'src/tissue/surface_seam.lua',
         ROOT / 'src/tissue/cell_certificate_print.lua')
SPEC = ('{nu=24,nv=14,spacing=0.001,origin={0,0,0},u={1,0,0},v={0,1,0},'
        'pretension=0.05,anchor_sides=false,anchor_from_row=3,grasp_radius=0.0025}')


class CellCertificatePrintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = LuaRuntime()
        print(f'Lua safety runtime: {cls.runtime.description}', flush=True)
        cls.source = '\n'.join(p.read_text(encoding='utf-8') for p in (FIXTURE, *SOURCES, *CHUNK))

    def lua(self, body):
        self.runtime.run(self.source + '\nsysCall_init(); auto=false\n' + body)

    def test_temporary_layout_codes_split_and_restore(self):
        self.lua("""
            tissueCreate(%s)
            local base={x=T.x,tri=T.tri,colors=T.colors,springs=T.springs,pinned=T.pinned}
            local baseN,c0=#T.tri,#T.colors
            local spy={}
            local orig=drawPrinted
            drawPrinted=function()
                spy.x=T.x;spy.tri=T.tri;spy.colors=T.colors;spy.springs=T.springs;spy.pinned=T.pinned
            end
            assert(setCellCertificatePrint(true))
            drawPrinted=orig
            assert(T.x==base.x and T.tri==base.tri and T.colors==base.colors
                   and T.springs==base.springs and T.pinned==base.pinned)
            assert(spy.x~=base.x and spy.tri~=base.tri and spy.colors~=base.colors)
            assert(spy.springs==base.springs and spy.pinned==base.pinned)
            for k=1,#base.x do assert(spy.x[k]==base.x[k]) end
            for k=1,#base.colors do assert(spy.colors[k]==base.colors[k]) end
            eq(#spy.colors,c0+54)
            for j=0,13 do for s=0,1 do
                local c=spy.colors[c0+j*2+s+1]
                eq(c[1],14/255);eq(c[2],(24+14*j)/255);eq(c[3],(147+58*s)/255)
            end end
            for j=0,12 do for h=0,1 do
                local c=spy.colors[c0+28+j*2+h+1]
                eq(c[1],248/255);eq(c[2],(24+14*j)/255);eq(c[3],(147+58*h)/255)
            end end
            local function area3(a,b,c)
                local u1,u2,u3=b[1]-a[1],b[2]-a[2],b[3]-a[3]
                local v1,v2,v3=c[1]-a[1],c[2]-a[2],c[3]-a[3]
                return {0.5*(u2*v3-u3*v2),0.5*(u3*v1-u1*v3),0.5*(u1*v2-u2*v1)}
            end
            local centre,pos=0,1
            for n=1,baseN do
                local par=base.tri[n]
                local i=math.floor((n-1)/2)%%23
                local j=math.floor((n-1)/46)
                local h=(n-1)%%2
                local tagged={}
                if i~=11 then
                    for c=1,3 do
                        local col=(par[c]-1)%%24
                        if col==11 or col==12 then tagged[c]=true end
                    end
                end
                local anytag=tagged[1] or tagged[2] or tagged[3]
                local nchild=(i==11 or not anytag) and 1 or 7
                local pa=area3(base.x[par[1]],base.x[par[2]],base.x[par[3]])
                local acc={0,0,0}
                for q=0,nchild-1 do
                    local ch=spy.tri[pos+q]
                    for s=4,6 do eq(ch[s],par[s]) end
                    local ca=area3(spy.x[ch[1]],spy.x[ch[2]],spy.x[ch[3]])
                    for k=1,3 do acc[k]=acc[k]+ca[k] end
                end
                for k=1,3 do eq(acc[k],pa[k]) end
                if i==11 then
                    local ch=spy.tri[pos]
                    centre=centre+1
                    for s=1,3 do eq(ch[s],par[s]) end
                    eq(ch[7],c0+28+j*2+h+1)
                elseif anytag then
                    for c=1,3 do
                        local ch=spy.tri[pos+c-1]
                        eq(ch[1],par[c])
                        if tagged[c] then
                            local col=(par[c]-1)%%24
                            eq(ch[7],c0+math.floor((par[c]-1)/24)*2+(col==12 and 2 or 1))
                        else
                            eq(ch[7],par[7])
                        end
                    end
                    for q=3,6 do eq(spy.tri[pos+q][7],par[7]) end
                else
                    local ch=spy.tri[pos]
                    for s=1,7 do eq(ch[s],par[s]) end
                end
                pos=pos+nchild
            end
            eq(centre,26)
            eq(pos-1,#spy.tri)
        """ % SPEC)

    def test_off_calls_original_and_error_rolls_back(self):
        self.lua("""
            tissueCreate(%s)
            local base={x=T.x,tri=T.tri,colors=T.colors}
            local spy={}
            local orig=drawPrinted
            drawPrinted=function() spy.off=true end
            assert(setCellCertificatePrint(false))
            assert(spy.off)
            drawPrinted=function() error('boom') end
            local ok,err=pcall(setCellCertificatePrint,true)
            drawPrinted=orig
            assert(not ok,'renderer failure swallowed')
            assert(tostring(err):match('boom'))
            assert(T.x==base.x and T.tri==base.tri and T.colors==base.colors)
        """ % SPEC)


if __name__ == '__main__':
    unittest.main()
