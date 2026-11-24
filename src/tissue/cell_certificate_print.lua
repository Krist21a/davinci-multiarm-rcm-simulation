-- Passive colour-certificate print overlay, staged after src/tissue/surface_seam.lua in
-- the same controller chunk (original drawPrinted and the T/v3/vadd/vsub/vmul helpers
-- are in scope). Renderer-only: setCellCertificatePrint(true) swaps in a temporary
-- split-triangle layout whose corner-cap and face inks mark the original material
-- triangles, calls the ORIGINAL mesh renderer inside pcall, and always restores
-- T.x/T.tri/T.colors -- including on renderer failure. No springs, pins, constraints or
-- damage state are read or changed; every emitted triangle inherits its parent's three
-- real spring ids, so no mark is drawn on a nonexistent host face.
function setCellCertificatePrint(on)
    if not on then drawPrinted() return true end
    local oldX,oldTri,oldColors=T.x,T.tri,T.colors
    local verts,colors={},{}
    for k,p in ipairs(oldX) do verts[k]=p end
    for k,c in ipairs(oldColors) do colors[k]=c end
    local cap0=#colors
    for j=0,13 do for s=0,1 do
        colors[#colors+1]={14/255,(24+14*j)/255,(147+58*s)/255}
    end end
    local face0=#colors
    for j=0,12 do for h=0,1 do
        colors[#colors+1]={248/255,(24+14*j)/255,(147+58*h)/255}
    end end
    local tris={}
    for n=1,#oldTri do
        local tr=oldTri[n]
        local i=math.floor((n-1)/2)%23
        local j=math.floor((n-1)/46)
        local h=(n-1)%2
        if i==11 then
            tris[#tris+1]={tr[1],tr[2],tr[3],tr[4],tr[5],tr[6],face0+j*2+h+1}
        else
            local tagged={}
            for c=1,3 do
                local col=(tr[c]-1)%24
                if col==11 or col==12 then tagged[c]=true end
            end
            if not (tagged[1] or tagged[2] or tagged[3]) then
                tris[#tris+1]={tr[1],tr[2],tr[3],tr[4],tr[5],tr[6],tr[7]}
            else
                local p={oldX[tr[1]],oldX[tr[2]],oldX[tr[3]]}
                local aid,bid={},{}
                for c=1,3 do
                    local nx=c%3+1
                    local pv=c==1 and 3 or c-1
                    aid[c]=#verts+1
                    verts[#verts+1]=vadd(p[c],vmul(vsub(p[nx],p[c]),0.15))
                    bid[c]=#verts+1
                    verts[#verts+1]=vadd(p[c],vmul(vsub(p[pv],p[c]),0.15))
                end
                for c=1,3 do
                    local cap=nil
                    if tagged[c] then
                        cap=cap0+math.floor((tr[c]-1)/24)*2+(((tr[c]-1)%24)==12 and 2 or 1)
                    end
                    tris[#tris+1]={tr[c],aid[c],bid[c],tr[4],tr[5],tr[6],cap or tr[7]}
                end
                local hex={aid[1],bid[2],aid[2],bid[3],aid[3],bid[1]}
                for q=1,4 do
                    tris[#tris+1]={hex[1],hex[q+1],hex[q+2],tr[4],tr[5],tr[6],tr[7]}
                end
            end
        end
    end
    T.x,T.tri,T.colors=verts,tris,colors
    local ok,err=pcall(drawPrinted)
    T.x,T.tri,T.colors=oldX,oldTri,oldColors
    if not ok then error(err) end
    return true
end
