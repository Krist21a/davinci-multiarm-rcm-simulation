-- Opt-in material ruler print for the RGB-D endpoint acquisition (config/task/ruler_endpoint_v1.json).
-- Appended to the controller code by scripts/task/ruler_world.py, in the same chunk as tissue.lua,
-- so T, render() and the tissueCreate it wraps share its scope. It only recolours existing
-- triangles of the plain 24x14 sheet; the mesh, constraints and cut logic are unchanged.
local createBaseSheet=tissueCreate
function tissueCreate(spec)
    local n=createBaseSheet(spec)
    assert(T.nu==24 and T.nv==14 and not spec.material_bands)
    for side=0,1 do
        for j=0,12 do
            local r,k=32+48*(j%4),35+46*math.floor(j/4)
            T.colors[#T.colors+1]={r/255,(side==0 and 235 or k)/255,(side==0 and k or 235)/255}
        end
    end
    for idx,tr in ipairs(T.tri) do
        local cell=math.floor((idx-1)/2)
        local i,j=cell%23,math.floor(cell/23)
        if i>=7 and i<=15 then
            local side=(i>11 or (i==11 and idx%2==1)) and 1 or 0
            tr[7]=2+side*13+j
        end
    end
    render()
    return n
end
