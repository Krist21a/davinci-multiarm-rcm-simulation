-- Surface-seam coupon layer for config/task/surface_seam_pilot_v1.json, staged after
-- src/tissue/ruler_print.lua in the same controller chunk. One mechanical change on the
-- printed 24x14 coupon: at creation the 13 non-face diagonals crossing columns 11/12 are
-- omitted; every remaining centre crossing bond is load bearing only while an incident
-- central triangle survives (coarse face-supported central strip). Nodes, pins, rest
-- lengths, cut radius, iteration order and the renderer are otherwise untouched, and no
-- ink, length or label is ever read to change behaviour.
local createPrinted=tissueCreate
local drawPrinted=render
local function live(tr)
    return T.springs[tr[4]].alive and T.springs[tr[5]].alive and T.springs[tr[6]].alive
end
render=function()
    if T and T.seamRefs then
        for k,refs in pairs(T.seamRefs) do
            local s=T.springs[k]
            if s.alive then
                local ok=false
                for _,tr in ipairs(refs) do
                    if live(tr) then ok=true;break end
                end
                if not ok then
                    s.alive=false
                    T.seamReleases=T.seamReleases+1
                end
            end
        end
    end
    drawPrinted()
end
function tissueCreate(spec)
    local n=createPrinted(spec)
    assert(T.nu==24 and T.nv==14 and not spec.material_bands and not spec.nodes)
    -- Node (i,j) is index j*24+i+1; a centre crossing has endpoint columns 11 and 12.
    local drop={}
    for k,s in ipairs(T.springs) do
        local ca,cb=(s.a-1)%24,(s.b-1)%24
        if (ca==11 and cb==12) or (ca==12 and cb==11) then
            local ra,rb=math.floor((s.a-1)/24),math.floor((s.b-1)/24)
            local r11,r12=(ca==11) and ra or rb,(ca==12) and ra or rb
            if r11==r12+1 then drop[k]=true end    -- reverse diagonal only
        end
    end
    local remap,kept,removed={},{},0
    for k,s in ipairs(T.springs) do
        if drop[k] then
            removed=removed+1
        else
            kept[#kept+1]=s
            remap[k]=#kept
        end
    end
    assert(removed==13)
    for _,tr in ipairs(T.tri) do
        for q=4,6 do
            assert(not drop[tr[q]])
            tr[q]=remap[tr[q]]
        end
    end
    T.springs=kept
    T.seamRefs={}
    T.seamReleases=0
    local crossing=0
    for k,s in ipairs(T.springs) do
        local ca,cb=(s.a-1)%24,(s.b-1)%24
        if (ca==11 and cb==12) or (ca==12 and cb==11) then
            local refs={}
            for _,tr in ipairs(T.tri) do
                if tr[4]==k or tr[5]==k or tr[6]==k then refs[#refs+1]=tr end
            end
            assert(s.alive and (#refs==1 or #refs==2))
            T.seamRefs[k]=refs
            crossing=crossing+1
        end
    end
    assert(crossing==27)
    return n
end

-- Flat 1/0 liveness of the two real triangles of each cell-column-11 material row, in order.
function surfaceSeamFaces()
    if not T then return {} end
    local out={}
    for j=0,12 do
        local base=2*(j*23+11)
        for k=1,2 do
            local tr=T.tri[base+k]
            out[#out+1]=(tr and live(tr)) and 1 or 0
        end
    end
    return out
end

function surfaceSeamReleases()
    return (T and T.seamReleases) or 0
end
