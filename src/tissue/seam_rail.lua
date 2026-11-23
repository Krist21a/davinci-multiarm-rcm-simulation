-- Support-rail fixture layer for config/task/seam_rail_v1.json, staged after
-- src/tissue/surface_seam.lua in the same controller chunk. railLeft is configured once
-- as a chunk local immediately before this append on the staged copy. Adds fourteen
-- fixed anchors on the paired rail columns railLeft / 23-railLeft, rows 6..12, to the
-- untouched seam coupon; nodes, springs, faces, cut radius, strengths and the renderer
-- are unchanged, and no ink or label is ever read to change behaviour.
local createSeam=tissueCreate
function tissueCreate(spec)
    assert(railLeft==8 or railLeft==9)
    assert(not spec.anchor_sides and not spec.nodes and not spec.material_bands)
    local count=createSeam(spec)
    local pinned={}
    for j=6,12 do for _,i in ipairs({railLeft,23-railLeft}) do
        local k=j*24+i+1
        assert(not T.pinned[k]); T.pinned[k]={fixed=v3(T.x[k])}; pinned[#pinned+1]=k-1
    end end
    T.railPins=pinned
    return count
end
function seamRailPins() return T and T.railPins or {} end
