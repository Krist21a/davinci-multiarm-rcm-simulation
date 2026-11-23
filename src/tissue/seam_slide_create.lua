-- Create wrapper for the slide-guide fixture (config/task/seam_slide_v1.json), staged
-- after the printed surface-seam parent. Computes the unit guide normal from the
-- flat-sheet spec itself (cross(spec.u,spec.v)) and registers the 130 lane nodes
-- (material rows 0..12, columns 6..10 and 13..17, ascending) the projection layer
-- guides. No pins, springs, faces or ink are changed. seamRailPins() here is a
-- compatibility getter returning the guide node IDs for the collector's readback;
-- they are NOT fixed pins -- nothing is added to T.pinned.
local createSeamSlide=tissueCreate
function tissueCreate(spec)
    local count=createSeamSlide(spec)
    assert(T.nu==24 and T.nv==14 and not spec.nodes and not spec.material_bands
           and spec.origin and spec.u and spec.v)
    local n={spec.u[2]*spec.v[3]-spec.u[3]*spec.v[2],
             spec.u[3]*spec.v[1]-spec.u[1]*spec.v[3],
             spec.u[1]*spec.v[2]-spec.u[2]*spec.v[1]}
    local L=vlen(n)
    assert(L>1e-12)
    slideAxis=vmul(n,1/L)
    slideOrigin=v3(spec.origin)
    slideKeys={}
    for j=0,12 do for _,i in ipairs({6,7,8,9,10,13,14,15,16,17}) do
        slideKeys[#slideKeys+1]=j*24+i+1
    end end
    return count
end
function seamRailPins()
    local out={}
    for _,k in ipairs(slideKeys or {}) do out[#out+1]=k-1 end
    return out
end
