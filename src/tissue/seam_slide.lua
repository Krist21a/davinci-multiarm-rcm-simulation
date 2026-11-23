-- Frictionless normal-guide layer for config/task/seam_slide_v1.json. Injected before
-- tissueStep on the staged copy, after T, vdot, vsub, vmul and clamp are in scope; the
-- create wrapper in src/tissue/seam_slide_create.lua configures slideAxis, slideOrigin
-- and slideKeys once per sheet. Each distance-solve iteration projects every unpinned
-- guide node back into a +/-0.1 mm band about the plane through slideOrigin normal to
-- slideAxis; tangential motion, pins, springs and the tear threshold are untouched.
local slideAxis,slideOrigin,slideKeys
local function projectSlide()
    if not T or not slideKeys then return end
    for _,k in ipairs(slideKeys) do
        if not T.pinned[k] then
            local p=T.x[k]
            local d=vdot(vsub(p,slideOrigin),slideAxis)
            local e=d-clamp(d,-0.0001,0.0001)
            if e~=0 then T.x[k]=vsub(p,vmul(slideAxis,e)) end
        end
    end
end
