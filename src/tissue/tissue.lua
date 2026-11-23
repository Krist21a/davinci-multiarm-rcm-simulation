-- Cuttable, graspable tissue sheet, appended after src/instruments/capability_guard.lua in
-- scenes/davinci_si_instruments.ttt. Kinematic model only: a grid of nodes joined by distance constraints
-- (position-based dynamics), pre-tensioned between pinned anchors. No contact with the instruments, no mass,
-- no fluid, no thermal or electrosurgical effect.
--   tissueCreate(spec)   spec={origin, u, v, nu, nv, spacing, pretension, anchor_sides, iterations, damping,
--                        tear_ratio, cut_radius, grasp_radius, color, seam_mode, guide_band_m,
--                        ruler_print}; node (i,j) = origin+(i-(nu-1)/2)*h*u+j*h*v;
--                        row j=nv-1 (and columns 0, nu-1 from row 1 when anchor_sides) are anchored.
--   tissueGrasp(index)   needs 'grasp_tissue': pins nodes within grasp_radius of the closed jaws to the tool tip.
--   tissueRelease(index)
--   closing scissors     actual opening >= 0.15 rad arms one closing cycle. Every accepted smaller angle cuts
--                        ONLY the two blade sweeps from the old angle to the new angle, including small steps
--                        below 0.15. Closure ends the cycle; reopening below 0.15 cancels it.
--   tissueStats(), tissueNodes(), tissueRemove()
-- A constraint stretched beyond tear_ratio * rest length breaks and counts as torn (tissue damage).
--
-- Canonical development coupon options on the flat 24x14 sheet only (spec.nodes/material_bands excluded):
--   spec.seam_mode='surface'  the 13 non-face diagonals crossing the centre columns are not created;
--                        every remaining centre crossing bond is load bearing only while an incident
--                        central triangle survives. Orphaned bonds are released by releaseUnsupported()
--                        from the cut and step paths -- never by the renderer -- and counted separately
--                        in T.seamReleases (not T.cut).
--   spec.guide_band_m      normal-only frictionless guide: each constraint iteration clamps the
--                        unpinned lane nodes (rows 0..12, columns 6..10 and 13..17) to +/- band about
--                        the plane through spec.origin normal to cross(spec.u,spec.v). Tangential
--                        motion, pins, springs and the tear threshold are untouched.
--   spec.ruler_print=true  immutable 13 left/right marker colours by triangle ordinal, exactly the
--                        ruler_print.lua formulas; requires seam_mode='surface'; no dynamic ink.
local T=nil
local cutArmed={}
local guideKeys,guideOrigin,guideAxis,guideBand
local CLOSED_EPS=1e-6   -- numerical joint-position tolerance, not a tissue/contact model
local function finite(x) return type(x)=='number' and x==x and math.abs(x)<math.huge end
local function v3(a) return {a[1],a[2],a[3]} end
local function vadd(a,b) return {a[1]+b[1],a[2]+b[2],a[3]+b[3]} end
local function vsub(a,b) return {a[1]-b[1],a[2]-b[2],a[3]-b[3]} end
local function vmul(a,k) return {a[1]*k,a[2]*k,a[3]*k} end
local function vdot(a,b) return a[1]*b[1]+a[2]*b[2]+a[3]*b[3] end
local function vlen(a) return math.sqrt(vdot(a,a)) end
local function clamp(x,a,b) return math.max(a,math.min(b,x)) end

-- Squared distance between segments p1-q1 and p2-q2.
local function segDist2(p1,q1,p2,q2)
    local d1,d2,r=vsub(q1,p1),vsub(q2,p2),vsub(p1,p2)
    local a,e,f=vdot(d1,d1),vdot(d2,d2),vdot(d2,r)
    local s,t
    if a<1e-18 and e<1e-18 then return vdot(r,r) end
    if a<1e-18 then s=0;t=clamp(f/e,0,1)
    else
        local c=vdot(d1,r)
        if e<1e-18 then t=0;s=clamp(-c/a,0,1)
        else
            local b=vdot(d1,d2);local den=a*e-b*b
            s=den>1e-18 and clamp((b*f-c*e)/den,0,1) or 0
            t=(b*s+f)/e
            if t<0 then t=0;s=clamp(-c/a,0,1) elseif t>1 then t=1;s=clamp((b-c)/a,0,1) end
        end
    end
    local d=vsub(vadd(p1,vmul(d1,s)),vadd(p2,vmul(d2,t)))
    return vdot(d,d)
end

-- The sheet is a mesh shape rebuilt every step (drawing objects are not rendered by vision sensors, so the
-- endoscope would not see them). Unlit flat colour for colour segmentation; not respondable, not in any
-- collision pair.
local function render()
    local verts,indices={},{}
    for i=1,#T.colors do indices[i]={} end
    local function live(tr)
        return T.springs[tr[4]].alive and T.springs[tr[5]].alive and T.springs[tr[6]].alive
    end
    if T.p.visualGap then
        -- Only appearance recedes: the solver, spring state and independent scorer
        -- still use T.x. A surviving face borders a physical fracture if its
        -- neighbouring triangle no longer has all three load-bearing edges.
        for _,tr in ipairs(T.tri) do
            if live(tr) then
                local col=0
                for q=1,3 do col=col+(tr[q]-1)%T.nu end
                col=col/3
                local free=false
                local row=math.floor((tr[1]-1)/T.nu)
                if row>=T.p.visualGapStartRow and col>=9.5 and col<=13.5 then
                    for q=4,6 do
                        local refs=T.edgeRefs[tr[q]]
                        if #refs==2 then
                            local other=refs[1]==tr and refs[2] or refs[1]
                            if not live(other) then free=true;break end
                        end
                    end
                end
                local sign=col<T.nu/2-0.5 and -1 or 1
                local shift=free and sign*T.p.visualGap or 0
                local idx=indices[tr[7]]
                for q=1,3 do
                    local x=T.x[tr[q]]
                    idx[#idx+1]=#verts/3
                    local vertexCol=(tr[q]-1)%T.nu
                    local offset=(T.p.visualGapStyle=='edge' and
                                  vertexCol~=T.nu/2-1 and vertexCol~=T.nu/2)
                                  and 0 or shift
                    for a=1,3 do verts[#verts+1]=x[a]+offset*T.visualAxis[a] end
                end
            end
        end
    else
        for _,x in ipairs(T.x) do verts[#verts+1]=x[1];verts[#verts+1]=x[2];verts[#verts+1]=x[3] end
        for _,tr in ipairs(T.tri) do
            if live(tr) then
                local idx=indices[tr[7]]
                idx[#idx+1]=tr[1]-1;idx[#idx+1]=tr[2]-1;idx[#idx+1]=tr[3]-1
            end
        end
    end
    if #T.shapes>0 then sim.removeObjects(T.shapes) end
    T.shapes={}
    for band,idx in ipairs(indices) do
        if #idx>0 then
            local shape=sim.createMeshShape(0,0,verts,idx)
            T.shapes[#T.shapes+1]=shape
            sim.setObjectAlias(shape,band==1 and 'TissueSheet' or ('TissueBand'..band))
            sim.setShapeColor(shape,nil,sim.colorcomponent_ambient_diffuse,{0,0,0})
            sim.setShapeColor(shape,nil,sim.colorcomponent_specular,{0,0,0})
            sim.setShapeColor(shape,nil,sim.colorcomponent_emission,T.colors[band])
            sim.setObjectInt32Param(shape,sim.shapeintparam_static,1)
            sim.setObjectInt32Param(shape,sim.shapeintparam_respondable,0)
        end
    end
    if T.bridgeRefs then
        for lane,refs in ipairs(T.bridgeRefs) do
            local intact=T.springs[refs[1]].alive or T.springs[refs[2]].alive or
                         T.springs[refs[3]].alive
            if intact or T.bridgeFractureFaces then
                local i=lane+7
                local a,b,c,d=T.x[8*T.nu+i+1],T.x[8*T.nu+i+2],
                              T.x[9*T.nu+i+1],T.x[9*T.nu+i+2]
                local function patch(umin,umax,color,alias)
                    local vertices={}
                    for _,sign in ipairs({1,-1}) do
                        for _,uv in ipairs({{umin,0.18},{umax,0.18},{umax,0.82},{umin,0.82}}) do
                            local u,v=uv[1],uv[2]
                            for axis=1,3 do
                                vertices[#vertices+1]=(1-u)*(1-v)*a[axis]+u*(1-v)*b[axis]+
                                    (1-u)*v*c[axis]+u*v*d[axis]+sign*0.00015*T.bridgeNormal[axis]
                            end
                        end
                    end
                    local shape=sim.createMeshShape(0,0,vertices,{0,1,2,0,2,3,4,6,5,4,7,6})
                    T.shapes[#T.shapes+1]=shape
                    sim.setObjectAlias(shape,alias)
                    sim.setShapeColor(shape,nil,sim.colorcomponent_ambient_diffuse,{0,0,0})
                    sim.setShapeColor(shape,nil,sim.colorcomponent_specular,{0,0,0})
                    sim.setShapeColor(shape,nil,sim.colorcomponent_emission,color)
                    sim.setObjectInt32Param(shape,sim.shapeintparam_static,1)
                    sim.setObjectInt32Param(shape,sim.shapeintparam_respondable,0)
                end
                if intact then
                    patch(0.18,0.82,T.bridgeColors[lane],'TissueBridge'..i)
                else
                    patch(0.10,0.45,T.bridgeFractureColors[1],'TissueFractureL'..i)
                    patch(0.55,0.90,T.bridgeFractureColors[2],'TissueFractureR'..i)
                end
            end
        end
    end
end

function tissueRemove()
    if T and #T.shapes>0 then sim.removeObjects(T.shapes) end
    T=nil;cutArmed={}
end

function tissueCreate(spec)
    -- Validate the spec before removing any existing sheet: a refused option must
    -- not destroy live state.
    assert(spec.seam_mode==nil or spec.seam_mode=='surface')
    local surface=spec.seam_mode=='surface'
    -- The development options apply to the existing flat 24x14 coupon only.
    assert(not surface or (spec.nu==24 and spec.nv==14 and not spec.nodes and not spec.material_bands))
    assert(spec.guide_band_m==nil or (finite(spec.guide_band_m) and spec.guide_band_m>0 and surface))
    assert(not spec.ruler_print or (spec.ruler_print==true and surface))
    assert(spec.indexed_print==nil or (type(spec.indexed_print)=='boolean' and surface))
    assert(not (spec.ruler_print and spec.indexed_print))
    assert(spec.visual_gap_m==nil or (finite(spec.visual_gap_m) and spec.visual_gap_m>0 and
           spec.visual_gap_m<=0.0004 and not surface and spec.material_bands and not spec.nodes and
           spec.nu==24 and spec.nv==14))
    assert(spec.visual_gap_style==nil or (spec.visual_gap_style=='edge' and spec.visual_gap_m))
    assert(spec.visual_gap_start_row==nil or (spec.visual_gap_start_row==7 and
           spec.visual_gap_style=='edge'))
    assert(spec.bridge_print==nil or (spec.bridge_print==true and spec.visual_gap_m==nil and
           not surface and spec.material_bands and not spec.nodes and spec.nu==24 and spec.nv==14))
    assert(spec.bridge_fracture_faces==nil or (spec.bridge_fracture_faces==true and spec.bridge_print))
    tissueRemove()
    local nu,nv,h=spec.nu,spec.nv,spec.spacing
    T={nu=nu,nv=nv,x={},prev={},pinned={},springs={},shapes={},cut=0,torn=0,cutLog={},tornLog={},maxStrain=0,
       seamMode=surface,seamRefs=surface and {} or nil,seamReleases=0,
       p={iterations=spec.iterations or 12,damping=spec.damping or 0.9,tear_ratio=spec.tear_ratio or 1.5,
          cut_radius=spec.cut_radius or 3e-4,grasp_radius=spec.grasp_radius or 1.5e-3,
          visualGap=spec.visual_gap_m,visualGapStyle=spec.visual_gap_style,
          visualGapStartRow=spec.visual_gap_start_row or 0}}
    if spec.visual_gap_m then
        local L=vlen(spec.u)
        assert(L>1e-12)
        T.visualAxis=vmul(spec.u,1/L)
    end
    if spec.bridge_print then
        local n={spec.u[2]*spec.v[3]-spec.u[3]*spec.v[2],
                 spec.u[3]*spec.v[1]-spec.u[1]*spec.v[3],
                 spec.u[1]*spec.v[2]-spec.u[2]*spec.v[1]}
        local L=vlen(n)
        assert(L>1e-12)
        T.bridgeNormal=vmul(n,1/L)
        T.bridgeColors={
            {0.18,0.82,0.25},{0.27,0.72,0.43},{0.19,0.57,0.72},
            {0.15,0.36,0.88},{0.47,0.76,0.32},{0.61,0.65,0.28},
            {0.34,0.73,0.64}
        }
        T.bridgeFractureFaces=spec.bridge_fracture_faces
        T.bridgeFractureColors={{0.12,0.94,0.17},{0.09,0.21,0.97}}
    end
    local function id(i,j) return j*nu+i+1 end
    for j=0,nv-1 do for i=0,nu-1 do
        local k=id(i,j)
        -- Optional explicit node geometry (e.g. the curved prostate-proxy ROI): rest lengths
        -- below already come from T.x, so they follow the supplied surface. h stays the
        -- nominal pitch for the anchor/grasp logic; the flat path is unchanged.
        local p=spec.nodes and v3(spec.nodes[k]) or vadd(spec.origin,vadd(vmul(spec.u,(i-(nu-1)/2)*h),vmul(spec.v,j*h)))
        T.x[k]=p;T.prev[k]=v3(p)
        -- Side anchors start at row spec.anchor_from_row (default 1): a pinned corner on the free near edge
        -- concentrates the pre-tension in one constraint (it tore at 53 % strain before any cut in the first
        -- trial; the 24-node-wide task sheet still tore at row 1 and uses row 3).
        if j==nv-1 or (spec.anchor_sides and j>=(spec.anchor_from_row or 1) and (i==0 or i==nu-1)) then T.pinned[k]={fixed=v3(p)} end
    end end
    local key={}
    local function spring(a,b)
        local s=key[a..'_'..b] or key[b..'_'..a]
        if s then return s end
        T.springs[#T.springs+1]={a=a,b=b,rest=vlen(vsub(T.x[a],T.x[b]))*(1-(spec.pretension or 0)),alive=true}
        key[a..'_'..b]=#T.springs
        return #T.springs
    end
    T.tri={}
    local bands=spec.material_bands
    T.colors={spec.color or {0.86,0.42,0.48}}
    if bands then
        T.colors={}
        for i,b in ipairs(bands) do T.colors[i]=b.color end
    end
    local band=1
    for j=0,nv-2 do for i=0,nu-2 do
        if bands and j>=bands[band].end_row then band=band+1 end
        local k00,k10,k01,k11=id(i,j),id(i+1,j),id(i,j+1),id(i+1,j+1)
        local s1,s2,s3,s4,s5=spring(k00,k10),spring(k10,k11),spring(k00,k11),spring(k00,k01),spring(k01,k11)
        -- surface mode omits only the reverse (non-face) diagonal of the centre column cells.
        if not (surface and i==nu/2-1) then spring(k10,k01) end
        T.tri[#T.tri+1]={k00,k10,k11,s1,s2,s3,band}
        T.tri[#T.tri+1]={k00,k11,k01,s3,s5,s4,band}
        if surface then
            -- Register the incident central faces of each live centre crossing bond as the
            -- triangles are created (crossing = endpoint columns nu/2-1 and nu/2).
            for q=#T.tri-1,#T.tri do
                local tr=T.tri[q]
                for e=4,6 do
                    local s=tr[e]
                    local ca,cb=(T.springs[s].a-1)%nu,(T.springs[s].b-1)%nu
                    if (ca==nu/2-1 and cb==nu/2) or (ca==nu/2 and cb==nu/2-1) then
                        local refs=T.seamRefs[s] or {}
                        refs[#refs+1]=tr
                        T.seamRefs[s]=refs
                    end
                end
            end
        end
    end end
    if spec.visual_gap_m then
        T.edgeRefs={}
        for _,tr in ipairs(T.tri) do
            for q=4,6 do
                local refs=T.edgeRefs[tr[q]] or {}
                refs[#refs+1]=tr
                T.edgeRefs[tr[q]]=refs
            end
        end
    end
    if spec.bridge_print then
        T.bridgeRefs={}
        for i=8,14 do
            local a,b,c,d=id(i,8),id(i+1,8),id(i,9),id(i+1,9)
            local function existing(x,y)
                local ref=key[x..'_'..y] or key[y..'_'..x]
                assert(ref)
                return ref
            end
            T.bridgeRefs[#T.bridgeRefs+1]={existing(a,b),existing(a,d),existing(b,c)}
        end
    end
    if surface then
        local crossing=0
        for k,refs in pairs(T.seamRefs) do
            assert(#refs==1 or #refs==2)
            crossing=crossing+1
        end
        assert(crossing==27)    -- 14 transverse bonds + 13 face diagonals
    end
    if spec.ruler_print then
        -- Immutable marker texture, identical formulas to src/tissue/ruler_print.lua:
        -- 13 rows x 2 sides of colour cells addressed by triangle ordinal.
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
    end
    if spec.indexed_print then
        -- Immutable full-face material addresses; only the original live triangles render.
        for j=0,12 do
            for k=0,7 do
                T.colors[#T.colors+1]={(42+13*j)/255,(50+21*k)/255,(42+25*((3*j+5*k)%8))/255}
            end
        end
        for idx,tr in ipairs(T.tri) do
            local cell=math.floor((idx-1)/2)
            local i,j=cell%23,math.floor(cell/23)
            if i>=7 and i<=15 then
                local k=(i<=8 and 0) or (i==9 and 1) or (i==10 and 2) or
                        (i==11 and (idx%2==1 and 4 or 3)) or (i==12 and 5) or (i==13 and 6) or 7
                tr[7]=2+8*j+k
            end
        end
    end
    if spec.guide_band_m then
        local n={spec.u[2]*spec.v[3]-spec.u[3]*spec.v[2],
                 spec.u[3]*spec.v[1]-spec.u[1]*spec.v[3],
                 spec.u[1]*spec.v[2]-spec.u[2]*spec.v[1]}
        local L=vlen(n)
        assert(L>1e-12 and spec.origin)
        guideBand=spec.guide_band_m
        guideAxis=vmul(n,1/L)
        guideOrigin=v3(spec.origin)
        guideKeys={}
        for j=0,12 do for _,i in ipairs({6,7,8,9,10,13,14,15,16,17}) do
            guideKeys[#guideKeys+1]=j*nu+i+1
        end end
    else
        guideKeys,guideOrigin,guideAxis,guideBand=nil
    end
    render()
    return nu*nv
end

local function faceLive(tr)
    return T.springs[tr[4]].alive and T.springs[tr[5]].alive and T.springs[tr[6]].alive
end

-- Release each still-alive centre crossing bond that lost all of its incident central
-- faces. Counted in T.seamReleases, not in the direct cut count; invoked by tissueCut
-- and tissueStep only -- render() stays read-only.
local function releaseUnsupported()
    if not T or not T.seamRefs then return end
    for k,refs in pairs(T.seamRefs) do
        local s=T.springs[k]
        if s.alive then
            local ok=false
            for _,tr in ipairs(refs) do
                if faceLive(tr) then ok=true;break end
            end
            if not ok then
                s.alive=false
                T.seamReleases=T.seamReleases+1
            end
        end
    end
end

local function jawSegment(arm)
    -- Blade / jaw line: from the jaw pivot (jaw joint position) to the tool tip.
    return sim.getObjectPosition(arm.grip[1],sim.handle_world),sim.getObjectPosition(arm.tcp,sim.handle_world)
end

local function rotate(v,k,a)      -- Rodrigues: v rotated by a about unit axis k
    local c,s=math.cos(a),math.sin(a)
    local kxv={k[2]*v[3]-k[3]*v[2],k[3]*v[1]-k[1]*v[3],k[1]*v[2]-k[2]*v[1]}
    return vadd(vadd(vmul(v,c),vmul(kxv,s)),vmul(k,vdot(k,v)*(1-c)))
end
-- Möller-Trumbore: does segment p-q cross triangle a,b,c?
local function segTri(p,q,a,b,c)
    local d=vsub(q,p);local e1,e2=vsub(b,a),vsub(c,a)
    local h={d[2]*e2[3]-d[3]*e2[2],d[3]*e2[1]-d[1]*e2[3],d[1]*e2[2]-d[2]*e2[1]}
    local det=vdot(e1,h)
    if math.abs(det)<1e-18 then return false end
    local f=1/det;local s=vsub(p,a);local u=f*vdot(s,h)
    if u<0 or u>1 then return false end
    local qv={s[2]*e1[3]-s[3]*e1[2],s[3]*e1[1]-s[1]*e1[3],s[1]*e1[2]-s[2]*e1[1]}
    local v=f*vdot(d,qv)
    if v<0 or u+v>1 then return false end
    local t=f*vdot(e2,qv)
    return t>=0 and t<=1
end

-- Includes coplanar segments and segments parallel to, but within cut_radius of, a swept triangle.
local function nearTri(p,q,a,b,c,r2)
    if segTri(p,q,a,b,c) or segDist2(p,q,a,b)<=r2 or segDist2(p,q,b,c)<=r2 or segDist2(p,q,c,a)<=r2 then return true end
    local e1,e2=vsub(b,a),vsub(c,a)
    local aa,bb,cc=vdot(e1,e1),vdot(e1,e2),vdot(e2,e2)
    local den=aa*cc-bb*bb
    if den<=0 then return false end
    for _,x in ipairs({p,q}) do
        local d=vsub(x,a);local da,db=vdot(d,e1),vdot(d,e2)
        local u,v=(cc*da-bb*db)/den,(aa*db-bb*da)/den
        if u>=0 and v>=0 and u+v<=1 then
            local off=vsub(d,vadd(vmul(e1,u),vmul(e2,v)))
            if vdot(off,off)<=r2 then return true end
        end
    end
    return false
end

-- Both joints receive the SAME positive angle, but their world axes face opposite ways (install_instruments).
-- Sweep each blade separately: a single triangle between open tips would cut the still-open centre gap.
-- Fan triangles approximate the circular tip arc to <= 0.05 rad per sector; no centre-line cut is added.
function tissueCut(index,oldAngle,newAngle)
    local arm=arms[index]
    if not T or not arm or not arm.tcp or not arm.grip or #arm.grip<2 or not hasInstrumentCapability(index,'cut') then return 0 end
    if not finite(oldAngle) or not finite(newAngle) or newAngle>=oldAngle or oldAngle<=0 then return 0 end
    oldAngle=math.max(0,oldAngle);newAngle=math.max(0,newAngle)
    local Q=sim.getObjectPosition(arm.tcp,sim.handle_world)
    local sectors={};local steps=math.max(1,math.ceil((oldAngle-newAngle)/0.05))
    for i=1,2 do
        local P=sim.getObjectPosition(arm.grip[i],sim.handle_world)
        local m=sim.getObjectMatrix(arm.grip[i],sim.handle_world)
        local axis={m[3],m[7],m[11]};local line=vsub(Q,P)
        local prev=vadd(P,rotate(line,axis,oldAngle))
        for k=1,steps do
            local tip=vadd(P,rotate(line,axis,oldAngle+(newAngle-oldAngle)*k/steps))
            sectors[#sectors+1]={P,prev,tip};prev=tip
        end
    end
    local r2,n=T.p.cut_radius^2,0
    for _,s in ipairs(T.springs) do
        if s.alive then
            local a,b=T.x[s.a],T.x[s.b]
            for _,tr in ipairs(sectors) do
                if nearTri(a,b,tr[1],tr[2],tr[3],r2) then s.alive=false;n=n+1;break end
            end
        end
    end
    T.cut=T.cut+n
    T.cutLog[#T.cutLog+1]={arm=arm.arm,t=sim.getSimulationTime(),constraints=n,from_angle=oldAngle,to_angle=newAngle}
    releaseUnsupported()
    render()
    return n
end

local function jawsClosed(arm)
    if not arm or not arm.grip or #arm.grip<2 or not arm.tcp then return false end
    for _,h in ipairs(arm.grip) do
        local q=sim.getJointPosition(h)
        if not finite(q) or math.abs(q)>CLOSED_EPS then return false end
    end
    return true
end

function tissueGrasp(index)
    local arm=arms[index]
    if not T or not arm then return 0 end
    if not hasInstrumentCapability(index,'grasp_tissue') then
        arm.capabilityRefusals=(arm.capabilityRefusals or 0)+1
        return -1
    end
    if not jawsClosed(arm) then tissueRelease(index);return 0 end
    local P,Q=jawSegment(arm)
    local inv=sim.getMatrixInverse(sim.getObjectMatrix(arm.tcp,sim.handle_world))
    local n=0
    for k,x in ipairs(T.x) do
        if not T.pinned[k] and segDist2(x,x,P,Q)<T.p.grasp_radius^2 then
            T.pinned[k]={arm=index,rel=sim.multiplyVector(inv,x)};n=n+1
        end
    end
    return n
end

function tissueRelease(index)
    if not T then return 0 end
    local n=0
    for k,p in pairs(T.pinned) do if p.arm==index then T.pinned[k]=nil;n=n+1 end end
    return n
end

local function tissueStep()
    if not T then return end
    -- Also release after an external joint change, not only after commands through setGrip.
    for i,arm in ipairs(arms) do if not jawsClosed(arm) then tissueRelease(i) end end
    for k,p in pairs(T.pinned) do
        if p.arm then T.x[k]=sim.multiplyVector(sim.getObjectMatrix(arms[p.arm].tcp,sim.handle_world),p.rel) else T.x[k]=v3(p.fixed) end
    end
    for k=1,#T.x do
        local x=T.x[k]
        if T.pinned[k] then T.prev[k]=x else T.x[k]=vadd(x,vmul(vsub(x,T.prev[k]),T.p.damping));T.prev[k]=x end
    end
    for _=1,T.p.iterations do
        for _,s in ipairs(T.springs) do
            if s.alive then
                local a,b=T.x[s.a],T.x[s.b]
                local d=vsub(b,a);local L=vlen(d)
                local wa=T.pinned[s.a] and 0 or 1;local wb=T.pinned[s.b] and 0 or 1
                if L>1e-12 and wa+wb>0 then
                    local c=(L-s.rest)/(L*(wa+wb))
                    if wa>0 then T.x[s.a]=vadd(a,vmul(d,c)) end
                    if wb>0 then T.x[s.b]=vsub(b,vmul(d,c)) end
                end
            end
        end
        if guideKeys then
            for _,k in ipairs(guideKeys) do
                if not T.pinned[k] then
                    local p=T.x[k]
                    local d=vdot(vsub(p,guideOrigin),guideAxis)
                    local e=d-clamp(d,-guideBand,guideBand)
                    if e~=0 then T.x[k]=vsub(p,vmul(guideAxis,e)) end
                end
            end
        end
    end
    local m=0
    for _,s in ipairs(T.springs) do
        if s.alive then
            local L=vlen(vsub(T.x[s.b],T.x[s.a]))
            m=math.max(m,L/s.rest-1)
            if L>T.p.tear_ratio*s.rest then
                s.alive=false;T.torn=T.torn+1
                T.tornLog[#T.tornLog+1]={t=sim.getSimulationTime(),a=s.a-1,b=s.b-1,strain=L/s.rest-1}
            end
        end
    end
    T.maxStrain=m
    releaseUnsupported()
    render()
end

function tissueStats()
    if not T then return {exists=false} end
    local alive,grasped,strains=0,{},{}
    for _,s in ipairs(T.springs) do
        if s.alive then alive=alive+1;strains[#strains+1]=vlen(vsub(T.x[s.b],T.x[s.a]))/s.rest-1 end
    end
    table.sort(strains)
    -- Tension of the sheet = median constraint strain; the maximum sits at corners and cut ends.
    local median=#strains>0 and strains[math.floor((#strains+1)/2)] or 0
    local p90=#strains>0 and strains[math.max(1,math.floor(0.9*#strains))] or 0
    for _,p in pairs(T.pinned) do if p.arm then local a=arms[p.arm].arm;grasped[a]=(grasped[a] or 0)+1 end end
    return {exists=true,nodes=#T.x,constraints=#T.springs,alive=alive,cut=T.cut,torn=T.torn,
            seam_releases=T.seamReleases,
            max_strain=T.maxStrain,median_strain=median,p90_strain=p90,grasped=grasped,cut_events=T.cutLog,torn_events=T.tornLog}
end

-- Constraint list as flat {a, b, alive} triples (node indices from 0, alive 1/0).
function tissueConstraints()
    if not T then return {} end
    local out={}
    for _,s in ipairs(T.springs) do out[#out+1]=s.a-1;out[#out+1]=s.b-1;out[#out+1]=s.alive and 1 or 0 end
    return out
end

function tissueNodes()
    if not T then return {} end
    local out={}
    for _,x in ipairs(T.x) do out[#out+1]=x[1];out[#out+1]=x[2];out[#out+1]=x[3] end
    return out
end

-- Flat 1/0 liveness of the two real triangles of each centre-column material row, in order.
function surfaceSeamFaces()
    if not (T and T.seamMode) then return {} end
    local out={}
    for j=0,T.nv-2 do
        local base=2*(j*(T.nu-1)+T.nu/2-1)
        for k=1,2 do
            local tr=T.tri[base+k]
            out[#out+1]=(tr and faceLive(tr)) and 1 or 0
        end
    end
    return out
end

-- Orphaned centre crossing bonds released after cuts/tears (never by the renderer).
function surfaceSeamReleases()
    return (T and T.seamReleases) or 0
end

-- Actual rest length of each row's centre transverse bond (columns nu/2-1 and nu/2,
-- 0-based), per sheet row. Read-only; on non-uniform central cells this is the
-- metrology basis, not the nominal grid pitch.
function tissueTransverseRestLengths()
    local out={}
    for j=0,T.nv-1 do
        local a=j*T.nu+math.floor(T.nu/2);local b=a+1;local rest=nil
        for _,s in ipairs(T.springs) do
            if (s.a==a and s.b==b) or (s.a==b and s.b==a) then assert(not rest);rest=s.rest end
        end
        assert(rest and rest>0);out[#out+1]=rest
    end
    return out
end

local cutSetGrip=setGrip
function setGrip(index,angle)
    local arm=arms[index]
    -- Reject before the capability wrapper's moving-hinge arithmetic as well as in the base controller.
    if not finite(angle) or not arm or not arm.grip or #arm.grip==0 then return false end
    local before=sim.getJointPosition(arm.grip[1])
    local ok=cutSetGrip(index,angle)
    if ok then
        local after=sim.getJointPosition(arm.grip[1])
        if not jawsClosed(arm) then tissueRelease(index) end
        if after>before+1e-9 then
            cutArmed[index]=after>=0.15
        elseif after<before-1e-9 then
            local armed=cutArmed[index] or before>=0.15
            if armed and T then tissueCut(index,before,after) end
            cutArmed[index]=armed and after>CLOSED_EPS
        elseif after>=0.15 then
            cutArmed[index]=true
        end
    end
    return ok
end

local tissueBaseActuation=sysCall_actuation
function sysCall_actuation()
    tissueBaseActuation()
    tissueStep()
end

local tissueBaseCleanup=sysCall_cleanup
function sysCall_cleanup()
    tissueRemove()
    if tissueBaseCleanup then tissueBaseCleanup() end
end
