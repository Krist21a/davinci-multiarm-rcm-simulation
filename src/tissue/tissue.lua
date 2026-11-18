-- Cuttable, graspable tissue sheet, appended after src/instruments/capability_guard.lua in
-- scenes/davinci_si_instruments.ttt. Kinematic model only: a grid of nodes joined by distance constraints
-- (position-based dynamics), pre-tensioned between pinned anchors. No contact with the instruments, no mass,
-- no fluid, no thermal or electrosurgical effect.
--   tissueCreate(spec)   spec={origin, u, v, nu, nv, spacing, pretension, anchor_sides, iterations, damping,
--                        tear_ratio, cut_radius, grasp_radius, color}; node (i,j) = origin+(i-(nu-1)/2)*h*u+j*h*v;
--                        row j=nv-1 (and columns 0, nu-1 from row 1 when anchor_sides) are anchored.
--   tissueGrasp(index)   needs 'grasp_tissue': pins nodes within grasp_radius of the closed jaws to the tool tip.
--   tissueRelease(index)
--   closing scissors     actual opening >= 0.15 rad arms one closing cycle. Every accepted smaller angle cuts
--                        ONLY the two blade sweeps from the old angle to the new angle, including small steps
--                        below 0.15. Closure ends the cycle; reopening below 0.15 cancels it.
--   tissueStats(), tissueNodes(), tissueRemove()
-- A constraint stretched beyond tear_ratio * rest length breaks and counts as torn (tissue damage).
local T=nil
local cutArmed={}
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
    local verts,idx={},{}
    for _,x in ipairs(T.x) do verts[#verts+1]=x[1];verts[#verts+1]=x[2];verts[#verts+1]=x[3] end
    for _,tr in ipairs(T.tri) do
        if T.springs[tr[4]].alive and T.springs[tr[5]].alive and T.springs[tr[6]].alive then
            idx[#idx+1]=tr[1]-1;idx[#idx+1]=tr[2]-1;idx[#idx+1]=tr[3]-1
        end
    end
    if T.shape then sim.removeObjects({T.shape});T.shape=nil end
    if #idx==0 then return end
    T.shape=sim.createMeshShape(0,0,verts,idx)
    sim.setObjectAlias(T.shape,'TissueSheet')
    sim.setShapeColor(T.shape,nil,sim.colorcomponent_ambient_diffuse,{0,0,0})
    sim.setShapeColor(T.shape,nil,sim.colorcomponent_specular,{0,0,0})
    sim.setShapeColor(T.shape,nil,sim.colorcomponent_emission,T.color)
    sim.setObjectInt32Param(T.shape,sim.shapeintparam_static,1)
    sim.setObjectInt32Param(T.shape,sim.shapeintparam_respondable,0)
end

function tissueRemove()
    if T and T.shape then sim.removeObjects({T.shape}) end
    T=nil;cutArmed={}
end

function tissueCreate(spec)
    tissueRemove()
    local nu,nv,h=spec.nu,spec.nv,spec.spacing
    T={nu=nu,nv=nv,x={},prev={},pinned={},springs={},cut=0,torn=0,cutLog={},tornLog={},maxStrain=0,
       p={iterations=spec.iterations or 12,damping=spec.damping or 0.9,tear_ratio=spec.tear_ratio or 1.5,
          cut_radius=spec.cut_radius or 3e-4,grasp_radius=spec.grasp_radius or 1.5e-3}}
    local function id(i,j) return j*nu+i+1 end
    for j=0,nv-1 do for i=0,nu-1 do
        local k=id(i,j)
        local p=vadd(spec.origin,vadd(vmul(spec.u,(i-(nu-1)/2)*h),vmul(spec.v,j*h)))
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
    for j=0,nv-2 do for i=0,nu-2 do
        local k00,k10,k01,k11=id(i,j),id(i+1,j),id(i,j+1),id(i+1,j+1)
        local s1,s2,s3,s4,s5=spring(k00,k10),spring(k10,k11),spring(k00,k11),spring(k00,k01),spring(k01,k11)
        spring(k10,k01)
        T.tri[#T.tri+1]={k00,k10,k11,s1,s2,s3}
        T.tri[#T.tri+1]={k00,k11,k01,s3,s5,s4}
    end end
    T.color=spec.color or {0.86,0.42,0.48}
    render()
    return nu*nv
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
