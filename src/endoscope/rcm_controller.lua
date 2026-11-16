-- Software RCM controller for the endoscope task scene (three PSMs + ECM).
-- CONFIG and COLLISION_PAIRS are prepended by scripts/build/build_endoscope_scene.py from
-- config/endoscope/rcm_config.json and config/endoscope/collision_pairs.json.
-- All pivots are fixed world points. Joint ranges are exploratory simulation bounds.
--
-- Per-arm CONFIG fields:
--   arm, joints{name,type,limits}, insertion_joint, shaft_frame (z = shaft axis), projection,
--   rcm_world, axis_home, transverse_x, transverse_y, insertion_home, rcm_distance_home,
--   limits{yaw_deg,pitch_deg,insertion_min,insertion_max}, tool_shape, port_radius_m,
--   constrain_roll (also hold the shaft-frame x axis, used for the camera), auto_demo,
--   demo{yaw_deg,pitch_deg,insertion_amp_m} (auto-demo amplitudes), and for PSMs grip_joints, tcp_frame.
sim=require('sim')
local function add(a,b) return {a[1]+b[1],a[2]+b[2],a[3]+b[3]} end
local function sub(a,b) return {a[1]-b[1],a[2]-b[2],a[3]-b[3]} end
local function mul(a,k) return {a[1]*k,a[2]*k,a[3]*k} end
local function dot(a,b) return a[1]*b[1]+a[2]*b[2]+a[3]*b[3] end
local function norm(a) return math.sqrt(dot(a,a)) end
local function cross(a,b) return {a[2]*b[3]-a[3]*b[2],a[3]*b[1]-a[1]*b[3],a[1]*b[2]-a[2]*b[1]} end
local function frame(h) local m=sim.getObjectMatrix(h,sim.handle_world);return {m[4],m[8],m[12]},{m[3],m[7],m[11]},{m[1],m[5],m[9]} end
local function solve(A,b)
    local n=#b
    for k=1,n do
        local pivot=k
        for i=k+1,n do if math.abs(A[i][k])>math.abs(A[pivot][k]) then pivot=i end end
        if math.abs(A[pivot][k])<1e-14 then return nil end
        A[k],A[pivot]=A[pivot],A[k];b[k],b[pivot]=b[pivot],b[k]
        local v=A[k][k]
        for j=k,n do A[k][j]=A[k][j]/v end;b[k]=b[k]/v
        for i=1,n do if i~=k then
            local f=A[i][k]
            for j=k,n do A[i][j]=A[i][j]-f*A[k][j] end;b[i]=b[i]-f*b[k]
        end end
    end
    return b
end
-- Rotate v by the minimal rotation that takes unit vector a onto unit vector b.
local function rotateMinimal(a,b,v)
    local k=cross(a,b);local s=norm(k);local c=dot(a,b)
    if s<1e-15 then return {v[1],v[2],v[3]} end
    k=mul(k,1/s)
    return add(add(mul(v,c),mul(cross(k,v),s)),mul(k,dot(k,v)*(1-c)))
end

local function collisionFree(arm)
 for _,p in ipairs(arm.collisionPairs) do
  if sim.checkCollision(p[1],p[2])>0 then
   local allowed=false
   -- The only intended contact: this arm's own shaft crossing the skin near its own port.
   if p[3]==arm.tool_shape and p[4]=='Patient_Skin' then
    local n,segments=sim.checkCollisionEx(p[1],p[2]);allowed=n>0
    for k=1,#segments,3 do
     local dx=segments[k]-arm.rcm_world[1];local dy=segments[k+1]-arm.rcm_world[2];local dz=segments[k+2]-arm.rcm_world[3]
     if dx*dx+dy*dy+dz*dz>arm.port_radius_m^2 then allowed=false;break end
    end
   end
   if not allowed then arm.lastCollision=p[3]..' / '..p[4];return false end
  end
 end
 arm.lastCollision='';return true
end

local function control(arm,yaw,pitch,insertion)
    local d=add(arm.axis_home,add(mul(arm.transverse_x,math.tan(math.rad(yaw))),mul(arm.transverse_y,math.tan(math.rad(pitch)))))
    d=mul(d,1/norm(d))
    local target=sub(arm.rcm_world,mul(d,arm.rcm_distance_home-(insertion-arm.insertion_home)))
    local xd=arm.constrain_roll and rotateMinimal(arm.axis_home,d,arm.transverse_x) or nil
    local old={};for i,h in ipairs(arm.handles) do old[i]=sim.getJointPosition(h) end
    local oldInsert=sim.getJointPosition(arm.insertion)
    local oldTarget=arm.lastTarget;local collisionRejected=false
    sim.setJointPosition(arm.insertion,insertion)
    local converged=false
    for iteration=1,18 do
        local p,a,x=frame(arm.shaft)
        local ep=sub(target,p);local ea=sub(d,a);local ex=xd and sub(xd,x) or nil
        if norm(ep)<2e-8 and norm(ea)<1e-7 and (not ex or norm(ex)<1e-7) then converged=true;break end
        local e={ep[1]/.3,ep[2]/.3,ep[3]/.3,ea[1],ea[2],ea[3]}
        if ex then e[7]=ex[1];e[8]=ex[2];e[9]=ex[3] end
        local cols={};local q={}
        for i,h in ipairs(arm.handles) do
            local jp,ja=frame(h);local dp=mul(cross(ja,sub(p,jp)),1/.3);local da=cross(ja,a)
            cols[i]={dp[1],dp[2],dp[3],da[1],da[2],da[3]}
            if ex then local dx=cross(ja,x);cols[i][7]=dx[1];cols[i][8]=dx[2];cols[i][9]=dx[3] end
            q[i]=sim.getJointPosition(h)
        end
        local A,b={},{}
        for i=1,#cols do
            A[i]={};b[i]=-1e-9*(q[i]-arm.home[i])
            for k=1,#e do b[i]=b[i]+cols[i][k]*e[k] end
            for j=1,#cols do
                local v=i==j and 1e-9 or 0
                for k=1,#e do v=v+cols[i][k]*cols[j][k] end;A[i][j]=v
            end
        end
        local dq=solve(A,b);if not dq then break end
        local scale=1
        for _,v in ipairs(dq) do scale=math.min(scale,.03/math.max(math.abs(v),1e-12)) end
        for i,h in ipairs(arm.handles) do
            local lim=arm.joints[i].limits;local value=math.max(lim[1],math.min(lim[2],q[i]+scale*dq[i]));sim.setJointPosition(h,value)
        end
    end
    if not converged then
        local p,a,x=frame(arm.shaft)
        converged=norm(sub(target,p))<1e-7 and norm(sub(d,a))<1e-6 and (not xd or norm(sub(xd,x))<1e-6)
    end
    if converged and not collisionFree(arm) then converged=false;collisionRejected=true;arm.collisionBlocks=arm.collisionBlocks+1 end
    if not converged then
        for i,h in ipairs(arm.handles) do sim.setJointPosition(h,old[i]) end
        sim.setJointPosition(arm.insertion,oldInsert);if not collisionRejected then arm.failures=arm.failures+1 end
    end
    local p,a=frame(arm.shaft);local delta=sub(arm.rcm_world,p);local distance=dot(delta,a);local projection=add(p,mul(a,distance));local drift=norm(sub(projection,arm.rcm_world))
    arm.maxDrift=math.max(arm.maxDrift,drift);arm.drift=drift;arm.distance=distance
    arm.lastTarget=converged and {yaw,pitch,insertion} or oldTarget;arm.targetError=norm(sub(target,p));arm.converged=converged
    sim.setObjectPosition(arm.projection,sim.handle_world,projection)
    return converged
end
function sysCall_init()
    local by={}
    for _,h in ipairs(sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)) do by[sim.getObjectAlias(h)]=h end
    local function need(name) local h=by[name];if not h then error('RCM controller: missing scene object '..name) end;return h end
    arms=CONFIG;auto=true;frozen=false
    for _,arm in ipairs(arms) do
        arm.handles={};arm.home={};arm.failures=0;arm.maxDrift=0;arm.drift=0;arm.collisionBlocks=0;arm.lastCollision='';arm.collisionPairs={}
        local prefix=arm.arm..'_'
        for _,p in ipairs(COLLISION_PAIRS) do
            if string.sub(p[1],1,#prefix)==prefix or string.sub(p[2],1,#prefix)==prefix then arm.collisionPairs[#arm.collisionPairs+1]={need(p[1]),need(p[2]),p[1],p[2]} end
        end
        for i,j in ipairs(arm.joints) do arm.handles[i]=need(j.name);arm.home[i]=sim.getJointPosition(arm.handles[i]) end
        arm.insertion=need(arm.insertion_joint);arm.shaft=need(arm.shaft_frame);arm.projection=need(arm.projection)
        arm.lastTarget={0,0,arm.insertion_home}
        arm.grip={};arm.gripAngle=0
        for i,n in ipairs(arm.grip_joints or {}) do arm.grip[i]=need(n) end
        if arm.tcp_frame then arm.tcp=need(arm.tcp_frame) end
    end
    grasped={}   -- object handle -> {parent, local matrix, arm} at the first grasp, restored at cleanup
end
-- Jaw opening (rad, both jaw joints); refused and rolled back if the jaws would hit anything.
function setGrip(index,angle)
    local arm=arms[index]
    if not arm or #arm.grip==0 then return false end
    local old={};for i,h in ipairs(arm.grip) do old[i]=sim.getJointPosition(h);sim.setJointPosition(h,angle) end
    if not collisionFree(arm) then
        for i,h in ipairs(arm.grip) do sim.setJointPosition(h,old[i]) end
        arm.collisionBlocks=arm.collisionBlocks+1
        return false
    end
    arm.gripAngle=angle
    return true
end
-- Kinematic grasp: the object follows the instrument tip frame until released (no contact physics).
function grasp(index,alias)
    local arm=arms[index]
    if not arm or not arm.tcp or arm.held then return false end
    local h=sim.getObject(string.sub(alias,1,1)=='/' and alias or '/'..alias)
    if not grasped[h] then grasped[h]={parent=sim.getObjectParent(h),matrix=sim.getObjectMatrix(h,sim.getObjectParent(h))} end
    sim.setObjectParent(h,arm.tcp,true);arm.held=h
    return true
end
function release(index)
    local arm=arms[index]
    if not arm or not arm.held then return false end
    sim.setObjectParent(arm.held,-1,true);arm.held=nil
    return true
end
function setRCMTarget(index,yaw,pitch,insertion)
    local arm=arms[index]
    if not arm then return false end
    local L=arm.limits
    if math.abs(yaw)>L.yaw_deg or math.abs(pitch)>L.pitch_deg or insertion<L.insertion_min or insertion>L.insertion_max then return false end
    auto=false;frozen=false
    local start=arm.lastTarget
    local saved={};for i,h in ipairs(arm.handles) do saved[i]=sim.getJointPosition(h) end
    local savedInsert=sim.getJointPosition(arm.insertion)
    local steps=math.max(1,math.ceil(math.abs(yaw-start[1])/.5),math.ceil(math.abs(pitch-start[2])/.5),math.ceil(math.abs(insertion-start[3])/.005))
    for k=1,steps do
        local f=k/steps
        if not control(arm,start[1]+f*(yaw-start[1]),start[2]+f*(pitch-start[2]),start[3]+f*(insertion-start[3])) then
            for i,h in ipairs(arm.handles) do sim.setJointPosition(h,saved[i]) end
            sim.setJointPosition(arm.insertion,savedInsert);arm.lastTarget=start
            return false
        end
    end
    return true
end
function setAuto(enabled) auto=enabled;frozen=false end
function hold() auto=false;frozen=true end
function homeRCM()
    auto=false;frozen=false
    for _,arm in ipairs(arms) do control(arm,0,0,arm.insertion_home) end
end
function getRCMStats()
    local result={}
    for _,arm in ipairs(arms) do result[#result+1]={arm=arm.arm,drift_m=arm.drift,max_drift_m=arm.maxDrift,failures=arm.failures,collision_blocks=arm.collisionBlocks,last_collision=arm.lastCollision,target=arm.lastTarget,converged=arm.converged,target_error_m=arm.targetError,shaft_distance_m=arm.distance,grip=arm.gripAngle,holding=arm.held~=nil} end
    return result
end
function sysCall_actuation()
    if not auto or frozen then return end
    local t=sim.getSimulationTime()%24;local phase=math.floor(t/8);local u=(t%8)/8;local wave=math.sin(2*math.pi*u)
    for _,arm in ipairs(arms) do
        if arm.auto_demo then
            -- Demo amplitudes are separate from the (possibly wider) manual limits.
            local D=arm.demo;local mid=arm.insertion_home;local amp=D.insertion_amp_m
            local yaw=phase~=1 and D.yaw_deg*wave or 0
            local pitch=phase~=1 and D.pitch_deg*math.sin(4*math.pi*u) or 0
            local insertion=mid+(phase~=0 and amp*wave or 0)
            control(arm,yaw,pitch,insertion)
        end
    end
end
function sysCall_cleanup()
    for h,s in pairs(grasped or {}) do sim.setObjectParent(h,s.parent,true);sim.setObjectMatrix(h,s.parent,s.matrix) end
    for _,arm in ipairs(arms or {}) do
        for _,h in ipairs(arm.grip or {}) do sim.setJointPosition(h,0) end
        for i,h in ipairs(arm.handles) do sim.setJointPosition(h,arm.home[i]) end
        sim.setJointPosition(arm.insertion,arm.insertion_home)
        sim.setObjectPosition(arm.projection,sim.handle_world,arm.rcm_world)
    end
end
