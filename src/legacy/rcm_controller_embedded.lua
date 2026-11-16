CONFIG={{["arm"]="psm1",["joints"]={{["name"]="psm1_setup1_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm1_setup2_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm1_setup3_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm1_setup4_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm1_mount_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm1_shoulder_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm1_elbow_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm1_carrier_joint",["type"]="revolute",["limits"]={-0.5,0.5}}},["rcm_world"]={-0.6437074385666491,-1.2185818026777295,0.5981067919749105},["axis_home"]={0.42136026751783556,-0.7541245340990977,-0.5037377413169645},["transverse_x"]={0.4163538680042432,-0.3325977120976049,0.8461845061836951},["transverse_y"]={-0.8056705167468916,-0.5662816869520483,0.17383920578379028},["shaft_home"]={-0.7701155188219998,-0.9923444424480001,0.7492281143699998},["insertion_home"]=0.04,["rcm_distance_home"]=0.3},{["arm"]="psm2",["joints"]={{["name"]="psm2_setup1_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm2_setup2_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm2_setup3_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm2_mount_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm2_shoulder_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm2_elbow_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm2_carrier_joint",["type"]="revolute",["limits"]={-0.5,0.5}}},["rcm_world"]={-0.34828477477969216,-1.2302552918026117,0.8297169122066101},["axis_home"]={0.43631576052769294,-0.5380302358487067,-0.7212156559779649},["transverse_x"]={0.5144515396832016,-0.5084399205597605,0.6905276681630997},["transverse_y"]={-0.7382195949937419,-0.6723186093614911,0.05495013260681152},["shaft_home"]={-0.47917950293800005,-1.0688462210479996,1.0460816089999996},["insertion_home"]=0.04,["rcm_distance_home"]=0.3},{["arm"]="psm3",["joints"]={{["name"]="psm3_setup1_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm3_setup2_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm3_setup3_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm3_mount_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm3_shoulder_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm3_elbow_joint",["type"]="revolute",["limits"]={-0.5,0.5}},{["name"]="psm3_carrier_joint",["type"]="revolute",["limits"]={-0.5,0.5}}},["rcm_world"]={0.21833967652800362,-1.2305584932107996,0.8256947791576494},["axis_home"]={-0.3341392075933212,-0.541800556965999,-0.7712348192478349},["transverse_x"]={0.8291801635640242,-0.5580200119186708,0.03277075907305997},["transverse_y"]={-0.4481196785467532,-0.6285426181012863,0.6357066390478668},["shaft_home"]={0.318581438806,-1.068018326121,1.0570652249319998},["insertion_home"]=0.04,["rcm_distance_home"]=0.3}}
-- CONFIG is prepended by the scene builder. All pivots are fixed world points.
sim=require('sim')
local function add(a,b) return {a[1]+b[1],a[2]+b[2],a[3]+b[3]} end
local function sub(a,b) return {a[1]-b[1],a[2]-b[2],a[3]-b[3]} end
local function mul(a,k) return {a[1]*k,a[2]*k,a[3]*k} end
local function dot(a,b) return a[1]*b[1]+a[2]*b[2]+a[3]*b[3] end
local function norm(a) return math.sqrt(dot(a,a)) end
local function cross(a,b) return {a[2]*b[3]-a[3]*b[2],a[3]*b[1]-a[1]*b[3],a[1]*b[2]-a[2]*b[1]} end
local function pose(h) local m=sim.getObjectMatrix(h,sim.handle_world);return {m[4],m[8],m[12]},{m[3],m[7],m[11]} end
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
local function control(arm,yaw,pitch,insertion)
    local d=add(arm.axis_home,add(mul(arm.transverse_x,math.tan(math.rad(yaw))),mul(arm.transverse_y,math.tan(math.rad(pitch)))))
    d=mul(d,1/norm(d))
    local target=sub(arm.rcm_world,mul(d,arm.rcm_distance_home-(insertion-arm.insertion_home)))
    local old={};for i,h in ipairs(arm.handles) do old[i]=sim.getJointPosition(h) end
    local oldInsert=sim.getJointPosition(arm.insertion)
    sim.setJointPosition(arm.insertion,insertion)
    local converged=false
    for iteration=1,18 do
        local p,a=pose(arm.shaft)
        local ep=sub(target,p);local ea=sub(d,a)
        if norm(ep)<2e-8 and norm(ea)<1e-7 then converged=true;break end
        local e={ep[1]/.3,ep[2]/.3,ep[3]/.3,ea[1],ea[2],ea[3]}
        local cols={};local q={}
        for i,h in ipairs(arm.handles) do
            local jp,ja=pose(h);local dp=mul(cross(ja,sub(p,jp)),1/.3);local da=cross(ja,a)
            cols[i]={dp[1],dp[2],dp[3],da[1],da[2],da[3]};q[i]=sim.getJointPosition(h)
        end
        local A,b={},{}
        for i=1,#cols do
            A[i]={};b[i]=-1e-9*q[i]
            for k=1,6 do b[i]=b[i]+cols[i][k]*e[k] end
            for j=1,#cols do
                local v=i==j and 1e-9 or 0
                for k=1,6 do v=v+cols[i][k]*cols[j][k] end;A[i][j]=v
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
        local p,a=pose(arm.shaft)
        converged=norm(sub(target,p))<1e-7 and norm(sub(d,a))<1e-6
    end
    if not converged then
        for i,h in ipairs(arm.handles) do sim.setJointPosition(h,old[i]) end
        sim.setJointPosition(arm.insertion,oldInsert);arm.failures=arm.failures+1
    end
    local p,a=pose(arm.shaft);local delta=sub(arm.rcm_world,p);local distance=dot(delta,a);local projection=add(p,mul(a,distance));local drift=norm(sub(projection,arm.rcm_world))
    arm.maxDrift=math.max(arm.maxDrift,drift);arm.drift=drift;arm.distance=distance
    arm.lastTarget={yaw,pitch,insertion};arm.targetError=norm(sub(target,p));arm.converged=converged
    sim.setObjectPosition(arm.projection,sim.handle_world,projection)
    return converged
end
function sysCall_init()
    local by={}
    for _,h in ipairs(sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)) do by[sim.getObjectAlias(h)]=h end
    arms=CONFIG;auto=true;frozen=false
    for _,arm in ipairs(arms) do
        arm.handles={};arm.home={};arm.failures=0;arm.maxDrift=0;arm.drift=0
        for i,j in ipairs(arm.joints) do arm.handles[i]=by[j.name];arm.home[i]=sim.getJointPosition(by[j.name]) end
        arm.insertion=by[arm.arm..'_joint7'];arm.shaft=by[arm.arm..'_joint8'];arm.projection=by[arm.arm..'_RCM_projection'];arm.lastTarget={0,0,.04}
    end
end
function setRCMTarget(index,yaw,pitch,insertion)
    if index<1 or index>3 or math.abs(yaw)>5 or math.abs(pitch)>4 or insertion<.02 or insertion>.06 then return false end
    auto=false;frozen=false
    local arm=arms[index];local start=arm.lastTarget
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
    for _,arm in ipairs(arms) do control(arm,0,0,.04) end
end
function getRCMStats()
    local result={}
    for _,arm in ipairs(arms) do result[#result+1]={arm=arm.arm,drift_m=arm.drift,max_drift_m=arm.maxDrift,failures=arm.failures,target=arm.lastTarget,converged=arm.converged,target_error_m=arm.targetError,shaft_distance_m=arm.distance} end
    return result
end
function sysCall_actuation()
    if not auto or frozen then return end
    local t=sim.getSimulationTime()%24;local phase=math.floor(t/8);local u=(t%8)/8;local wave=math.sin(2*math.pi*u)
    local yaw=phase~=1 and 5*wave or 0
    local pitch=phase~=1 and 4*math.sin(4*math.pi*u) or 0
    local insertion=.04+(phase~=0 and .02*wave or 0)
    for _,arm in ipairs(arms) do control(arm,yaw,pitch,insertion) end
end
function sysCall_cleanup()
    for _,arm in ipairs(arms or {}) do
        for i,h in ipairs(arm.handles) do sim.setJointPosition(h,arm.home[i]) end
        sim.setJointPosition(arm.insertion,.04)
        sim.setObjectPosition(arm.projection,sim.handle_world,arm.rcm_world)
    end
end
