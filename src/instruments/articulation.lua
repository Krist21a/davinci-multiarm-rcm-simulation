-- Optional distal EndoWrist controls, appended after controller/capability, before tissue.
-- This requires migrated scene geometry, not just a Lua refresh. INSTRUMENTS[arm].articulation =
-- {roll_joint, pitch_joint, yaw_joint, roll_sign=1, pitch_sign=1, yaw_sign=1,
--  limits={roll={lo,hi},pitch={lo,hi},yaw={lo,hi}}}. All angles are radians, relative to scene joint zeros.
-- Positive logical angles map to sign * physical joint position. The independent yaw parent carries
-- yaw housing, jaws/hinge slide and TCP. Grip remains two equal half-opening joint values; no extra DOF.
-- setRCMTarget yaw/pitch are shaft-direction commands in degrees, NOT these wrist articulation angles.
local wrists={}
local wristAxes={'roll','pitch','yaw'}
local WRIST_STEP=0.02  -- sampled collision checks, not a continuous-collision guarantee
local function wristFinite(x) return type(x)=='number' and x==x and math.abs(x)<math.huge end

local function wristSnapshot(w)
    local q={}
    for i,h in ipairs(w.handles) do q[i]=sim.getJointPosition(h) end
    return q
end

local function wristRestore(w,q)
    local ok=true
    for i,h in ipairs(w.handles) do
        local restored=pcall(sim.setJointPosition,h,q[i]);ok=restored and ok
    end
    if not ok then w.available=false;w.reason='joint rollback failed' end
    return ok
end

local wristBaseInit=sysCall_init
function sysCall_init()
    wristBaseInit()
    wrists={}
    local by={}
    for _,h in ipairs(sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)) do by[sim.getObjectAlias(h)]=h end
    for index,arm in ipairs(arms) do
        local entry=INSTRUMENTS and INSTRUMENTS[arm.arm]
        local cfg=entry and entry.articulation
        local w={available=false,reason='articulation metadata absent',handles={},signs={},limits={}}
        wrists[index]=w
        if type(cfg)=='table' then
            w.available=true;w.reason=''
            local seen={}
            for i,axis in ipairs(wristAxes) do
                local h=by[cfg[axis..'_joint']];local sign=cfg[axis..'_sign'] or 1
                local lim=cfg.limits and cfg.limits[axis]
                local forbidden=h==arm.insertion
                for _,other in ipairs(arm.grip or {}) do forbidden=forbidden or h==other end
                for _,other in ipairs(arm.handles or {}) do forbidden=forbidden or h==other end
                if not h or seen[h] or forbidden or (sign~=1 and sign~=-1) or type(lim)~='table' or
                   not wristFinite(lim[1]) or not wristFinite(lim[2]) or lim[1]>lim[2] then
                    w.available=false;w.reason='invalid articulation metadata or missing joint';break
                end
                seen[h]=true;w.handles[i]=h;w.signs[i]=sign;w.limits[i]={lim[1],lim[2]}
            end
            if w.available then
                local ok,q=pcall(wristSnapshot,w)
                if ok then
                    for _,v in ipairs(q) do if not wristFinite(v) then ok=false end end
                end
                if ok then w.home=q else w.available=false;w.reason='unreadable articulation joints' end
            end
        end
    end
end

function getWristState(index)
    local w=wrists[index]
    if not w or not w.available then return {available=false,reason=w and w.reason or 'unknown arm'} end
    local out={available=true,units='rad',limits={},last_error=w.lastError or ''}
    for i,axis in ipairs(wristAxes) do
        out[axis]=w.signs[i]*sim.getJointPosition(w.handles[i])
        out.limits[axis]={w.limits[i][1],w.limits[i][2]}
    end
    local arm=arms[index]
    if arm.grip and #arm.grip>=2 then
        local a,b=sim.getJointPosition(arm.grip[1]),sim.getJointPosition(arm.grip[2])
        out.grip=(a+b)/2;out.jaw=a+b
    end
    return out
end

function setWrist(index,roll,pitch,yaw)
    local w=wrists[index]
    if not w or not w.available then return false end
    local target={roll,pitch,yaw}
    -- Check every logical bound and installed physical interval before touching ANY joint.
    for i=1,3 do
        local value=target[i];local lim=w.limits[i]
        if not wristFinite(value) or value<lim[1] or value>lim[2] then w.lastError='wrist target outside limits';return false end
        local cyclic,range=sim.getJointInterval(w.handles[i]);local q=value*w.signs[i]
        if not cyclic and (type(range)~='table' or not wristFinite(range[1]) or not wristFinite(range[2]) or
            range[2]<0 or q<range[1] or q>range[1]+range[2]) then w.lastError='installed joint limit';return false end
        target[i]=q
    end
    local old=wristSnapshot(w);local steps=1
    for i=1,3 do
        if not wristFinite(old[i]) then w.lastError='invalid joint readback';return false end
        steps=math.max(steps,math.ceil(math.abs(target[i]-old[i])/WRIST_STEP))
    end
    hold()  -- an accepted manual command or failed motion never resumes the shaft auto-demo
    local ran,accepted=pcall(function()
        for k=1,steps do
            for i,h in ipairs(w.handles) do sim.setJointPosition(h,old[i]+(target[i]-old[i])*k/steps) end
            for i,h in ipairs(w.handles) do
                local q=sim.getJointPosition(h);local wanted=old[i]+(target[i]-old[i])*k/steps
                if not wristFinite(q) or math.abs(q-wanted)>1e-7 then w.lastError='joint readback mismatch';return false end
            end
            if not checkArmCollision(index) then
                arms[index].collisionBlocks=arms[index].collisionBlocks+1
                w.lastError='wrist collision';return false
            end
        end
        return true
    end)
    if not ran or not accepted then
        if not ran then w.lastError='wrist motion error: '..tostring(accepted) end
        wristRestore(w,old)
        return false
    end
    w.lastError=''
    return true
end

local wristBaseHome=homeRCM
function homeRCM()
    -- Home wrists first at the current shaft poses, then home the shafts. This ordering can reject a
    -- feasible alternate path; rejection is safer than bypassing the same sampled collision gate.
    hold()
    local saved={}
    for i,w in pairs(wrists) do if w.available then saved[i]=wristSnapshot(w) end end
    local function rollback()
        for i,q in pairs(saved) do wristRestore(wrists[i],q) end
        hold()
    end
    for i=1,#arms do
        local w=wrists[i]
        if w and w.available and not setWrist(i,w.signs[1]*w.home[1],w.signs[2]*w.home[2],w.signs[3]*w.home[3]) then
            rollback();return false
        end
    end
    if not wristBaseHome() then rollback();return false end
    hold()
    return true
end

local wristBaseCleanup=sysCall_cleanup
function sysCall_cleanup()
    if wristBaseCleanup then wristBaseCleanup() end
    for _,w in pairs(wrists) do if w.home then wristRestore(w,w.home) end end
end