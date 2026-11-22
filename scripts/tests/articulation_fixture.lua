-- Extends lua_safety_fixture.lua with three independent distal joints and a moving TCP.
-- It exercises interface isolation, not the actual migrated mesh geometry or CoppeliaSim FK.
stub.writeLog={};stub.collisionChecks=0
for i=1,3 do
    local name='a'..i
    local cfg={version=1,roll_joint=name..'_roll',pitch_joint=name..'_pitch',yaw_joint=name..'_yaw',
        roll_sign=1,pitch_sign=1,yaw_sign=1,limits={roll={-4.5,4.5},pitch={-1.4,1.4},yaw={-1.4,1.4}}}
    INSTRUMENTS[name].articulation=cfg
    for _,axis in ipairs({'roll','pitch','yaw'}) do
        local h=cfg[axis..'_joint']
        stub.objects[#stub.objects+1]=h;stub.q[h]=0
        local lim=cfg.limits[axis];stub.intervals[h]={lim[1],lim[2]-lim[1]}
    end
end
local baseSet=sim.setJointPosition
function sim.setJointPosition(h,q)
    stub.writeLog[#stub.writeLog+1]={h,q}
    if stub.throwJoint==h then stub.throwJoint=nil;error('injected setter failure') end
    if stub.readbackJoint==h then q=q+0.001;stub.readbackJoint=nil end
    return baseSet(h,q)
end
local baseCollision=sim.checkCollision
function sim.checkCollision(a,b)
    stub.collisionChecks=stub.collisionChecks+1
    if stub.blockWrist and stub.blockWrist(a,b) then return 1 end
    return baseCollision(a,b)
end
local basePosition=sim.getObjectPosition
function sim.getObjectPosition(h,rel)
    local arm=h:match('^(a%d)_tcp$')
    if arm then return {0.01,0.01*math.sin(stub.q[arm..'_yaw']),0.01*math.sin(stub.q[arm..'_pitch'])} end
    return basePosition(h,rel)
end