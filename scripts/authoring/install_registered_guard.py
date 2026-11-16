from davinci_geometry import *
import json,time,shutil,tempfile
OUT=ROOT.parent/'davinci_prostate_rcm';cfg=json.loads((OUT/'rcm_config.json').read_text());pairs=json.loads((OUT/'collision_pairs.json').read_text())['labels']
def lua(v):
 if isinstance(v,dict):return '{'+','.join('['+json.dumps(k)+']='+lua(x) for k,x in v.items())+'}'
 if isinstance(v,list):return '{'+','.join(lua(x) for x in v)+'}'
 return json.dumps(v)
guard="""
local function collisionFree(arm)
 for _,p in ipairs(arm.collisionPairs) do
  if sim.checkCollision(p[1],p[2])>0 then
   local allowed=false
   if p[3]==arm.arm..'_tool_main_respondable_visual' and p[4]=='Patient_Skin' then
    local n,segments=sim.checkCollisionEx(p[1],p[2]);allowed=n>0
    for k=1,#segments,3 do
     local dx=segments[k]-arm.rcm_world[1];local dy=segments[k+1]-arm.rcm_world[2];local dz=segments[k+2]-arm.rcm_world[3]
     if dx*dx+dy*dy+dz*dz>0.016^2 then allowed=false;break end
    end
   end
   if not allowed then arm.lastCollision=p[3]..' / '..p[4];return false end
  end
 end
 arm.lastCollision='';return true
end
"""
code=(OUT/'rcm_controller.lua').read_text()
code=code.replace('local function control(',guard+'\nlocal function control(',1)
code=code.replace('local oldInsert=sim.getJointPosition(arm.insertion)','local oldInsert=sim.getJointPosition(arm.insertion)\n    local oldTarget=arm.lastTarget;local collisionRejected=false',1)
code=code.replace('    if not converged then\n        for i,h in ipairs(arm.handles)',"    if converged and not collisionFree(arm) then converged=false;collisionRejected=true;arm.collisionBlocks=arm.collisionBlocks+1 end\n    if not converged then\n        for i,h in ipairs(arm.handles)",1)
code=code.replace('arm.failures=arm.failures+1','if not collisionRejected then arm.failures=arm.failures+1 end',1)
code=code.replace('arm.lastTarget={yaw,pitch,insertion};','arm.lastTarget=converged and {yaw,pitch,insertion} or oldTarget;')
code=code.replace('arm.handles={};arm.home={};arm.failures=0;arm.maxDrift=0;arm.drift=0',"arm.handles={};arm.home={};arm.failures=0;arm.maxDrift=0;arm.drift=0;arm.collisionBlocks=0;arm.lastCollision='';arm.collisionPairs={}\n        for _,p in ipairs(COLLISION_PAIRS) do if string.sub(p[1],1,4)==arm.arm or string.sub(p[2],1,4)==arm.arm then arm.collisionPairs[#arm.collisionPairs+1]={by[p[1]],by[p[2]],p[1],p[2]} end end")
code=code.replace('failures=arm.failures,target=', 'failures=arm.failures,collision_blocks=arm.collisionBlocks,last_collision=arm.lastCollision,target=')
embedded='CONFIG='+lua(cfg)+'\nCOLLISION_PAIRS='+lua(pairs)+'\n'+code
(OUT/'rcm_guarded_embedded.lua').write_text(embedded,encoding='utf-8')
sys.path.insert(0,r'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python\src')
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
c=RemoteAPIClient(port=23020);s=c.require('sim');c.socket.RCVTIMEO=120000
assert s.getSimulationState()==s.simulation_stopped;s.removeObjects([s.getObject('/RCMController')]);h=s.createScript(s.scripttype_simulation,embedded,0,'lua');s.setObjectAlias(h,'RCMController')
panel=(ROOT.parent/'davinci_rcm'/'rcm_panel.lua').read_text().replace('Three-arm RCM | software constraint','Abdominal RCM | foot-end docking').replace('Fixed world pivots / original joint-chain control','Registered skin ports / guarded kinematic control').replace('Yaw (-5 to +5 deg)','Yaw (-1 to +1 deg)').replace('minimum="-50" maximum="50"','minimum="-10" maximum="10"').replace('Pitch (-4 to +4 deg)','Pitch (-0.75 to +0.75 deg)').replace('minimum="-40" maximum="40"','minimum="-75" maximum="75"').replace('pitch=v/10','pitch=v/100').replace('Insertion (20 to 60 mm)','Insertion (38 to 42 mm)').replace('minimum="20" maximum="60"','minimum="38" maximum="42"').replace('Setup rotations coordinated; base and lifts fixed.','Exploratory CAD joint ranges; no clinical validation.').replace("a.failures) end","a.failures+a.collision_blocks) end")
(OUT/'rcm_panel.lua').write_text(panel,encoding='utf-8');s.removeObjects([s.getObject('/RCMControlPanel')]);h=s.createScript(s.scripttype_customization,panel,0,'lua');s.setObjectAlias(h,'RCMControlPanel')
print('Installed bounded controls and mesh collision rollback guard.',flush=True)
