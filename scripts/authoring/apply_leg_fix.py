from davinci_geometry import *
import json,time,math,tempfile,shutil
from PIL import Image
OUT=ROOT.parent/'davinci_prostate_rcm'
sys.path.insert(0,r'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python\src')
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
c=RemoteAPIClient(port=23020);s=c.require('sim');c.socket.RCVTIMEO=120000
if s.getSimulationState()!=s.simulation_stopped:s.stopSimulation();time.sleep(.5)
stage=Path(tempfile.gettempdir())/'leg_fix_source.ttt';shutil.copy2(OUT/'davinci_foot_docking_rcm.ttt',stage);s.loadScene(str(stage));time.sleep(.4)
data=np.load(OUT/'corrected_skin.npz');s.removeObjects([s.getObject('/Patient_Skin')]);h=s.createMeshShape(0,math.radians(30),data['vertices'].flatten().tolist(),data['faces'].flatten().tolist());s.setObjectAlias(h,'Patient_Skin');s.setObjectParent(h,s.getObject('/BodyParts3D_Male'),False);s.setShapeColor(h,None,s.colorcomponent_ambient_diffuse,[.72,.50,.38]);s.setShapeColor(h,None,s.colorcomponent_transparency,[1]);s.setObjectInt32Param(h,s.shapeintparam_static,1);s.setObjectInt32Param(h,s.shapeintparam_respondable,0)
s.removeObjects([s.getObject('/PatientViewPanel')]);panel=s.createScript(s.scripttype_customization,(OUT/'patient_panel.lua').read_text(encoding='utf-8'),0,'lua');s.setObjectAlias(panel,'PatientViewPanel');time.sleep(.3);s.callScriptFunction('fullBody',panel)
plan=json.loads((OUT/'registered_plan.json').read_text());B=np.array(plan['rotation']);shift=np.array(plan['shift'])
for sign in [-1,1]:
 rot=Rotation.from_euler('z',sign*plan['patient_pose']['leg_abduction_each_deg'],degrees=True).as_matrix();pos=np.array([-.035,sign*.08,.39])+rot@np.array([.47,sign*.035,0]);s.setObjectMatrix(s.getObject('/LegSupport__1' if sign<0 else '/LegSupport_1'),-1,np.column_stack((B@rot,B@pos+shift)).flatten().tolist())
def look(pos,target,up):
 z=target-pos;z/=np.linalg.norm(z);x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x);return np.column_stack((x,y,z,pos)).flatten().tolist()
sensor=s.createVisionSensor(1+2+4+128,[1000,1000,0,0],[.001,12,math.radians(45),.001,0,0,.10,.13,.17,0,0])
hidden=[]
for obj in s.getObjectsInTree(s.handle_scene,s.handle_all,0):
 name=s.getObjectAlias(obj)
 if name.endswith('_visual') or name.endswith('_RCM_fixed_ring'):
  hidden.append((obj,s.getObjectInt32Param(obj,s.objintparam_visibility_layer)));s.setObjectInt32Param(obj,s.objintparam_visibility_layer,0)
for name,localpos,localtarget in [('leg_fix_top',[.10,0,2.65],[.10,0,.5]),('leg_fix_oblique',[.55,-1.6,1.8],[.22,0,.5])]:
 s.setObjectMatrix(sensor,-1,look(B@np.array(localpos)+shift,B@np.array(localtarget)+shift,B@np.array([-1.,0,0])));s.handleVisionSensor(sensor);raw,res=s.getVisionSensorImg(sensor);Image.frombytes('RGB',res,raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(OUT/(name+'.png'))
for obj,layer in hidden:s.setObjectInt32Param(obj,s.objintparam_visibility_layer,layer)
s.removeObjects([sensor]);s.callScriptFunction('sceneOverview',panel)
stage=Path(tempfile.gettempdir())/'davinci_foot_docking_rcm_legs_fixed.ttt';s.saveScene(str(stage));shutil.copy2(stage,OUT/stage.name)
scanner=s.createScript(s.scripttype_customization,"""sim=require('sim')
function sysCall_init() end
function scan(ps) local hits={} for i,p in ipairs(ps) do if sim.checkCollision(p[1],p[2])>0 then local n,v=sim.checkCollisionEx(p[1],p[2]);hits[#hits+1]={index=i,segments=v or {}} end end return hits end""",0,'lua');s.setObjectAlias(scanner,'RegistrationCollisionProbe')
print('Corrected skin applied to isolated final scene.',flush=True)
