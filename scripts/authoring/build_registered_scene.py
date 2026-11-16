from davinci_geometry import *
import json,math,time,tempfile,shutil
OUT=ROOT.parent/'davinci_prostate_rcm';plan=json.loads((OUT/'registered_plan.json').read_text());B=np.array(plan['rotation']);shift=np.array(plan['shift'])
sys.path.insert(0,r'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python\src')
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
c=RemoteAPIClient(port=23020);s=c.require('sim');c.socket.RCVTIMEO=120000
stage=Path(tempfile.gettempdir())/'davinci_registration_source.ttt';shutil.copy2(ROOT.parent/'davinci_patient'/'davinci_male_patient.ttt',stage);s.loadScene(str(stage));time.sleep(.3)
by={s.getObjectAlias(h):h for h in s.getObjectsInTree(s.handle_scene,s.handle_all,0)}
if 'setup_lift_revision' in plan:
 rev=plan['setup_lift_revision'];lo,hi=rev['new_limits_m'];s.setJointInterval(by[rev['joint']],False,[lo,hi-lo])
for name,(lo,hi) in plan.get('joint_limit_revisions',{}).items():s.setJointInterval(by[name],False,[lo,hi-lo])
R=np.array([[0,0,-1],[1,0,0],[0,-1,0.]]);t=B@np.array([.78,0,.4652476])+shift
s.setObjectMatrix(by['BodyParts3D_Male'],-1,np.column_stack((B@R,t)).flatten().tolist());s.setObjectMatrix(by['TaskSurface'],-1,np.column_stack((B,B@np.array([0,0,.39])+shift)).flatten().tolist())
if plan.get('patient_pose'):
 from plan_prostate_rcm import obj,surface,target
 from patient_leg_pose import pose_legs,exterior_skin
 v,f=exterior_skin(*obj('FJ2810.obj'));local=v@R.T+np.array([.78,0,.4652476]);posed=local.copy()
 posed=pose_legs(local,plan['patient_pose']['leg_abduction_each_deg'],f)
 src=(posed-np.array([.78,0,.4652476]))@R
 s.removeObjects([by['Patient_Skin']]);h=s.createMeshShape(0,math.radians(30),src.flatten().tolist(),f.flatten().tolist());s.setObjectAlias(h,'Patient_Skin');s.setObjectParent(h,by['BodyParts3D_Male'],False);s.setShapeColor(h,None,s.colorcomponent_ambient_diffuse,[.72,.50,.38]);s.setShapeColor(h,None,s.colorcomponent_transparency,[1]);s.setObjectInt32Param(h,s.shapeintparam_static,1);s.setObjectInt32Param(h,s.shapeintparam_respondable,0);by['Patient_Skin']=h
 s.removeObjects([by['TaskSurface']])
 def tablepiece(name,size,pos,rotation):
  h=s.createPrimitiveShape(s.primitiveshape_cuboid,size,0);s.setObjectAlias(h,name);s.setObjectMatrix(h,-1,np.column_stack((B@rotation,B@pos+shift)).flatten().tolist());s.setShapeColor(h,None,s.colorcomponent_ambient_diffuse,[.18,.30,.34]);s.setObjectInt32Param(h,s.shapeintparam_static,1);s.setObjectInt32Param(h,s.shapeintparam_respondable,0);return h
 by['TaskSurface']=tablepiece('TaskSurface',[1.10,.74,.06],np.array([-.45,0,.39]),np.eye(3))
 for sign in [-1,1]:
  rot=Rotation.from_euler('z',sign*plan['patient_pose']['leg_abduction_each_deg'],degrees=True).as_matrix();pos=np.array([-.035,sign*.08,.39])+rot@np.array([.47,sign*.035,0]);tablepiece('LegSupport_'+str(sign),[.93,.24,.06],pos,rot)
 camera_port=surface(-.16,target[1]);camera_world=B@camera_port+shift;h=s.createDummy(.014);s.setObjectAlias(h,'PlannedCameraPort');s.setObjectPosition(h,-1,camera_world.tolist());plan['camera_port_world']=camera_world.tolist();(OUT/'registered_plan.json').write_text(json.dumps(plan,indent=2))
cfg=json.loads((ROOT.parent/'davinci_rcm'/'rcm_config.json').read_text())
for a,cf in zip(plan['arms'],cfg):
 for n,v in a['q'].items():s.setJointPosition(by[n],v)
 s.setJointPosition(by[a['arm']+'_joint7'],.04)
 m=np.array(s.getObjectMatrix(by[a['arm']+'_joint8'],-1)).reshape(3,4);axis=m[:,2];port=np.array(a['port']);err=np.linalg.norm(m[:,3]-a['shaft_target']);assert err<1e-6,(a['arm'],err)
 cf.update(rcm_world=port.tolist(),axis_home=axis.tolist(),transverse_x=m[:,0].tolist(),transverse_y=m[:,1].tolist(),shaft_home=m[:,3].tolist(),rcm_distance_home=float(np.dot(port-m[:,3],axis)))
 for j in cf['joints']:
  if j['name'] in plan.get('joint_limit_revisions',{}):j['limits']=plan['joint_limit_revisions'][j['name']]
 for suf in ['_RCM_reference','_RCM_projection']:s.setObjectPosition(by[a['arm']+suf],-1,port.tolist())
 s.removeObjects([by[a['arm']+'_RCM_fixed_ring']]);verts=[];faces=[]
 for i in range(48):
  aa=2*math.pi*i/48
  for j in range(8):
   bb=2*math.pi*j/8;rr=.008+.0006*math.cos(bb);verts.extend((port+rr*math.cos(aa)*m[:,0]+rr*math.sin(aa)*m[:,1]+.0006*math.sin(bb)*axis).tolist())
 for i in range(48):
  for j in range(8):
   aa=i*8+j;bb=((i+1)%48)*8+j;cc=((i+1)%48)*8+(j+1)%8;dd=i*8+(j+1)%8;faces.extend([aa,bb,cc,aa,cc,dd])
 h=s.createMeshShape(0,.5,verts,faces);s.setObjectAlias(h,a['arm']+'_RCM_fixed_ring');s.setShapeColor(h,None,s.colorcomponent_ambient_diffuse,[1,.55,.04]);s.setObjectInt32Param(h,s.shapeintparam_static,1);s.setObjectInt32Param(h,s.shapeintparam_respondable,0)
 goal=s.createDummy(.006);s.setObjectAlias(goal,a['arm']+'_ProstateApproachTarget');s.setObjectPosition(goal,-1,a['target'])
def lua(v):
 if isinstance(v,dict):return '{'+','.join('['+json.dumps(k)+']='+lua(x) for k,x in v.items())+'}'
 if isinstance(v,list):return '{'+','.join(lua(x) for x in v)+'}'
 return json.dumps(v)
code=(ROOT.parent/'davinci_rcm'/'rcm_controller.lua').read_text().replace('math.abs(yaw)>5','math.abs(yaw)>2').replace('math.abs(pitch)>4','math.abs(pitch)>1.5').replace('insertion<.02 or insertion>.06','insertion<.036 or insertion>.044').replace('and 5*wave','and 2*wave').replace('and 4*math.sin','and 1.5*math.sin').replace('and .02*wave','and .004*wave').replace('b[i]=-1e-9*q[i]','b[i]=-1e-9*(q[i]-arm.home[i])')
code=code.replace('math.abs(yaw)>2','math.abs(yaw)>1').replace('math.abs(pitch)>1.5','math.abs(pitch)>.75').replace('insertion<.036 or insertion>.044','insertion<.038 or insertion>.042').replace('and 2*wave','and 1*wave').replace('and 1.5*math.sin','and .75*math.sin').replace('and .004*wave','and .002*wave')
s.removeObjects([by['RCMController']]);h=s.createScript(s.scripttype_simulation,'CONFIG='+lua(cfg)+'\n'+code,0,'lua');s.setObjectAlias(h,'RCMController')
(OUT/'rcm_config.json').write_text(json.dumps(cfg,indent=2));(OUT/'rcm_controller.lua').write_text(code)
stage=Path(tempfile.gettempdir())/'davinci_prostate_registered.ttt';s.saveScene(str(stage));shutil.copy2(stage,OUT/stage.name)
# Return exact mesh intersections for contact classification, rather than ignoring entire organ pairs.
scanner=s.createScript(s.scripttype_customization,"""sim=require('sim')
function sysCall_init() end
function scan(ps)
 local hits={}
 for i,p in ipairs(ps) do if sim.checkCollision(p[1],p[2])>0 then local n,segments=sim.checkCollisionEx(p[1],p[2]);hits[#hits+1]={index=i,segments=segments or {}} end end
 return hits
end""",0,'lua');s.setObjectAlias(scanner,'RegistrationCollisionProbe');time.sleep(.3)
by={s.getObjectAlias(h):h for h in s.getObjectsInTree(s.handle_scene,s.handle_all,0)}
tool=[(n,h) for n,h in by.items() if n.startswith('psm') and n.endswith('_visual') and any(k in n for k in ['tool_main_respondable','rot_ref_respondable','wrist_ref_respondable','yaw_ref1_respondable','yaw_ref2_respondable'])]
anatomy=[(n,h) for n,h in by.items() if n.startswith('Patient_') and s.getObjectType(h)==s.sceneobject_shape]
pairs=[[h,b] for n,h in tool for nn,b in anatomy];labels=[[n,nn] for n,h in tool for nn,b in anatomy]
hits=s.callScriptFunction('scan',scanner,pairs);out=[]
for hit in hits:
 item={'pair':labels[hit['index']-1]};pts=np.array(hit['segments']).reshape(-1,3)
 if len(pts):item['bounds']=[pts.min(0).tolist(),pts.max(0).tolist()]
 out.append(item)
(OUT/'home_contacts.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
