from davinci_geometry import *
import json,itertools,time
from plan_prostate_rcm import obj
OUT=ROOT.parent/'davinci_prostate_rcm'
sys.path.insert(0,r'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python\src')
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
c=RemoteAPIClient(port=23020);s=c.require('sim');c.socket.RCVTIMEO=120000;ctrl=s.getObject('/RCMController');cfg=json.loads((OUT/'rcm_config.json').read_text())
c.setStepping(True);s.startSimulation();c.step();s.callScriptFunction('homeRCM',ctrl)
res={'manual_corners':[]}
for i in range(1,4):
 for yaw,pitch,ins in itertools.product([-1,1],[-.75,.75],[.038,.042]):
  ok=s.callScriptFunction('setRCMTarget',ctrl,i,yaw,pitch,ins);res['manual_corners'].append({'arm':i,'target':[yaw,pitch,ins],'accepted':ok})
s.callScriptFunction('homeRCM',ctrl)
assert not s.callScriptFunction('setRCMTarget',ctrl,1,2,0,.04)
res['out_of_range_rejected']=True
bladder=s.getObject('/Patient_Bladder');original=s.getObjectMatrix(bladder,-1);M=np.array(original).reshape(3,4);vv,_=obj('FJ3149.obj');center=vv.mean(0)@M[:,:3].T+M[:,3]
shaft=np.array(s.getObjectMatrix(s.getObject('/psm1_joint8'),-1)).reshape(3,4);delta=shaft[:,3]+.20*shaft[:,2]-center
joints=[s.getObject('/'+j['name']) for j in cfg[0]['joints']]+[s.getObject('/psm1_joint7')];before=[s.getJointPosition(h) for h in joints]
try:
 s.setObjectPosition(bladder,-1,(M[:,3]+delta).tolist())
 ok=s.callScriptFunction('setRCMTarget',ctrl,1,.2,0,.04)
 after=[s.getJointPosition(h) for h in joints];res['fault_injection']={'temporary_obstacle':'Bladder moved into external shaft path for rollback test, then restored','rejected':not ok,'joint_rollback_max_error':float(np.max(abs(np.array(after)-before))),'stats':s.callScriptFunction('getRCMStats',ctrl)[0]}
 assert not ok and res['fault_injection']['joint_rollback_max_error']<1e-12
finally:s.setObjectMatrix(bladder,-1,original)
res['obstacle_restored']=bool(np.max(abs(np.array(s.getObjectMatrix(bladder,-1))-original))<1e-12)
s.callScriptFunction('homeRCM',ctrl);s.stopSimulation()
for _ in range(30):
 if s.getSimulationState()==s.simulation_stopped:break
 c.step()
c.setStepping(False)
(OUT/'control_validation.json').write_text(json.dumps(res,indent=2),encoding='utf-8');print(json.dumps(res),flush=True)
