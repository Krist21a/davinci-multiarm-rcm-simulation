from davinci_geometry import *
import json,time,tempfile,math
from PIL import Image
OUT=ROOT.parent/'davinci_prostate_rcm'
sys.path.insert(0,r'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python\src')
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
c=RemoteAPIClient(port=23021);s=c.require('sim');c.socket.RCVTIMEO=120000
if s.getSimulationState()!=s.simulation_stopped:s.stopSimulation();time.sleep(.5)
s.loadScene(str(Path(tempfile.gettempdir())/'davinci_foot_docking_rcm.ttt'));time.sleep(.6)
panel=s.getObject('/PatientViewPanel');rcmpanel=s.getObject('/RCMControlPanel');skin=s.getObject('/Patient_Skin');checks=[]
for name in ['organsOnly','fullBody','transparentBody','pelvicAnatomy','fullBody','sceneOverview']:
 s.callScriptFunction(name,panel);checks.append(name)
color=s.getShapeColor(skin,None,s.colorcomponent_transparency);assert s.getObjectInt32Param(skin,s.objintparam_visibility_layer)==1
assert abs(color[1][0]-1)<1e-8,color
h=s.createVisionSensor(1+2+4+128,[1280,960,0,0],[.001,12,math.radians(45),.001,0,0,.10,.13,.17,0,0]);s.setObjectMatrix(h,-1,s.getObjectMatrix(s.getObject('/DefaultCamera'),-1));s.handleVisionSensor(h);raw,res=s.getVisionSensorImg(h);Image.frombytes('RGB',res,raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(OUT/'gui_whole_body_verified.png');s.removeObjects([h])
s.setBoolParam(s.boolparam_realtime_simulation,True);s.callScriptFunction('run',rcmpanel);time.sleep(3)
ctrl=s.getObject('/RCMController');s.callScriptFunction('hold',rcmpanel);s.callScriptFunction('armChanged',rcmpanel,0,0,1);s.callScriptFunction('yawChanged',rcmpanel,0,0,5);s.callScriptFunction('pitchChanged',rcmpanel,0,0,25);s.callScriptFunction('insertChanged',rcmpanel,0,0,41)
manual=s.callScriptFunction('getRCMStats',ctrl);assert np.allclose(manual[1]['target'],[.5,.25,.041]);assert manual[1]['converged'];s.callScriptFunction('home',rcmpanel);s.callScriptFunction('run',rcmpanel);s.callScriptFunction('sceneOverview',panel);time.sleep(3)
stats=s.callScriptFunction('getRCMStats',ctrl);assert all(a['failures']==0 and a['collision_blocks']==0 for a in stats)
result={'visible_gui':not s.getBoolParam(s.boolparam_headless),'scene':s.getStringParam(s.stringparam_scene_name),'view_callbacks':checks,'whole_body_opacity':color,'manual_controls_passed':True,'simulation_time_s':s.getSimulationTime(),'left_running':True,'stats':stats}
(OUT/'gui_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
