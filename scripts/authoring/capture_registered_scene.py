from davinci_geometry import *
import json,math,time
from PIL import Image
OUT=ROOT.parent/'davinci_prostate_rcm'
sys.path.insert(0,r'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python\src')
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
c=RemoteAPIClient(port=23020);s=c.require('sim');c.socket.RCVTIMEO=120000
def look(pos,target,up=np.array([0.,0.,1.])):
 pos=np.array(pos,float);z=np.array(target)-pos;z/=np.linalg.norm(z);x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x);return np.column_stack((x,y,z,pos)).flatten().tolist()
plan=json.loads((OUT/'registered_plan.json').read_text());p=np.mean([a['target'] for a in plan['arms']],axis=0)
views={'overview':look([2.3,-3.25,2.15],[-.07,-.8,.83]),'body':look([1.35,-1.55,1.7],[-.07,-1.08,.64]),'pelvis':look(p+[.35,-.35,.42],p)}
def lua(v):
 if isinstance(v,dict):return '{'+','.join('['+json.dumps(k)+']='+lua(x) for k,x in v.items())+'}'
 if isinstance(v,list):return '{'+','.join(lua(x) for x in v)+'}'
 return json.dumps(v)
text=(OUT/'patient_panel.lua').read_text(encoding='utf-8').split('\n',1)[1];text=text.replace('Supine / original anatomical scale / static visual model','Foot-end docking / 20 deg head-down / split legs').replace('RCM rings are virtual test points; ports NOT registered.','Abdominal RCM ports / static anatomy / no tissue response.')
text='VIEWS='+lua(views)+'\n'+text;s.removeObjects([s.getObject('/PatientViewPanel')]);panel=s.createScript(s.scripttype_customization,text,0,'lua');s.setObjectAlias(panel,'PatientViewPanel');time.sleep(.3)
(OUT/'patient_panel.lua').write_text(text,encoding='utf-8');s.callScriptFunction('fullBody',panel)
h=s.createVisionSensor(1+2+4+128,[1280,960,0,0],[.001,12,math.radians(45),.001,0,0,.10,.13,.17,0,0])
for name in ['overview','body','pelvis']:
 if name=='pelvis':s.callScriptFunction('transparentBody',panel)
 s.setObjectMatrix(h,-1,views[name]);s.handleVisionSensor(h);raw,res=s.getVisionSensorImg(h);Image.frombytes('RGB',res,raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(OUT/(name+'.png'))
s.removeObjects([h]);s.callScriptFunction('fullBody',panel);s.callScriptFunction('sceneOverview',panel)
