from davinci_geometry import *
import json,time,math,shutil,tempfile
from PIL import Image,ImageDraw
import imageio_ffmpeg
OUT=ROOT.parent/'davinci_prostate_rcm'
sys.path.insert(0,r'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python\src')
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
c=RemoteAPIClient(port=23020);s=c.require('sim');c.socket.RCVTIMEO=120000;ctrl=s.getObject('/RCMController');panel=s.getObject('/PatientViewPanel');skin=s.getObject('/Patient_Skin')
def look(pos,target):
 pos=np.array(pos,float);z=np.array(target)-pos;z/=np.linalg.norm(z);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x);return np.column_stack((x,y,z,pos)).flatten().tolist()
plan=json.loads((OUT/'registered_plan.json').read_text());p=np.mean([a['target'] for a in plan['arms']],axis=0)
views=[look([2.15,-3.15,2.0],[-.07,-.8,.8]),look(p+[.23,-.23,.30],p)]
sensors=[]
for view in views:
 h=s.createVisionSensor(1+2+4+128,[640,600,0,0],[.001,12,math.radians(45),.001,0,0,.10,.13,.17,0,0]);s.setObjectMatrix(h,-1,view);sensors.append(h)
def frame(h):
 s.handleVisionSensor(h);raw,res=s.getVisionSensorImg(h);return Image.frombytes('RGB',res,raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
writer=imageio_ffmpeg.write_frames(str(OUT/'foot_docking_rcm_demo.mp4'),(1280,720),fps=20,codec='libx264',quality=8);writer.send(None)
c.setStepping(True);s.setBoolParam(s.boolparam_realtime_simulation,False);s.startSimulation();maximum=[0,0,0]
try:
 for k in range(480):
  c.step();stats=s.callScriptFunction('getRCMStats',ctrl);t=s.getSimulationTime();stage=['SWING around skin ports','INSERT through skin ports','SWING + INSERT'][min(2,int((t%24)/8))]
  canvas=Image.new('RGB',(1280,720),'#18202a');d=ImageDraw.Draw(canvas);d.text((18,10),'FOOT-END DOCKING | 20 deg head-down | split-leg visual pose | '+stage,fill='white')
  s.setObjectInt32Param(skin,s.objintparam_visibility_layer,1);s.callScriptFunction('bonesVisible',panel,1);canvas.paste(frame(sensors[0]),(0,42))
  s.setObjectInt32Param(skin,s.objintparam_visibility_layer,0);s.callScriptFunction('bonesVisible',panel,0);canvas.paste(frame(sensors[1]),(640,42))
  d.text((18,650),'Static anatomy. Tips approach the prostate region; no tissue contact or deformation.',fill='#e0d9b8')
  for i,a in enumerate(stats):
   maximum[i]=max(maximum[i],a['drift_m']);d.text((18+i*420,675),f'PSM{i+1}: RCM {a["drift_m"]*1000:.6f} mm | blocks {a["collision_blocks"]}',fill='#ffc066')
  writer.send(canvas.tobytes())
  if k in [0,120,240,400]:canvas.save(OUT/f'foot_demo_{k:03d}.png')
  if k%160==159:print('recorded',k+1,'frames',flush=True)
finally:
 writer.close();s.stopSimulation()
 for _ in range(30):
  if s.getSimulationState()==s.simulation_stopped:break
  c.step()
 c.setStepping(False);s.removeObjects(sensors);s.callScriptFunction('fullBody',panel);s.callScriptFunction('sceneOverview',panel);s.setBoolParam(s.boolparam_realtime_simulation,True)
try:s.removeObjects([s.getObject('/RegistrationCollisionProbe')])
except Exception:pass
time.sleep(.2);stage=Path(tempfile.gettempdir())/'davinci_foot_docking_rcm.ttt';s.saveScene(str(stage));shutil.copy2(stage,OUT/stage.name)
(OUT/'video_validation.json').write_text(json.dumps({'frames':480,'fps':20,'duration_s':24,'max_rcm_error_m':maximum,'last_stats':stats},indent=2));print('saved final scene and video',flush=True)
