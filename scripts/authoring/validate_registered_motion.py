from davinci_geometry import *
import json,time,math
import xml.etree.ElementTree as E
OUT=ROOT.parent/'davinci_prostate_rcm'
sys.path.insert(0,r'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python\src')
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
c=RemoteAPIClient(port=23020);s=c.require('sim');c.socket.RCVTIMEO=120000
by={s.getObjectAlias(h):h for h in s.getObjectsInTree(s.handle_scene,s.handle_all,0)};scanner=by['RegistrationCollisionProbe'];ctrl=by['RCMController']
plan=json.loads((OUT/'registered_plan.json').read_text());cfg=json.loads((OUT/'rcm_config.json').read_text());ports={a['arm']:np.array(a['port']) for a in plan['arms']}
shapes={n:h for n,h in by.items() if s.getObjectType(h)==s.sceneobject_shape}
tools={n:h for n,h in shapes.items() if n.startswith('psm') and n.endswith('_visual') and any(k in n for k in ['tool_main_respondable','rot_ref_respondable','wrist_ref_respondable','yaw_ref1_respondable','yaw_ref2_respondable'])}
cad={n:h for n,h in shapes.items() if n.endswith('_visual') and n not in tools and 'tool_tip' not in n}
anatomy={n:h for n,h in shapes.items() if n.startswith('Patient_')}
pairs=[];labels=[];categories=[]
def pair(n,m,category):pairs.append([shapes[n],shapes[m]]);labels.append([n,m]);categories.append(category)
tables=[n for n in shapes if n=='TaskSurface' or n.startswith('LegSupport_')]
for n in tools:
 for m in anatomy:pair(n,m,'instrument_anatomy')
 for m in list(cad)+tables:
  if n.endswith('_tool_main_respondable_visual') and m==n[:4]+'_cad_slide_visual':continue
  pair(n,m,'instrument_structure')
for n in cad:
 for m in anatomy:pair(n,m,'structure_anatomy')
 for m in tables:pair(n,m,'structure_table')
armshapes={i:[n for n in list(cad)+list(tools) if n.startswith(f'psm{i}_')] for i in range(1,4)}
for i in range(1,4):
 for j in range(i+1,4):
  for n in armshapes[i]:
   for m in armshapes[j]:pair(n,m,'cross_arm')
urdf=E.parse(ROOT/'davinci_multiarm.urdf').getroot();adjacent={frozenset((j.find('parent').get('link'),j.find('child').get('link'))) for j in urdf.findall('joint')}
for names in armshapes.values():
 for i,n in enumerate(names):
  for m in names[i+1:]:
   if frozenset((n.removesuffix('_visual'),m.removesuffix('_visual'))) in adjacent:continue
   if 'yaw_ref' in n and 'yaw_ref' in m:continue
   if n in tools or m in tools:continue # already tested against own structures; wrist assembly has intentional mating faces
   pair(n,m,'nonadjacent_same_arm_structure')
def check():
 bad=[];expected=[]
 for hit in s.callScriptFunction('scan',scanner,pairs):
  idx=hit['index']-1;names=labels[idx];points=np.array(hit['segments']).reshape(-1,3);port=ports.get(names[0][:4]);allowed=False;radius=None
  if names[0] in tools and names[1]=='Patient_Skin' and 'tool_main' in names[0] and len(points):
   radius=float(np.linalg.norm(points-port,axis=1).max());allowed=radius<=.016
  item={'pair':names,'category':categories[idx]}
  if radius is not None:item['port_intersection_max_radius_m']=radius
  (expected if allowed else bad).append(item)
 return bad,expected
bad,expected=check();print('HOME',json.dumps(bad),'expected port crossings',len(expected),'pairs',len(pairs),flush=True)
(OUT/'collision_pairs.json').write_text(json.dumps({'labels':labels,'categories':categories},indent=2))
if '--home-only' in sys.argv:raise SystemExit
result={'pair_count':len(pairs),'home_unexpected':bad,'home_port_crossings':expected,'samples':[],'max_rcm_error_m':[0,0,0],'solver_failures':[0,0,0],'unexpected_events':[],'port_radius_threshold_m':.016}
from plan_prostate_rcm import obj
pv,pf=obj('FJ3139.obj');pm=np.array(s.getObjectMatrix(by['BodyParts3D_Male'],-1)).reshape(3,4);pt=(pv@pm[:,:3].T+pm[:,3])[pf]
def surface_distance(p):
 a=pt[:,0];u=pt[:,1]-a;v=pt[:,2]-a;normal=np.cross(u,v);nn=np.sum(normal*normal,axis=1);w=p-a;dot=np.sum(w*normal,axis=1);project=p-normal*(dot/np.maximum(nn,1e-20))[:,None];ww=project-a
 uu=np.sum(u*u,axis=1);vv=np.sum(v*v,axis=1);uv=np.sum(u*v,axis=1);wu=np.sum(ww*u,axis=1);wv=np.sum(ww*v,axis=1);den=uu*vv-uv*uv;b=(wu*vv-wv*uv)/np.maximum(den,1e-20);cc=(wv*uu-wu*uv)/np.maximum(den,1e-20);inside=(b>=0)&(cc>=0)&(b+cc<=1)&(nn>1e-20)
 dist=np.where(inside,abs(dot)/np.sqrt(np.maximum(nn,1e-20)),np.inf)
 for j,k in [(0,1),(1,2),(2,0)]:
  edge=pt[:,k]-pt[:,j];t=np.clip(np.sum((p-pt[:,j])*edge,axis=1)/np.maximum(np.sum(edge*edge,axis=1),1e-20),0,1);dist=np.minimum(dist,np.linalg.norm(p-pt[:,j]-t[:,None]*edge,axis=1))
 return float(dist.min())
fixed={n:np.array(s.getObjectMatrix(h,-1)) for n,h in by.items() if n in ['BodyParts3D_Male','TaskSurface'] or n.endswith('_RCM_reference') or n.endswith('_lift_joint')}
result.update(independent_max_rcm_error_m=[0,0,0],collision_blocks=[0,0,0],fixed_frame_max_change=0,tip_prostate_surface_distance_range_m=[[1e9,0] for _ in range(3)],tip_approach_target_max_error_m=[0,0,0])
c.setStepping(True);s.setBoolParam(s.boolparam_realtime_simulation,False);s.startSimulation()
for k in range(481):
 c.step();stats=s.callScriptFunction('getRCMStats',ctrl)
 for i,a in enumerate(stats):result['max_rcm_error_m'][i]=max(result['max_rcm_error_m'][i],a['drift_m']);result['solver_failures'][i]=a['failures'];result['collision_blocks'][i]=a.get('collision_blocks',0)
 if k%10==0:
  bad,expected=check();t=s.getSimulationTime();result['samples'].append({'time':t,'stats':stats,'expected_port_crossings':expected})
  for i,a in enumerate(plan['arms']):
   mat=np.array(s.getObjectMatrix(by[a['arm']+'_joint8'],-1)).reshape(3,4);root=mat[:,3];axis=mat[:,2];port=ports[a['arm']];err=float(np.linalg.norm(np.cross(port-root,axis)));result['independent_max_rcm_error_m'][i]=max(result['independent_max_rcm_error_m'][i],err)
   tip=root+.4498*axis;dist=surface_distance(tip);result['tip_prostate_surface_distance_range_m'][i][0]=min(result['tip_prostate_surface_distance_range_m'][i][0],dist);result['tip_prostate_surface_distance_range_m'][i][1]=max(result['tip_prostate_surface_distance_range_m'][i][1],dist)
   result['tip_approach_target_max_error_m'][i]=max(result['tip_approach_target_max_error_m'][i],float(np.linalg.norm(tip-np.array(a['target']))))
  for name,m in fixed.items():result['fixed_frame_max_change']=max(result['fixed_frame_max_change'],float(np.max(abs(np.array(s.getObjectMatrix(by[name],-1))-m))))
  if bad:result['unexpected_events'].append({'time':t,'contacts':bad})
 if k%120==0:print('sample',k,'failures',result['solver_failures'],'collision snapshots',len(result['unexpected_events']),flush=True)
s.stopSimulation()
for _ in range(30):
 if s.getSimulationState()==s.simulation_stopped:break
 c.step()
c.setStepping(False);s.setBoolParam(s.boolparam_realtime_simulation,True)
(OUT/'motion_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(OUT/'collision_pairs.json').write_text(json.dumps({'labels':labels,'categories':categories},indent=2))
print('RESULT',result['max_rcm_error_m'],result['solver_failures'],'unexpected',len(result['unexpected_events']),flush=True)
