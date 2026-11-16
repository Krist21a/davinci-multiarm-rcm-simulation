from solve_prostate_layout import *
import time
OUT=ROOT.parent/'davinci_prostate_rcm';plan=json.loads((OUT/'registered_plan.json').read_text())
sys.path.insert(0,r'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python\src')
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
c=RemoteAPIClient(port=23020);s=c.require('sim');c.socket.RCVTIMEO=120000
by={s.getObjectAlias(h):h for h in s.getObjectsInTree(s.handle_scene,s.handle_all,0)};pairs=json.loads((OUT/'collision_pairs.json').read_text())['labels']
checker=s.createScript(s.scripttype_customization,"""sim=require('sim')
function sysCall_init() end
function test(hs,qs,pairs)
 for i,h in ipairs(hs) do sim.setJointPosition(h,qs[i]) end
 local hits={};local length=0;for i,p in ipairs(pairs) do if sim.checkCollision(p[1],p[2])>0 then hits[#hits+1]=i;local n,v=sim.checkCollisionEx(p[1],p[2]);for j=1,#v,6 do length=length+math.sqrt((v[j]-v[j+3])^2+(v[j+1]-v[j+4])^2+(v[j+2]-v[j+5])^2) end end end;return {hits=hits,length=length}
end""",0,'lua');time.sleep(.3)
rng=np.random.default_rng(18);saved=[]
for idx in range(3):
 arm=Arm(idx+1);pre=arm.pre;row=plan['arms'][idx];seed=np.array(list(row['q'].values()));names=arm.names;hs=[by[n] for n in names]
 for k,name in enumerate(names):
  if name in plan.get('joint_limit_revisions',{}):arm.lo[k],arm.hi[k]=plan['joint_limit_revisions'][name]
 if idx==0 and ('--extended-lift' in sys.argv or 'setup_lift_revision' in plan):
  arm.hi[0]=.35;s.setJointInterval(hs[0],False,[-.06,.41]);plan['setup_lift_revision']={'joint':names[0],'old_limits_m':[-.06,.06],'new_limits_m':[-.06,.35],'basis':'CAD mast z=0.11..1.71 m; PSM1 sleeve z=0.594..0.944 m at q=0. Geometry-based simulation allowance, not verified actuator travel.'}
 # The three intended shaft/skin entry intersections are checked separately with their coordinates.
 pp=[p for p in pairs if any(n.startswith(pre+'_') for n in p) and not (p[1]=='Patient_Skin' and 'tool_main_respondable' in p[0])];handles=[[by[n] for n in p] for p in pp]
 best=None
 for k in range(250):
  x=seed if k==0 else np.clip(seed+rng.normal(0,[.08,.18,.35,.55][k%4],len(seed)),arm.lo+1e-5,arm.hi-1e-5)
  sol=arm.solve(np.array(row['port']),np.array(row['target']),x)
  if sol['error']>1e-6:continue
  q=np.array(list(sol['q'].values()));contact=s.callScriptFunction('test',checker,hs,q.tolist(),handles);hits=contact['hits'];margin=float(np.min(np.minimum(q-arm.lo,arm.hi-q)[1:]));score=contact['length']+.001*len(hits)+max(0,.035-margin)*.1
  if best is None or score<best['score']:
   best={'score':score,'solution':sol,'hits':[pp[i-1] for i in hits],'margin':margin};seed=q.copy();print(pre,k,len(hits),margin,'intersection length',contact['length'],flush=True)
  if len(hits)==0 and margin>.035:break
 if best:
  row['q']=best['solution']['q'];row['error']=best['solution']['error'];saved.append(best);s.callScriptFunction('test',checker,hs,list(row['q'].values()),[])
(OUT/'posture_search.json').write_text(json.dumps(saved,indent=2));(OUT/'registered_plan.json').write_text(json.dumps(plan,indent=2));s.removeObjects([checker])
