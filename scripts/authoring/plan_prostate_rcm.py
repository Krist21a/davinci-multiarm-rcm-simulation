from davinci_geometry import *
import json,xml.etree.ElementTree as E
from scipy.optimize import least_squares
# Reuse FK definitions without executing the previous feasibility experiment.
exec(Path(__file__).with_name('rcm_feasibility.py').read_text(encoding='utf-8').split('results=[];config=[]')[0])
OUT=ROOT.parent/'davinci_prostate_rcm';OUT.mkdir(exist_ok=True)
def obj(name):
 p=ROOT.parent/'davinci_patient'/'meshes'/name;v=[];f=[]
 for line in p.read_text().splitlines():
  a=line.split()
  if a and a[0]=='v':v.append([float(x)*.001 for x in a[1:4]])
  if a and a[0]=='f':
   ids=[int(x.split('/')[0])-1 for x in a[1:]]
   for k in range(1,len(ids)-1):f.append([ids[0],ids[k],ids[k+1]])
 return np.array(v),np.array(f)
R=np.array([[0,0,-1],[1,0,0],[0,-1,0.]])
skin,sf=obj('FJ2810.obj');skin=skin@R.T+np.array([.78,0,.4652476]);tri=skin[sf]
pv,pf=obj('FJ3139.obj');pv=pv@R.T+np.array([.78,0,.4652476]);target=pv.mean(0)
def surface(x,y):
 # Vertical ray through the anterior (upper) abdominal surface.
 a=tri[:,0,:2];u=tri[:,1,:2]-a;v=tri[:,2,:2]-a;w=np.array([x,y])-a;det=u[:,0]*v[:,1]-u[:,1]*v[:,0]
 with np.errstate(divide='ignore',invalid='ignore'):
  b=(w[:,0]*v[:,1]-w[:,1]*v[:,0])/det;c=(u[:,0]*w[:,1]-u[:,1]*w[:,0])/det
 mask=(b>=0)&(c>=0)&(b+c<=1)&(abs(det)>1e-12);zz=tri[:,0,2]+b*(tri[:,1,2]-tri[:,0,2])+c*(tri[:,2,2]-tri[:,0,2]);return np.array([x,y,zz[mask].max()])
if __name__=='__main__':
 print('target body',target,'tip offset',np.linalg.inv(fk({'psm1_joint7':.04})[0]['psm1_rot_ref_respondable'])@fk({'psm1_joint7':.04})[0]['psm1_tool_tip'],flush=True)
 # Ports above and lateral to pelvis; these are engineering candidates, not clinical recommendations.
 ports=[surface(-.21,-.09),surface(-.25,.0),surface(-.18,.09)]
 jslist=[[j for j in records if j['name'].startswith(f'psm{i}_') and j['kind']!='fixed' and (j['name'].endswith('_joint'))] for i in range(1,4)]
 best=None
 for dy in [-1.35,-1.45,-1.55]:
  for dx in [-.25,-.1,0,.15]:
   shift=np.array([dx,dy,0]);solutions=[]
   for i,(js,port) in enumerate(zip(jslist,ports),1):
    pre=f'psm{i}';end=target+shift+np.array([0,(i-2)*.012,.018]);p=port+shift;axis=(end-p);axis/=np.linalg.norm(axis)
    off=np.linalg.inv(fk({pre+'_joint7':.04})[0][pre+'_rot_ref_respondable'])@fk({pre+'_joint7':.04})[0][pre+'_tool_tip'];length=off[2,3]
    desired=end-length*axis
    def fun(x):
     q={j['name']:v for j,v in zip(js,x)};q[pre+'_joint7']=.04;f,_=fk(q);T=f[pre+'_rot_ref_respondable'];return np.r_[(T[:3,3]-desired)*3,T[:3,2]-axis]
    lo=np.array([j['limits'][0] for j in js]);hi=np.array([j['limits'][1] for j in js]);sol=least_squares(fun,np.zeros(len(js)),bounds=(lo+1e-7,hi-1e-7),max_nfev=160,gtol=1e-9,ftol=1e-9,xtol=1e-9)
    solutions.append({'arm':pre,'q':dict(zip([j['name'] for j in js],sol.x.tolist())),'error':float(np.linalg.norm(fun(sol.x))),'port':p.tolist(),'target':end.tolist(),'axis':axis.tolist(),'shaft_target':desired.tolist(),'length':float(length)})
   score=max(x['error'] for x in solutions);print('placement',dx,dy,'errors',[round(x['error'],5) for x in solutions],flush=True)
   if best is None or score<best['score']:best={'score':score,'shift':shift.tolist(),'arms':solutions}
 (OUT/'initial_plan.json').write_text(json.dumps(best,indent=2),encoding='utf-8');print(json.dumps(best),flush=True)
