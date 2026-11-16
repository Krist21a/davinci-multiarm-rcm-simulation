from plan_prostate_rcm import *
from scipy.optimize import least_squares
OUT=ROOT.parent/'davinci_prostate_rcm'
class Arm:
 def __init__(self,i):
  self.pre=f'psm{i}';chain=[];link=self.pre+'_rot_ref_respondable'
  while link!='world':
   j=next(j for j in records if j['child']==link);chain.insert(0,j);link=j['parent']
  self.chain=chain;self.js=[j for j in chain if j['kind']!='fixed' and j['name'].endswith('_joint')];self.names=[j['name'] for j in self.js];self.lo=np.array([j['limits'][0] for j in self.js]);self.hi=np.array([j['limits'][1] for j in self.js])
 def fk(self,x):
  q=dict(zip(self.names,x));q[self.pre+'_joint7']=.04;T=np.eye(4);axes=[]
  for j in self.chain:
   T=T@j['T'];v=q.get(j['name'],0)
   if j in self.js:axes.append((T[:3,3].copy(),T[:3,:3]@j['axis']))
   if j['kind']=='revolute':D=np.eye(4);D[:3,:3]=Rotation.from_rotvec(j['axis']*v).as_matrix();T=T@D
   elif j['kind']=='prismatic':T[:3,3]+=T[:3,:3]@j['axis']*v
  return T,axes
 def solve(self,port,end,seed=None):
  axis=end-port;axis/=np.linalg.norm(axis);desired=end-.4498*axis
  def fun(x):
   T,ax=self.fk(x);return np.r_[(T[:3,3]-desired)*3,T[:3,2]-axis]
  def jac(x):
   T,axes=self.fk(x);return np.array([np.r_[a*3,[0,0,0]] if j['kind']=='prismatic' else np.r_[np.cross(a,T[:3,3]-p)*3,np.cross(a,T[:3,2])] for j,(p,a) in zip(self.js,axes)]).T
  sol=least_squares(fun,np.zeros(len(self.js)) if seed is None else seed,jac=jac,bounds=(self.lo+1e-6,self.hi-1e-6),max_nfev=100,gtol=1e-9,ftol=1e-8,xtol=1e-8)
  return {'arm':self.pre,'q':dict(zip(self.names,sol.x.tolist())),'error':float(np.linalg.norm(fun(sol.x))),'port':port.tolist(),'target':end.tolist(),'axis':axis.tolist(),'shaft_target':desired.tolist(),'length':.4498}
if __name__=='__main__':
 arms=[Arm(i) for i in range(1,4)];best=None
 for cranial in [-.18,-.10]:
  localports=[surface(cranial,-.095),surface(cranial-.035,0),surface(cranial,.095)]
  for angle in [-90,-60,-120,0]:
   B=Rotation.from_euler('z',angle,degrees=True).as_matrix()
   for dy in [-1.15,-1.35,-1.55]:
    for dx in [-.35,-.15,.05]:
     shift=np.array([dx,dy,.06]);ports=[B@p+shift for p in localports];ends=[B@(target+np.array([.005,(i-1)*.018,.018]))+shift for i in range(3)]
     solutions=[a.solve(p,e) for a,p,e in zip(arms,ports,ends)];score=max(x['error'] for x in solutions)
     if best is None or score<best['score']:
      best={'score':score,'rotation':B.tolist(),'shift':shift.tolist(),'local_ports':[p.tolist() for p in localports],'arms':solutions};print('BEST',cranial,angle,dx,dy,score,[round(x['error'],5) for x in solutions],flush=True);(OUT/'layout_plan.json').write_text(json.dumps(best,indent=2),encoding='utf-8')
     if score<1e-6:print('FEASIBLE',flush=True);raise SystemExit
