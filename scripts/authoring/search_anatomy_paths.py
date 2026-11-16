from plan_prostate_rcm import *
from scipy.spatial import cKDTree
OUT=ROOT.parent/'davinci_prostate_rcm'
organfiles={'bladder':'FJ3149.obj','rectum':'FJ2571.obj','right_hip':'FJ3152.obj','left_hip':'FJ3288.obj','sacrum':'FJ3393.obj','right_ureter':'FJ3146.obj','left_ureter':'FJ3144.obj'}
points=[]
for name,file in organfiles.items():
 v,f=obj(file);v=v@R.T+np.array([.78,0,.4652476]);tt=v[f]
 for i in range(9):
  for j in range(9-i):points.append(tt[:,0]*(1-(i+j)/8)+tt[:,1]*i/8+tt[:,2]*j/8)
tree=cKDTree(np.concatenate(points));ptree=cKDTree(pv)
tri=tri[(tri[:,:,0].max(1)>-.32)&(tri[:,:,0].min(1)<-.025)&(tri[:,:,1].max(1)>-.15)&(tri[:,:,1].min(1)<.15)];surface.__globals__['tri']=tri
rng=np.random.default_rng(42);good=[];best=None
for k in range(12000):
 end=target+rng.uniform([-.045,-.048,-.035],[.035,.048,.050]);dist=ptree.query(end)[0]
 if dist<.006 or dist>.035 or tree.query(end)[0]<.006:continue
 port=surface(rng.uniform(-.30,-.055),rng.uniform(-.12,.12));path=port[None,:]+np.linspace(0,1,75)[:,None]*(end-port)[None,:];clear=float(tree.query(path)[0].min())
 if best is None or clear>best['clearance']:best={'clearance':clear,'port':port.tolist(),'target':end.tolist(),'target_distance':float(dist)}
 if clear>.006:
  good.append({'port':port.tolist(),'target':end.tolist(),'clearance_sample_m':clear,'target_surface_distance_m':float(dist)})
 if k%2000==0:print(k,len(good),flush=True)
(OUT/'anatomy_path_candidates.json').write_text(json.dumps(good,indent=2));print('safe approach candidates',len(good),'best',best,flush=True)
