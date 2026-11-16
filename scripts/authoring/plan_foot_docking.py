from solve_prostate_layout import *
OUT=ROOT.parent/'davinci_prostate_rcm'
paths=json.loads((OUT/'anatomy_path_candidates.json').read_text())
old=json.loads((OUT/'superseded_side_docking_plan.json').read_text())
B=Rotation.from_euler('z',90,degrees=True).as_matrix()@Rotation.from_euler('y',-20,degrees=True).as_matrix()
arms=[Arm(i) for i in range(1,4)]
for a in arms:
 for k,j in enumerate(a.js):
  if j['kind']=='prismatic':a.lo[k]=-.06;a.hi[k]=.35 if a.pre=='psm1' else .25
  elif '_setup' in j['name'] or '_mount' in j['name']:a.lo[k]=-math.pi if False else -np.pi;a.hi[k]=np.pi
  else:a.lo[k]=-1.2;a.hi[k]=1.2
rng=np.random.default_rng(31);best=None
for center_y in [-1.0,-.95,-.85]:
 for center_z in [.65,.75,.85]:
  shift=np.array([-.0691917,center_y,center_z])-B@target;solutions=[]
  for i,a in enumerate(arms):
   path=old['arms'][[2,1,0][i]]['path'];port=B@np.array(path['port'])+shift;end=B@np.array(path['target'])+shift;winner=None
   for k in range(5):
    seed=np.zeros(len(a.js)) if k==0 else rng.uniform(np.maximum(a.lo,-1.4),np.minimum(a.hi,1.4));seed[0]=.12
    sol=a.solve(port,end,seed);sol['path']=path
    if winner is None or sol['error']<winner['error']:winner=sol
    if sol['error']<1e-7:break
   solutions.append(winner)
  score=max(a['error'] for a in solutions);print(center_y,center_z,score,flush=True)
  if best is None or score<best['score']:
   best={'score':score,'rotation':B.tolist(),'shift':shift.tolist(),'local_ports':[a['path']['port'] for a in solutions],'arms':solutions,'patient_pose':{'docking':'foot end, centerline aligned','head_down_deg':20,'leg_abduction_each_deg':30},'joint_limit_revisions':{j['name']:[float(a.lo[k]),float(a.hi[k])] for a in arms for k,j in enumerate(a.js)},'limit_basis':'Original manifest states CAD limits were conservative demo bounds, not manufacturer limits. Revised exploratory simulation bounds require mesh collision validation; hardware travel remains unverified.'};(OUT/'foot_docking_plan.json').write_text(json.dumps(best,indent=2))
  if score<1e-7:raise SystemExit
