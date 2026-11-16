from solve_prostate_layout import *
OUT=ROOT.parent/'davinci_prostate_rcm';ini=json.loads((OUT/'registered_plan.json').read_text());candidates=json.loads((OUT/'anatomy_path_candidates.json').read_text());B=np.array(ini['rotation']);shift=np.array(ini['shift']);ranked=[]
for i in range(3):
 arm=Arm(i+1);seed=np.array(list(ini['arms'][i]['q'].values()));results=[]
 cs=sorted(candidates,key=lambda a:a['target_surface_distance_m'])
 # Candidate ports are kept separated by side; PSM2 uses a distinct central/medial port.
 cs=[p for p in cs if (p['port'][1]<-.045 if i==0 else (abs(p['port'][1])<.075 if i==1 else p['port'][1]>.045))]
 for k,p in enumerate(cs[:65]):
  sol=arm.solve(B@np.array(p['port'])+shift,B@np.array(p['target'])+shift,seed);sol.update(path=p)
  sol['rank']=sol['error']*10+p['target_surface_distance_m']-min(p['clearance_sample_m'],.012)*.5;results.append(sol)
 results.sort(key=lambda a:a['rank']);ranked.append(results[:12]);print('arm',i+1,[(round(a['error'],5),round(a['path']['target_surface_distance_m']*1000,1),round(a['path']['clearance_sample_m']*1000,1)) for a in results[:5]],flush=True)
(OUT/'ranked_approaches.json').write_text(json.dumps(ranked,indent=2))
