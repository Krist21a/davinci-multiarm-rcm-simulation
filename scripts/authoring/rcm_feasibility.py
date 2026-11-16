from davinci_geometry import *
import xml.etree.ElementTree as E,json
from scipy.optimize import least_squares
OUT=ROOT.parent/'davinci_rcm';OUT.mkdir(exist_ok=True)
root=E.parse(ROOT/'davinci_multiarm.urdf').getroot()
records=[]
for j in root.findall('joint'):
    o=j.find('origin');t=np.eye(4);t[:3,3]=np.fromstring(o.get('xyz'),sep=' ');t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_matrix();ax=j.find('axis');lim=j.find('limit')
    records.append(dict(name=j.get('name'),kind=j.get('type'),parent=j.find('parent').get('link'),child=j.find('child').get('link'),T=t,axis=np.fromstring(ax.get('xyz'),sep=' ') if ax is not None else None,limits=[float(lim.get('lower')),float(lim.get('upper'))] if lim is not None else [0,0]))
def fk(q):
    frames={'world':np.eye(4)};axes={}
    for j in records:
        b=frames[j['parent']]@j['T'];d=np.eye(4);v=q.get(j['name'],0)
        if j['kind']!='fixed':axes[j['name']]=(b[:3,3],b[:3,:3]@j['axis'])
        if j['kind']=='revolute':d[:3,:3]=Rotation.from_rotvec(j['axis']*v).as_matrix()
        elif j['kind']=='prismatic':d[:3,3]=j['axis']*v
        frames[j['child']]=b@d
    return frames,axes
results=[];config=[]
for arm in range(1,4):
    pre=f'psm{arm}';homeq={pre+'_joint7':.04};f,axes=fk(homeq);t=f[pre+'_rot_ref_respondable'];p0=t[:3,3];a0=t[:3,2];rcm=p0+.30*a0
    allj=[j for j in records if j['name'].startswith(pre+'_') and j['name'].endswith('_joint') and j['kind']!='fixed']
    for mode in ['operating_only','last_setup_assist','rotational_assist','coordinated']:
        last_setup=[j['name'] for j in allj if '_setup' in j['name']][-1]
        js=[j for j in allj if mode=='coordinated' or (mode=='rotational_assist' and j['kind']=='revolute') or (mode=='last_setup_assist' and j['name']==last_setup) or any(k in j['name'] for k in ['_mount_','_shoulder_','_elbow_','_carrier_'])]
        lo=np.array([j['limits'][0] for j in js]);hi=np.array([j['limits'][1] for j in js]);q=np.zeros(len(js));errs=[];steps=[];positions=[]
        def residual(x,desired,depth):
            conf={j['name']:v for j,v in zip(js,x)};conf[pre+'_joint7']=depth;ff,aa=fk(conf);tt=ff[pre+'_rot_ref_respondable'];return np.r_[(tt[:3,3]-(rcm-(.34-depth)*desired))/.3,tt[:3,2]-desired]
        _,ax=fk(homeq);cols=[]
        for j in js:
            pj,aj=ax[j['name']];cols.append(np.r_[aj/.3,np.zeros(3)] if j['kind']=='prismatic' else np.r_[np.cross(aj,p0-pj)/.3,np.cross(aj,a0)])
        sv=np.linalg.svd(np.array(cols).T,compute_uv=False)
        for k in range(49):
            phase=2*np.pi*k/48;desired=a0+np.tan(np.deg2rad(5)*np.sin(phase))*t[:3,0]+np.tan(np.deg2rad(4)*np.sin(2*phase))*t[:3,1];desired/=np.linalg.norm(desired);depth=.04+.02*np.sin(phase)
            old=q.copy();sol=least_squares(lambda x:np.r_[residual(x,desired,depth),1e-7*(x-old)],q,bounds=(lo,hi),xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=100);q=sol.x;err=residual(q,desired,depth);errs.append(float(np.linalg.norm(err[:3])*.3));steps.append(float(max(abs(q-old))));positions.append(q.tolist())
        result=dict(arm=pre,mode=mode,joints=[j['name'] for j in js],jacobian_rank=int(sum(sv>1e-6)),singular_values=sv.tolist(),max_base_target_error_m=max(errs),max_sample_joint_change=max(steps),max_abs_q=np.max(abs(np.array(positions)),axis=0).tolist());results.append(result);print(json.dumps(result),flush=True)
        if mode=='rotational_assist':
            assert max(errs)<1e-5
            config.append(dict(arm=pre,joints=[dict(name=j['name'],type=j['kind'],limits=j['limits']) for j in js],rcm_world=rcm.tolist(),axis_home=a0.tolist(),transverse_x=t[:3,0].tolist(),transverse_y=t[:3,1].tolist(),shaft_home=p0.tolist(),insertion_home=.04,rcm_distance_home=.30))
(OUT/'feasibility.json').write_text(json.dumps(results,indent=2),encoding='utf-8');(OUT/'rcm_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
