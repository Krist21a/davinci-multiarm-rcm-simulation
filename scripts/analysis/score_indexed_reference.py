"""Seal sensor-only predictions, then check indexed landmarks against matched structural records."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from replay_bilateral_3d import ROOT, angle, digest, load, variants, write
from indexed_cut_perception import IndexedCutPerception
from indexed_reference import decode, measure
from junction_material_support import facets
from seam_topology import material_state

SPEC=ROOT/'config/task/m5_indexed_reference_v1.json'
HERE=Path(__file__)
SOURCES=[HERE,SPEC,HERE.with_name('replay_bilateral_3d.py'),HERE.with_name('junction_material_support.py'),
         *(ROOT/'scripts/vision'/s for s in ('indexed_reference.py','indexed_cut_perception.py',
                                              'cut_perception.py','bilateral_surface.py')),
         *(ROOT/'scripts/task'/s for s in ('seam_topology.py','workspace.py'))]


def synthetic_image(n,case):
    # Independent ray/plane fixture with the mesh's known 1-mm material grid.
    w,h,f=208,184,1300.
    yy,xx=np.indices((h,w))
    rays=np.stack([(w/2-xx-.5)/f,(h/2-yy-.5)/f,np.ones((h,w))],axis=-1)
    a=np.deg2rad(23)
    v=np.array([0,np.cos(a),np.sin(a)])
    normal=np.array([0,-np.sin(a),np.cos(a)])
    c=np.array([0,0,.105])
    depth=(c@normal)/(rays@normal)
    xyz=rays*depth[...,None]-c
    u,jv=xyz[:,:,0]*1000+11.5,(xyz@v)*1000+6.5
    col,row=np.floor(u).astype(int),np.floor(jv).astype(int)
    printed=(row>=0)&(row<13)&(col>=7)&(col<=15)
    h0=(u-col)>=(jv-row)
    central=col==11
    live=printed&(~central|(row>=n))
    if case=='orphan':
        live &= ~((col==10)&np.isin(row,[6,7]))
    if case=='bridge':
        live |= (row==5)&central
    if case in ('half0','half1','hidden_half'):
        kept=(row==8)&central&(h0 if case!='half1' else ~h0)
        live |= kept
    if case=='missing_station':
        live &= ~((col==9)&(row==7))
    lane=np.select([col<=8,col==9,col==10,central&~h0,central&h0,col==12,col==13],[0,1,2,3,4,5,6],7)
    colour=np.stack([42+13*row,50+21*lane,42+25*((3*row+5*lane)%8)],axis=-1)
    im=np.zeros((h,w,3),np.uint8)
    z=np.full((h,w),.18)
    im[live],z[live]=colour[live],depth[live]
    if case in ('foreground','hidden_half','bad_depth'):
        occ=(row==8)&central if case!='hidden_half' else kept
        im[occ],z[occ]=100,np.nan if case=='bad_depth' else .07
    if case=='shutter':
        im[:],z[:]=90,.03
    return im,z,w,h,f,c,v


def controls():
    rows=[]
    cases=[(n,'plain',n) for n in (0,3,6,8,9,10,11)]
    cases += [(8,'orphan',8),(9,'half0',8),(9,'half1',8),
              *((9,c,None) for c in ('bridge','hidden_half','foreground','bad_depth','shutter','missing_station'))]
    for n,case,want in cases:
        im,z,w,h,f,c,v=synthetic_image(n,case)
        values=[]
        for dep in variants(z):
            obs=measure(im,dep,w,h,f,.3)
            got=obs['local_mm']
            errors=[float(np.linalg.norm(np.array(s['camera_m'])-(c+np.array([(s['column']-11.5)*.001,0,0])
                         +(s['row']-6.5)*.001*v))*1000) for s in obs['stations']]
            ids={(s['row'],s['column']) for s in obs['stations']}
            ok=got==(None if want in (0,11) else want)
            if case.startswith('half'):
                ok=got in (None,8)
            if n==0:
                ok=ok and obs.get('first_present_row')==0
            if case=='orphan':
                ok=ok and all((j,k) in ids for j in (7,8) for k in (9,14)) and obs['front_xyz'] is None
            if case=='missing_station':
                ok=ok and (8,9) not in ids
            ok=bool(ok and all(e<.5 for e in errors) and all(k in (9,14) for _,k in ids)
                    and obs['prefix_mm'] is None and not obs['motion_authorized'])
            values.append(dict(passed=ok,local_mm=got,stations=len(ids),max_mm=max(errors,default=None),
                               apertures=obs.get('apertures'),first_present=obs.get('first_present_row')))
        rows.append(dict(length=n,case=case,variants=values))
    im,z,w,h,f,_,_=synthetic_image(9,'plain')
    for case,bad in [('wrong_shape',z[:3]),('invalid',np.full_like(z,np.nan))]:
        obs=measure(im,bad,w,h,f,.3)
        rows.append(dict(length=9,case=case,variants=[dict(passed=not obs['stations'] and obs['local_mm'] is None)]))
    return rows


def seal(out,reg,cal):
    images={}
    files={}
    for r in reg['criteria']['runs']:
        case=r['name']
        paths=sorted((out/case/'hd').glob('*.npz'))
        if len(paths)!=(2 if case=='pull' else 6):
            raise ValueError(f'{case}: incomplete acquisition')
        if load(out/case/'calibration.json')!=cal:
            raise ValueError('inconsistent calibration')
        for p in paths:
            sha=digest(p)
            files[p.relative_to(out).as_posix()]=sha
            if sha in images:
                continue
            with np.load(p,allow_pickle=False) as d:
                if set(d.files)!={'rgb','depth'}:
                    raise ValueError('privileged sensor fields')
                im,z=d['rgb'],d['depth']
            if im.shape!=(960,1280,3) or im.dtype!=np.uint8 or z.shape!=im.shape[:2] or z.dtype!=np.float32:
                raise ValueError('not native archived RGB-D')
            values=[]
            for dep in variants(z):
                observation=IndexedCutPerception(**cal).estimate(im,dep,np.eye(3),np.zeros(3))
                if observation['cut']['front_valid'] or observation['cut']['progress_m'] is not None:
                    raise ValueError('passive material reference supplied a control fix')
                values.append(dict(reference=observation['material_reference'],tool=observation['scissors']))
            images[sha]=values
    write(out/'predictions.json',dict(files=files,images=images,lock=digest(out/'reader_lock.json')))


def score(out):
    reg=load(out/'registration.json')
    spec=load(SPEC)
    parent=ROOT/spec['parent']
    if (reg['criteria']!=spec or any(digest(ROOT/p)!=sha for p,sha in reg['sources'].items())
            or digest(parent/'registration.json')!=spec['parent_registration_sha256']):
        raise ValueError('acquisition or parent lock mismatch')
    lock=dict(protocol=spec,registration=digest(out/'registration.json'),
              diagnostic='After sealing, count the four expected ink sectors within 2.5 px of each actual target station. Count false PRESENT and missing PRESENT central halves separately. Not an input to either reader or a gate threshold change.',
              sources={p.relative_to(ROOT).as_posix():digest(p) for p in SOURCES})
    write(out/'reader_lock.json',lock)
    tests=controls()
    write(out/'synthetic.json',tests)
    cal=load(out/'a/calibration.json')
    if (cal!=load(parent/'a/calibration.json') or (cal['W'],cal['H'])!=(1280,960)
            or not np.isclose(cal['f'],640/np.tan(np.deg2rad(35))) or not cal['ray_marker_fit']):
        raise ValueError('wrong optics')
    seal(out,reg,cal)
    sealed=load(out/'predictions.json')
    rows=[]
    physical=[]
    wrong_stations=[]
    wrong_local=[]
    wrong_faces=[]
    orphan_witness=None
    allwork=defaultdict(list)
    used=set()
    for run in spec['runs']:
        case=run['name']
        folder,old=out/case,parent/case
        o,p=load(folder/'oracle.json'),load(old/'oracle.json')
        if (reg['parent_oracles'][case]!=digest(old/'oracle.json') or len(o['frames'])!=len(p['frames'])
                or o['calibration_sha256']!=digest(folder/'calibration.json')
                or o['registration_sha256']!=digest(out/'registration.json') or o['optics']!=spec['optics']
                or o['config']['sheet']!=spec['sheet'] or o['config']['cut']!=p['config']['cut']
                or o['final_state']!='DONE' or o.get('error') or o.get('cleanup_error')
                or o['roles']!=p['roles'] or not all(v['ok'] for v in o['roles'].values())
                or not o['closed'] or not o['at_ready'] or o['initial_view']!=p['initial_view']
                or (case!='pull' and (o['look']['residual_m']>.0005 or o['look']['distance_to_ready_m']<=.0005))
                or (case=='pull' and o['pull_motion_m']!=p['pull_motion_m'])):
            raise ValueError(f'{case}: unqualified or unmatched run')
        R,t=np.asarray(o['initial_view']['applied_R']),np.asarray(o['initial_view']['applied_t'])
        for fr,ref in zip(o['frames'],p['frames'],strict=True):
            key=f"{case}/hd/{fr['key']}.npz"
            sha=fr['hd_sha256']
            if key in used or sealed['files'][key]!=sha:
                raise ValueError('duplicate/mismatched exposure')
            used.add(key)
            sp=folder/'structure'/(fr['key']+'.npz')
            rp=old/'structure'/(ref['key']+'.npz')
            hp=old/'hd'/(ref['key']+'.npz')
            if (digest(sp)!=fr['structure_sha256'] or digest(rp)!=ref['structure_sha256']
                    or digest(hp)!=ref['hd_sha256']):
                raise ValueError('corrupt scoring files')
            with np.load(sp) as s,np.load(rp) as sr:
                same_mesh=all(np.array_equal(s[k],sr[k]) for k in ('nodes','edges','faces'))
                state=material_state(s['edges'])
                alive=facets(s['edges'])
                xyz=(s['nodes']-t) @ R
                if not np.array_equal(state['faces'],s['faces']):
                    raise ValueError('structural face mismatch')
            with np.load(out/key) as hd,np.load(hp) as od:
                d1,d0=hd['depth'],od['depth']
                material_ink=decode(hd['rgb'])
                same_depth=np.array_equal(d1,d0)
                changed=int(np.count_nonzero(d1!=d0))
                max_change=float(np.max(abs(d1-d0))*1000)
            mm,phase=state['prefix_rows'],fr['phase']
            matched=(same_mesh and same_depth and mm==fr['prefix_mm'] and state['farthest_row_end']==fr['farthest_mm']
                     and fr['farthest_mm']<=10 and not fr['obs']['torn'] and not fr['obs']['faults'] and fr['closed']
                     and all(fr[k]==ref[k] for k in ('phase','bites','direct_cuts','releases','prefix_mm','farthest_mm',
                                                      'tick','tcp','closed','obs')))
            if not matched:
                physical.append(f'{case}/{phase}/{mm}')
            if mm==8 and phase=='look':
                def count_incident(row,col):
                    return sum(row*24+col in tri for tri,_ in alive.values())
                orphan_witness=dict(case=case,image_sha256=sha,row=7,old_center_column=11,
                                    old_center_incident_faces=count_incident(7,11),
                                    retained_incident_faces={str(k):count_incident(7,k) for k in (9,14)},
                                    matches_parent=matched)
            tcp=np.asarray(fr['tcp'])
            witnesses=[]
            target=[(j,k) for j in (mm-1,mm) for k in (9,14)] if mm else []
            near_ink=[]
            for j,k in target:
                q=xyz[j,k]
                uv=np.array([cal['W'],cal['H']])/2-cal['f']*q[:2]/q[2]
                x,y=np.floor(uv).astype(int)
                x0,y0=max(0,x-3),max(0,y-3)
                patch=material_ink[y0:y+4,x0:x+4]
                yy,xx=np.indices(patch.shape)
                mask=np.hypot(xx+x0+.5-uv[0],yy+y0+.5-uv[1])<=2.5
                lanes=(0,1) if k==9 else (6,7)
                counts=[int(np.count_nonzero(mask&(patch==8*t+l))) for t in (j-1,j) for l in lanes]
                near_ink.append(dict(row=j,column=k,counts=counts))
            for i,val in enumerate(sealed['images'][sha]):
                obs,sc=val['reference'],val['tool']
                ids={}
                for s in obs['stations']:
                    j,k=s['row'],s['column']
                    if (j,k) in ids or not (1<=j<=12 and k in (9,14)):
                        raise ValueError('unregistered or repeated station')
                    # At a bottom-left or top-right vertex either incident triangle
                    # of that sector suffices; the other two sectors each have one.
                    node=j*24+k
                    sectors={(r,c) for (r,c,h),(tri,_) in alive.items() if node in tri}
                    supported=all((r,c) in sectors for r in (j-1,j) for c in (k-1,k))
                    err=float(1000*np.linalg.norm(s['camera_m']-xyz[j,k]))
                    ids[j,k]=(err,supported)
                    if not supported or not err<.5:
                        wrong_stations.append([case,phase,mm,i,j,k,err,supported])
                for s in obs['central_faces']:
                    if not state['faces'][s['row']][s['half']]:
                        wrong_faces.append([case,phase,mm,i,s['row'],s['half']])
                local=obs['local_mm']
                if local is not None and (not mm or abs(local-mm)>=.5):
                    wrong_local.append([case,phase,mm,i,local])
                if obs['prefix_mm'] is not None or obs['motion_authorized'] or obs['front_xyz'] is not None:
                    raise ValueError('unregistered inference about centre, prefix or motion')
                tip=float(1000*np.linalg.norm(sc['tip']-(tcp[:,3]-t)@R)) if sc['valid'] else None
                jaw=angle(sc['jaw_axis'],tcp[:,2]@R) if sc['valid'] else None
                tool_ok=bool(sc['valid'] and tip<=1.5 and jaw<=5)
                good=lambda k:k in ids and ids[k][0]<.5 and ids[k][1]
                target_ok=bool(target and all(good(k) for k in target))
                witnesses.append(dict(stations=len(ids),max_station_error_mm=max((e for e,_ in ids.values()),default=None),
                                      target=[dict(row=j,column=k,error_mm=ids[j,k][0] if (j,k) in ids else None,
                                                   ok=good((j,k))) for j,k in target],
                                      target_ok=target_ok,tool_ok=tool_ok,tip_mm=tip,jaw_deg=jaw,
                                      first_present=obs.get('first_present_row'),local_mm=local,
                                      detected_halves=len(obs['central_faces']),
                                      missing_live_halves=int(np.sum(state['faces']))-sum(
                                          bool(state['faces'][s['row']][s['half']]) for s in obs['central_faces']),
                                      local_ok=tool_ok and ((not mm and obs.get('first_present_row')==0 and local is None)
                                                          or (mm>0 and local==mm and target_ok))))
            if phase!='park':
                allwork[case,mm,phase if not mm else fr['bites']].append(all(v['local_ok'] for v in witnesses))
            rows.append(dict(case=case,phase=phase,mm=mm,sha256=sha,matched=matched,same_depth=same_depth,
                             changed_depth_pixels=changed,max_depth_change_mm=max_change,same_mesh=same_mesh,
                             scorer_only_target_ink_neighbourhoods=near_ink,
                             variants=witnesses))
    if used!=set(sealed['files']) or len(used)!=20:
        raise ValueError('incomplete scoring join')
    paired=[r for r in rows if r['phase']=='look' and r['mm'] in (8,9)]
    if sorted(r['mm'] for r in paired)!=[8,9]:
        raise ValueError('missing 8/9 look')
    partial=all(v['target_ok'] and v['tool_ok'] for r in paired for v in r['variants'])
    paired_local=all(v['local_ok'] for r in paired for v in r['variants'])
    orphan_ok=bool(orphan_witness and orphan_witness['matches_parent'] and not orphan_witness['old_center_incident_faces']
                   and all(orphan_witness['retained_incident_faces'].values()))
    clean=orphan_ok and not(physical or wrong_stations or wrong_local or wrong_faces) and all(
        v['passed'] for c in tests for v in c['variants'])
    result=dict(protocol=spec,reader_lock=digest(out/'reader_lock.json'),predictions_sha256=digest(out/'predictions.json'),
                synthetic_sha256=digest(out/'synthetic.json'),physical_mismatch=physical,wrong_stations=wrong_stations,
                wrong_local=wrong_local,wrong_faces=wrong_faces,rows=rows,paired_station_observed=partial,
                paired_local_observed=paired_local,landmark_gate_passed=clean and partial,
                failed_local_lengths=sorted({mm for (_,mm,_),v in allwork.items() if not any(v)}),
                orphan_witness=orphan_witness,measured_orphan_reproduced=orphan_ok,
                passed=clean and partial and all(any(v) for v in allwork.values()),
                prefix_proved=False,control_enabled=False,confirmation_used=False)
    write(out/'decision.json',result)
    print('indexed reference:',result['passed'],'8/9 station',result['landmark_gate_passed'],
          '8/9 local',paired_local,'failed local lengths',result['failed_local_lengths'])
    return result['passed']


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    raise SystemExit(0 if score(ap.parse_args().out) else 1)
