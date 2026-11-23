"""Seal CutPerception predictions before evaluating the registered existing HD development set."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/k) for k in ('vision', 'task')]
from cut_perception import CutPerception
from seam_topology import material_state

SPEC = ROOT/'config/task/m5_bilateral_3d_dev_v1.json'
SOURCES = [Path(__file__), SPEC, *(ROOT/'scripts/vision'/k for k in
           ('bilateral_surface.py','cut_perception.py','material_coordinate.py','ruler_readout.py')),
           *(ROOT/'scripts/task'/k for k in ('ground_truth.py','workspace.py','seam_topology.py'))]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    return json.loads(p.read_bytes())


def write(p, obj):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def variants(z):
    yield z
    seed = int(hashlib.sha256(z.tobytes()).hexdigest()[:16], 16)
    yield z + np.random.default_rng(seed).normal(0, .0005, z.shape)


def predict_image(im, z, cal):
    vals = []
    for depth in variants(z):
        e = CutPerception(**cal, bilateral_surface=True).estimate(im,depth,np.eye(3),np.zeros(3))
        vals.append(dict(cut=e['cut'],tool=e['scissors']))
    return vals


def fixture(n, tilt, fault=''):
    """Independent perspective renderer: same immutable side/face print, no pixel classifier logic."""
    w,h,f = 260,300,1800.
    yy,xx = np.indices((h,w))
    rays = np.stack([(w/2-xx-.5)/f,(h/2-yy-.5)/f,np.ones((h,w))],axis=-1)
    a = np.deg2rad(tilt)
    u = np.array([1.,0,0])
    v = np.array([0,np.cos(a),np.sin(a)])
    normal = np.cross(u,v)
    center = np.array([0.,0.,.11])
    z0 = (normal @ center)/(rays @ normal)
    surf = rays*z0[...,None]-center
    x,y = surf @ u*1000, surf @ v*1000+6.5
    im = np.zeros((h,w,3),np.uint8)
    z = np.full((h,w),.18)
    for j in range(13):
        row = (y>=j) & (y<j+1) & (abs(x)<4.5)
        t = y-j
        # The two triangles meet on the diagonal from (-.5,j) to (+.5,j+1).
        for s in (0,1):
            right = (x+.5 >= t)
            central = abs(x)<.5
            live = (j>=n or (fault=='bridge' and j==4))
            if fault == f'half{s}' and j==8:
                live = True
            use = row & ((x>=.5 if s else x<=-.5) | (central & (right==bool(s)) & live))
            red,k = 32+48*(j%4), 35+46*(j//4)
            im[use] = (red,k,235) if s else (red,235,k)
            z[use] = z0[use]
    front = center+(n-6.5)*.001*v
    if fault in ('foreground','invalid','hidden'):
        target = (abs(x)<(.5 if fault!='hidden' else 4.6)) & (y>=8) & (y<10)
        if fault!='invalid':
            im[target] = 105
        z[target] = np.nan if fault=='invalid' else .025
    if fault=='shutter':
        im[:]=100
        z[:]=.025
    cal=dict(W=w,H=h,f=f,far=.3,rcm_world=[0,0,0])
    return im,z,cal,front,v,-normal


def angle(a, b):
    a,b=map(np.asarray,(a,b))
    return float(np.degrees(np.arccos(np.clip((a @ b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))


def compare(c, mm, front, tangent, normal):
    if not c['front_valid']:
        return None
    err = dict(material_error_mm=abs(c['progress_m']*1000-mm),
               front_error_mm=float(1000*np.linalg.norm(c['front']-front)),
               tangent_error_deg=angle(c['direction'],tangent),normal_error_deg=angle(c['normal'],normal))
    mm_est = c['progress_m']*1000
    err['geometry_ok'] = bool(err['material_error_mm'] < .5 and err['front_error_mm'] < .5
                              and max(err['tangent_error_deg'],err['normal_error_deg']) <= 5.
                              and ((mm < 9 and mm_est < 9) or (9 <= mm <= 10 and 9 <= mm_est <= 10))
                              and ((mm==0)==(c['mode']=='bilateral_entry')))
    return err


def synthetic():
    results = []
    for tilt in (0,32):
        for n,mode,expect in ([(k,'',k if k<=10 else None) for k in (0,3,6,8,9,10,11)]
                              + [(9,k,None) for k in ('bridge','invalid','foreground','hidden','shutter')]
                              + [(9,k,8) for k in ('half0','half1')]):
            im,z,cal,point,v,normal = fixture(n,tilt,mode)
            values = []
            for pred in predict_image(im,z,cal):
                c=pred['cut']
                metric = compare(c,expect,point+(expect-n)*.001*v,v,normal) if expect is not None else None
                if expect is None:
                    ok = not c['front_valid']
                elif mode.startswith('half') and not c['front_valid']:
                    ok = True
                else:
                    ok = bool(metric and metric['geometry_ok'])
                values.append(dict(passed=ok,cut=c,errors=metric))
            results.append(dict(nominal=n,tilt=tilt,case=mode or 'plain',variants=values))
    return results


def register(out):
    out.mkdir(parents=True, exist_ok=False)
    spec=load(SPEC)
    src=ROOT/spec['source']
    if digest(src/'registration.json')!=spec['source_registration_sha256'] or digest(src/'decision.json')!=spec['source_decision_sha256']:
        raise ValueError('unknown source run')
    hashes = {}
    for case,num in [('a',6),('b',6),('c',6),('pull',2)]:
        folder=src/case
        paths=sorted((folder/'hd').glob('*.npz'))
        if len(paths)!=num or digest(folder/'calibration.json')!=spec['calibration_sha256']:
            raise ValueError('wrong exposure count or calibration')
        hashes.update({p.relative_to(src).as_posix():digest(p) for p in paths})
    write(out/'registration.json',dict(criteria=spec,exposures=hashes,
          sources={p.relative_to(ROOT).as_posix():digest(p) for p in SOURCES}))


def predict(out):
    reg=load(out/'registration.json')
    src=ROOT/reg['criteria']['source']
    cal=load(src/'a/calibration.json')
    if (cal['W'],cal['H'])!=(1280,960) or not np.isclose(cal['f'],640/np.tan(np.deg2rad(35))):
        raise ValueError('unexpected calibration')
    pred={}
    for rel,sha in reg['exposures'].items():
        p=src/rel
        if digest(p)!=sha:
            raise ValueError('exposure changed')
        if sha in pred:
            continue
        with np.load(p,allow_pickle=False) as d:
            if set(d.files)!={'rgb','depth'}:
                raise ValueError('not an RGB-D-only exposure')
            im,z=d['rgb'],d['depth']
        if im.shape!=(960,1280,3) or z.shape!=im.shape[:2] or im.dtype!=np.uint8:
            raise ValueError('not actual native HD')
        pred[sha]=predict_image(im,z,cal)
    write(out/'predictions.json', dict(registration_sha256=digest(out/'registration.json'),images=pred))


def score(out):
    reg=load(out/'registration.json')
    spec=reg['criteria']
    src=ROOT/spec['source']
    artifact=load(out/'predictions.json')
    if (any(digest(ROOT/k)!=v for k,v in reg['sources'].items())
            or artifact['registration_sha256']!=digest(out/'registration.json')):
        raise ValueError('source or registration changed')
    images=artifact['images']
    used, rows, issues=set(),[],[]
    gates={}
    controls=load(out/'synthetic.json')
    if not all(v['passed'] for c in controls for v in c['variants']):
        issues.append('synthetic requirements not met')
    for case,expected_hash in spec['oracles'].items():
        folder=src/case
        if digest(folder/'oracle.json')!=expected_hash:
            raise ValueError('changed mechanical source')
        o=load(folder/'oracle.json')
        if (o['final_state']!='DONE' or o.get('error') or o.get('cleanup_error') or not o['closed']
                or o['registration_sha256']!=spec['source_registration_sha256']
                or o['calibration_sha256']!=spec['calibration_sha256']
                or 'CAMERA_ADJUST' in o['visited'] or not np.allclose(o['initial_view']['target'],[4.5,0,.08])
                or (case!='pull' and (not o['at_ready'] or o['look']['residual_m']>.0005
                                     or o['look']['distance_to_ready_m']<=.0005))):
            raise ValueError('unqualified development reference')
        R,t = map(np.array,(o['initial_view']['applied_R'],o['initial_view']['applied_t']))
        for fr in o['frames']:
            rel=f"{case}/hd/{fr['key']}.npz"
            sha=fr['hd_sha256']
            sp=folder/'structure'/(fr['key']+'.npz')
            if (reg['exposures'][rel]!=sha or rel in used or digest(sp)!=fr['structure_sha256']
                    or fr['obs']['torn'] or fr['obs']['faults'] or not fr['closed']):
                raise ValueError('sensor join or physical guard failed')
            used.add(rel)
            with np.load(sp,allow_pickle=False) as st:
                state=material_state(st['edges'])
                xyz=(st['nodes']-t) @ R
                if not np.array_equal(state['faces'],st['faces']):
                    raise ValueError('inconsistent faces')
            mm=state['prefix_rows']
            if (mm!=fr['prefix_mm'] or state['farthest_row_end']!=fr['farthest_mm']
                    or state['farthest_row_end']>10):
                raise ValueError('structural label mismatch')
            m=xyz[:,11:13].mean(1)
            direct=m[mm+3]-m[mm]
            normal=np.cross(direct,xyz[mm,12]-xyz[mm,11])
            if normal @ m[mm]>0:
                normal=-normal
            tcp=np.array(fr['tcp'])
            p=(tcp[:,3]-t) @ R
            jaw=tcp[:,2] @ R
            values=[]
            for val in images[sha]:
                c,sc=val['cut'],val['tool']
                met=compare(c,mm,m[mm],direct,normal)
                t_err=1000*float(np.linalg.norm(sc['tip']-p)) if sc['valid'] else None
                j_err=angle(sc['jaw_axis'],jaw) if sc['valid'] else None
                tool_ok=bool(sc['valid'] and t_err<=1.5 and j_err<=5)
                guard=c.get('sweep_limit_from_front_m')
                excess=None
                if met:
                    if guard is not None:
                        ref=float(np.min((xyz[10,11:13]-np.array(c['front'])) @ c['direction']))
                        excess=(guard-ref)*1000
                    if not met['geometry_ok'] or (excess is not None and excess>1e-8):
                        issues.append(f'{case}/{fr["phase"]}@{mm}: incorrect geometry or optimistic guard')
                ok=bool(met and met['geometry_ok'] and tool_ok and excess is not None and excess<=1e-8)
                values.append(dict(progress_mm=None if not met else c['progress_m']*1000,
                                   reason=c['reason'],tool_valid=sc['valid'],tool_ok=tool_ok,
                                   tip_error_mm=t_err,jaw_error_deg=j_err,errors=met,
                                   guard_excess_mm=excess,joint_ok=ok))
            phase=fr['phase']
            if phase!='park':
                key=(case,'zero_'+phase if not mm else f'bite{fr["bites"]}',mm)
                gates[key]=gates.get(key,False) or all(v['joint_ok'] for v in values)
            rows.append(dict(case=case,phase=phase,mm=mm,sha256=sha,variants=values))
    if used!=set(reg['exposures']) or set(images)!=set(reg['exposures'].values()):
        raise ValueError('incomplete blind join')
    failures=sorted({mm for (_,_,mm),ok in gates.items() if not ok})
    issues.extend(f'{c}/{b}: no qualifying exposure at {mm} mm' for (c,b,mm),ok in gates.items() if not ok)
    if {mm for _,_,mm in gates}!={0,3,6,8,9,10}:
        raise ValueError('missing length')
    result=dict(criteria=spec,registration_sha256=digest(out/'registration.json'),
                predictions_sha256=digest(out/'predictions.json'),synthetic_sha256=digest(out/'synthetic.json'),
                failed_lengths_mm=failures,issues=issues,passed=not issues,rows=rows,
                control_enabled=False,confirmation_used=False)
    write(out/'decision.json',result)
    print('bilateral development:',result['passed'],'failed lengths',failures,flush=True)
    return result['passed']


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    register(args.out)
    write(args.out/'synthetic.json',synthetic())
    predict(args.out)
    raise SystemExit(0 if score(args.out) else 1)
