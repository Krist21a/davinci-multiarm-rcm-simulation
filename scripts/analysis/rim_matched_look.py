"""Guarded single look-back from matched threshold states; privilege only at offline attribution."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from capture_surface_seam import ROOT, add_arguments, connect, sha256, stop_simulation, write_new
from cell_certificate_v1 import block, corners, decode, projected_interior
from rim_near_control import OUT as NEAR
import rim_zoom_pilot as zoom
from ruler_readout import inks

OUT=ROOT/'validation/task/seam_rim_look_v1'
SPEC=ROOT/'config/task/seam_rim_look_v1.json'


class Finished(Exception):
    pass


class Probe(zoom.Capture20):
    def save(self,phase,obs):
        if phase=='park':
            raise ValueError('checkpoint not reached before park')
        super().save(phase,obs)
        f=self.record['frames'][-1]
        if phase!='work' or f['tick']!=self.ref['tick']:
            return
        w=self.w
        if (not np.array_equal(f['tcp'],self.ref['tcp']) or self.bites!=self.ref['bites'] or not w.jaws_closed()
                or not np.array_equal(w.camera()[0],self.old_view['applied_R']) or not np.array_equal(w.camera()[1],self.old_view['applied_t'])):
            raise ValueError('unmatched work pose')
        for kind in ('sensor','high','structure'):
            original=self.ref_source/kind/(self.ref['key']+'.npz')
            if sha256(original)!=self.ref[kind+'_sha256']:
                raise ValueError('reference digest mismatch')
            with np.load(original) as before,np.load(self.folder/kind/(f['key']+'.npz')) as actual:
                if set(before.files)!=set(actual.files) or any(not np.array_equal(before[k],actual[k]) for k in before.files):
                    raise ValueError('unmatched work '+kind)
        self.record['matched']=True
        if not w.clear_view(.003):
            raise ValueError('guarded backoff refused')
        new=w.observe()
        if (new.torn or new.faults or not self.grasps_ok(new) or not self.tension_ok(new) or not w.jaws_closed()
                or not new.scissors_visible or np.linalg.norm(self.tcp(self.cutter)[:,3]-self.ready[:,3])<=.0005
                or len(w.call('tissueStats').get('cut_events') or [])!=self.bites or w.tick<=f['tick']):
            raise ValueError('not a valid post-backoff observation')
        super().save('backoff',new)
        g=self.record['frames'][-1]
        with np.load(self.folder/'structure'/(f['key']+'.npz')) as old,np.load(self.folder/'structure'/(g['key']+'.npz')) as latest:
            if any(not np.array_equal(old[k],latest[k]) for k in ('edges','faces')):
                raise ValueError('connectivity changed')
        self.record.update(qualified=True,actual_backoff_m=float(np.linalg.norm(
            np.asarray(g['tcp'])[:,3]-np.asarray(f['tcp'])[:,3])),camera=[v.tolist() for v in w.camera()])
        raise Finished()


def run(args):
    summary=json.loads((NEAR/'decision.json').read_bytes())
    src=(NEAR if args.case=='near' else zoom.OUT)/'asym'/args.case
    path=src/'oracle.json'
    row=next(x for x in summary['comparisons'] if x['phase']=='work' and x['case']==args.case)
    if not row['mechanical']:
        raise ValueError('invalid physical reference')
    data=json.loads(path.read_bytes())
    f=[f for f in data['frames'] if f['phase']=='work'][-1]
    if f['high_sha256']!=row['sensor_sha256'] or f['structure_sha256']!=row['struct_sha256']:
        raise ValueError('source join mismatch')
    folder=OUT/args.case
    folder.mkdir(parents=True,exist_ok=False)
    for d in ('sensor','high','structure'):
        (folder/d).mkdir()
    js=dict(criteria=json.loads(SPEC.read_bytes()),plan_sha256=sha256(SPEC),code_sha256=sha256(Path(__file__)),
            ref_sha256=sha256(path),frames=[],observations=[],qualified=False,matched=False)
    client=sim=None
    try:
        args.scene=ROOT/'scenes/davinci_si_articulated_verified.ttt'
        if sha256(args.scene)!=data['scene_sha256']:
            raise ValueError('scene drift')
        args.initial_view,args.initial_view_insertion_m=data['view']['target'][:2],data['view']['target'][2]
        client,sim=connect(args)
        with patch.object(zoom.PREV.asym,'asymmetric_source',zoom.rim.staged):
            w=zoom.PREV.World(client,sim,args,data['cfg'],mode='full')
        if w.controller_sha256!=data['controller_sha256']:
            raise ValueError('staged controller drift')
        task=Probe(w,data['cfg'],folder,js,False)
        task.old_view,task.ref,task.ref_source=data['view'],f,src
        try:
            task.run()
        except Finished:
            pass
    except Exception as exc:
        js['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            stop_simulation(client,sim)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
        write_new(folder/'probe.json',js)
    print(args.case,js.get('error') or js['qualified'])
    return int(not js['qualified'])


def factor(im,z,xyz,faces):
    focal=1280/np.tan(np.deg2rad(10))
    pp=np.array([1280,960])
    width=im.shape[1]
    boxes=[]
    for y in range(0,len(im),64):
        cap,face=decode(im[y:y+64])
        yy,xx=np.where((cap>=0)|(face>=0))
        if len(yy):
            boxes.append([xx.min(),yy.min()+y,xx.max(),yy.max()+y])
    if not boxes:
        return dict(missing=list(range(28)),evidence=[])
    a=np.asarray(boxes)
    lo=np.maximum(a[:,:2].min(0)-8,0)
    hi=np.minimum(a[:,2:].max(0)+9,[width,len(im)])
    roi=np.s_[lo[1]:hi[1],lo[0]:hi[0]]
    rgb,depth=im[roi],z[roi]
    cap,face=decode(rgb)
    yy,xx=np.indices(depth.shape)
    xy=np.stack((xx+.5,yy+.5),axis=-1)
    pp=pp-lo
    ns=[corners(cap==k,depth,xy,pp,focal,.00025) for k in range(28)]
    missing=[k for k,p in enumerate(ns) if p is None]
    blank=(cap<0)&(face<0)&(inks(rgb)<0)&np.isfinite(depth)
    pc=xyz[:,11:13].reshape(28,3)
    pix=pp-focal*pc[:,:2]/pc[:,2:]
    rows=[]
    for j in range(13):
        for h,ids in enumerate(([2*j,2*j+1,2*j+3],[2*j,2*j+3,2*j+2])):
            if faces[j,h]:
                continue
            variants={}
            nodes=[ns[k] for k in ids]
            seen=all(n is not None for n in nodes)
            for pos in ('O','I'):
                if pos=='O' and not seen:
                    continue
                c,r=(pix[ids],np.zeros((3,2))) if pos=='I' else tuple(np.array(v) for v in zip(*nodes))[:2]
                l=np.maximum(np.floor(c.min(0)).astype(int)-1,0)
                u=np.minimum(np.ceil(c.max(0)).astype(int)+2,depth.shape[::-1])
                sub=np.s_[l[1]:u[1],l[0]:u[0]]
                interior=projected_interior(c,r,xy[sub])
                for dep in ('O','I'):
                    if dep=='O' and not seen:
                        continue
                    zmax=max(pc[ids,2]) if dep=='I' else max(it[2][1] for it in nodes)
                    variants[pos+dep]=int(block(interior&blank[sub]&(depth[sub]>zmax+.00225)).sum())
            rows.append(dict(row=j,half=h,blocks=variants))
    return dict(missing=missing,evidence=rows)


def score():
    pred={}
    for path in sorted(OUT.glob('*/high/*.npz'),key=lambda p:p.stem):
        if path.stem in pred:
            raise ValueError('duplicate capture')
        pred[path.stem]=zoom.read20(path)
    write_new(OUT/'blind.json',dict(plan_sha256=sha256(SPEC),predictions=pred))
    cases=[]
    for name in ('near','b','c'):
        folder=OUT/name
        p=json.loads((folder/'probe.json').read_bytes())
        if not p['qualified']:
            cases.append(dict(case=name,qualified=False,error=p.get('error')))
            continue
        if p['plan_sha256']!=sha256(SPEC) or not p['matched']:
            raise ValueError('unverified look')
        f=p['frames'][-1]
        raw,s=folder/'high'/(f['key']+'.npz'),folder/'structure'/(f['key']+'.npz')
        prediction=pred[f['key']]
        if f['phase']!='backoff' or not sha256(raw)==prediction['sensor_sha256']==f['high_sha256'] or sha256(s)!=f['structure_sha256']:
            raise ValueError('invalid backoff record')
        with np.load(raw) as v,np.load(s) as data:
            R,t=map(np.array,p['camera'])
            grid=(data['nodes']-t)@R
            diagnostic=factor(v['rgb'],v['depth'],grid,data['faces'])
        cases.append(dict(case=name,qualified=True,mm=f['prefix_mm'],motion_mm=1000*p['actual_backoff_m'],
            decisions=[it['decision'] for it in prediction['predictions']],diagnostic=diagnostic,
            probe_sha256=sha256(folder/'probe.json')))
    write_new(OUT/'decision.json',dict(criteria=json.loads(SPEC.read_bytes()),cases=cases,blind_sha256=sha256(OUT/'blind.json'),
              confirmation=False,vision=False))
    for c in cases:
        print(c['case'],c.get('decisions'),c.get('error'),c.get('diagnostic',{}).get('missing'),
              {k:sum(x['blocks'].get(k,0)>0 for x in c.get('diagnostic',{}).get('evidence',[])) for k in ('OO','IO','OI','II')})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--stage',choices=('collect','score'),required=True)
    ap.add_argument('--case',choices=('near','b','c'))
    opts=ap.parse_args()
    if opts.stage=='collect':
        if not opts.case:
            ap.error('missing case')
        raise SystemExit(run(opts))
    score()
