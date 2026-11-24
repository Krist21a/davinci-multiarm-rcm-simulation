"""Collect just the missing development negative; distinguish intact hosts from visible ink."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from capture_surface_seam import ROOT, add_arguments, sha256, write_new
import rim_zoom_pilot as zoom
from score_cell_camera import hosts
from score_cell_certificate import ceiling

SPEC=ROOT/'config/task/seam_rim_near_v1.json'
OUT=ROOT/'validation/task/seam_rim_near_v1'


def analyze(folder):
    passed=json.loads((zoom.OUT/'selection.json').read_bytes())
    prev=json.loads((zoom.OUT/'decision.json').read_bytes())
    if not passed['passed'] or prev['passed']:
        raise ValueError('not the registered failed development study')
    orig_pred=json.loads((zoom.OUT/'work_predictions.json').read_bytes())['predictions']
    if sha256(zoom.OUT/'work_predictions.json')!=prev['predictions_sha256']:
        raise ValueError('prior predictions changed')
    spec=json.loads((folder/'registration.json').read_bytes())
    source=folder/'asym/near'
    nearpred={p.stem:zoom.read20(p) for p in sorted((source/'high').glob('*.npz'))}
    write_new(folder/'blind.json',dict(spec_sha256=sha256(folder/'registration.json'),predictions=nearpred))
    oracle=json.loads((source/'oracle.json').read_bytes())
    manifest={}
    for f in oracle['frames']:
        p=nearpred[f['key']]
        if not f['restored'] or not np.isclose(f['zoom_angle'],np.deg2rad(20.),rtol=0,atol=1e-6) or f['high_sha256']!=p['sensor_sha256']:
            raise ValueError('unverified near capture')
        manifest[f['key']]=dict(sensor_sha256=f['sensor_sha256'],native=p['predictions'][0],noisy=p['predictions'][1])
    seen=set()
    with patch.multiple(zoom.PREV.asym.parent,PLAN=folder/'registration.json',OUTPUT=folder/'asym'):
        new=zoom.PREV.asym.parent.run_score(spec['entry'],manifest,seen,spec,{p:sha256(ROOT/p) for p in SOURCES})
    if set(nearpred)!=seen:
        raise ValueError('incomplete manifests')
    joined=[]
    for src,report,pred in [(source,new,nearpred)]+[(zoom.OUT/'asym'/r['case'],r,orig_pred) for r in prev['reports'] if r['case'] in ('b','c')]:
        o=json.loads((src/'oracle.json').read_bytes())
        if sha256(src/'oracle.json')!=report['oracle_sha256'] or 'CAMERA_ADJUST' in o['visited']:
            raise ValueError('unmatched mechanical evidence')
        for f in o['frames']:
            if f['phase'] not in ('work','park') or f['prefix_mm']<7:
                continue
            hi,st=[src/k/(f['key']+'.npz') for k in ('high','structure')]
            v=pred[f['key']]
            if not sha256(hi)==v['sensor_sha256']==f['high_sha256'] or sha256(st)!=f['structure_sha256']:
                raise ValueError('sensor or structure drift')
            with np.load(hi) as im,np.load(st) as data:
                count=hosts(data['edges'])
                xyz=(data['nodes'].reshape(-1,3)-o['view']['applied_t'])@np.array(o['view']['applied_R'])
                ideal=ceiling(im['depth'],xyz,data['faces'],1280/np.tan(np.deg2rad(10)),radius=0,sigma=0)
                joined.append(dict(case=report['case'],phase=f['phase'],mm=f['prefix_mm'],mechanical=report['mechanics_ok'],
                    sensor_sha256=sha256(hi),struct_sha256=sha256(st),target_cell8_faces=data['faces'][8],
                    hostless_corner_ids=[k for k,num in enumerate(count) if num==0],
                    not_visible_corner_ids=[k for k,c in enumerate(v['corners']) if not c['patch']],
                    predictions=[p['decision'] for p in v['predictions']],ideal_geometry_only=ideal['decision'],
                    ideal_zero_block_rows=[j for j,row in enumerate(ideal['rows']) if any(f['state']=='UNKNOWN' for f in row)]))
    write_new(folder/'decision.json',dict(criteria=spec,old_development_sha256=sha256(zoom.OUT/'decision.json'),near=new,
              blind_sha256=sha256(folder/'blind.json'),comparisons=joined,
              has_safe_work_8=any(r['phase']=='work' and r['mm']==8 and r['mechanical'] for r in joined),
              visual_control=False,confirmation=False))
    for r in joined:
        print(r['case'],r['phase'],r['mm'],r['predictions'],'no host',r['hostless_corner_ids'],'no patch',r['not_visible_corner_ids'],
              'ideal',r['ideal_geometry_only'],r['ideal_zero_block_rows'])


SOURCES=(*zoom.PREV.SOURCES,'scripts/analysis/planar_rim_pilot.py','config/task/seam_asymmetric_rest_v1.json',
         'scripts/analysis/rim_zoom_pilot.py','scripts/analysis/rim_near_control.py')


def main(args):
    if args.stage=='score':
        analyze(args.out_root)
        return 0
    base=zoom.OUT/'registration.json'
    selection=json.loads((zoom.OUT/'selection.json').read_bytes())
    original=json.loads(base.read_bytes())
    spec=json.loads(SPEC.read_bytes())
    if not selection['passed'] or selection['spec_sha256']!=sha256(base) or selection['source_sha256']!={p:sha256(ROOT/p) for p in SOURCES[:-1]}:
        raise ValueError('original preflight no longer qualifies')
    folder=args.out_root
    folder.mkdir(parents=True,exist_ok=False)
    lock=folder/'registration.json'
    write_new(lock,{**original,**spec,'development_runs':[spec['entry']],'base_registration_sha256':sha256(base),
                    'frozen_failed_gate_sha256':sha256(zoom.OUT/'decision.json')})
    write_new(folder/'selection.json',dict(scope='reused previous validated preflight for identical setup',
              original_selection_sha256=sha256(zoom.OUT/'selection.json'),selected=selection['selected'],
              source_sha256={p:sha256(ROOT/p) for p in SOURCES},spec_sha256=sha256(lock)))
    args.case,args.rail='near','asym'
    with patch.object(zoom.PREV.asym,'asymmetric_source',zoom.rim.staged),patch.multiple(zoom.PREV.acquisition,
            PLAN=lock,SOURCES=SOURCES,SeamRailWorld=zoom.PREV.World,RailCapture=zoom.Capture20):
        return zoom.PREV.acquisition.acquire(args)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--stage',required=True,choices=('collect','score'))
    ap.add_argument('--out-root',type=Path,default=OUT)
    raise SystemExit(main(ap.parse_args()))
