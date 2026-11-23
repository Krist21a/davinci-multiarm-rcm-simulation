"""Original right bank and displaced left bank; same strict fully-intact preflight."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import capture_seam_rail as acquisition
from capture_surface_seam import ROOT, add_arguments, sha256, write_new
from screen_cell_settle import OUT as SETTLE, asymmetric_source
from score_cell_camera import blind
import score_surface_seam as parent
from seam_slide_world import SeamSlideWorld
import wide_cell_pilot as wide

SPEC=ROOT/'config/task/seam_asymmetric_cell_v1.json'
OUT=ROOT/'validation/task/seam_asymmetric_cell_v1'
SOURCES=(*wide.SOURCES,'scripts/analysis/asymmetric_cell_pilot.py','scripts/analysis/screen_cell_settle.py',
         'config/task/seam_wide_cell_v1.json')


class World(SeamSlideWorld):
    def start(self):
        with patch('seam_slide_world.staged_source',asymmetric_source):
            super().start()


def evaluate(folder,stage,contract,hashes):
    pre=stage=='preflight'
    entries=contract['development_runs'][:1] if pre else contract['development_runs']
    predictions={}
    for e in entries:
        for path in sorted((folder/'asym'/e['id']/'high').glob('*.npz')):
            if path.stem in predictions:
                raise ValueError('duplicate optical frame')
            predictions[path.stem]=blind(path)
    predfile=folder/(stage+'_predictions.json')
    write_new(predfile,dict(criteria=contract,spec_sha256=sha256(SPEC),hashes=hashes,predictions=predictions))
    reports,seen=[],set()
    hold_poses=[24*j+i for j in range(13) for i in (*range(6,11),*range(13,18))]
    for e in entries:
        folder_e=folder/'asym'/e['id']
        o=json.loads((folder_e/'oracle.json').read_bytes())
        manifest={}
        for f in o['frames']:
            k=f['key']
            v=predictions[k]
            if not f['restored'] or not np.isclose(f['zoom_angle'],np.deg2rad(30.),rtol=0,atol=1e-6) or v['sensor_sha256']!=f['high_sha256']:
                raise ValueError('unverified optical capture')
            manifest[k]=dict(sensor_sha256=f['sensor_sha256'],native=v['predictions'][0],noisy=v['predictions'][1])
        with patch.multiple(parent,PLAN=SPEC,OUTPUT=folder/'asym'):
            q=parent.run_score(e,manifest,seen,contract,hashes)
        if o.get('pins')!=hold_poses or not o.get('roles') or not all(v['ok'] for v in o['roles'].values()):
            q['issues'].append('guides or roles not verified')
        for f in q['rows']:
            v=predictions[f['key']]
            with np.load(folder_e/'structure'/(f['key']+'.npz')) as data:
                far=data['nodes'][-1]
            span=far[-1]-far[0]
            axis=span/np.linalg.norm(span)
            centre=(far[0]+far[-1])/2
            offsets=(far[[11,12]]-centre)@axis
            if not np.allclose([np.linalg.norm(span),*offsets],[.023,-.0025,.0005],rtol=0,atol=1e-8):
                q['issues'].append('wrong specimen dimensions')
            f['missing_corner_patches']=[k for k,c in enumerate(v['corners']) if not c['patch']]
            f['missing_face_patches']=[k for k,c in enumerate(v['faces']) if not c['patch']]
            if f['phase'] in ('loaded','pulled'):
                f['optical_ok'] &= all(x.get('states')==[['PRESENT','PRESENT']]*13 for x in v['predictions'])
            elif f['phase']=='work' and f['actual_mm']>=9:
                f['optical_ok'] &= f['decisions']==['STOP']*2
        q['mechanics_ok']=not q['issues']
        reports.append(q)
    if seen!=set(predictions) or len({q['run_id'] for q in reports})!=len(reports):
        raise ValueError('unjoined or duplicate records')
    ok=all(q['mechanics_ok'] and all(f['optical_ok'] for f in q['rows']) for q in reports)
    if not pre:
        cover={n:{q['case'] for q in reports for f in q['rows'] if f['phase']=='work' and f['actual_mm']==n} for n in (8,9,10)}
        ok &= bool(cover[10]) and any(x!=y for x in cover[8] for y in cover[9])
    write_new(folder/('selection.json' if pre else 'decision.json'),dict(criteria=contract,spec_sha256=sha256(SPEC),
              source_sha256=hashes,passed=ok,selected=dict(rail='asym',scale=4) if ok else None,
              reports=reports,predictions_sha256=sha256(predfile),vision=False))
    print('passed',ok,[(q['case'],q['mechanics_ok'],q['issues'],[(f['phase'],f['missing_corner_patches'],
              f['missing_face_patches']) for f in q['rows']]) for q in reports])
    return int(not ok)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--stage',choices=('collect','preflight','work'),required=True)
    ap.add_argument('--case',choices=('pull','a','b','c','d','e'))
    ap.add_argument('--out-root',type=Path,default=OUT)
    args=ap.parse_args()
    # The immutable capture harness reads its protocol from disk; provide a separate, complete
    # registration for this geometry. Never patch the archived configuration on disk.
    asym=json.loads(SPEC.read_bytes())
    inherited=ROOT/asym['inherits']
    spec=json.loads(inherited.read_bytes())
    contract={**spec,**asym}
    hashes={p:sha256(ROOT/p) for p in SOURCES}
    screen=json.loads((SETTLE/'decision.json').read_bytes())
    if (not screen['asymmetric_eligible_only_for_live_preflight'] or hashes['scripts/vision/cell_certificate_v1.py']!=spec['certificate_sha256']
            or hashes['src/tissue/cell_certificate_print.lua']!=spec['print_sha256']):
        raise ValueError('registration mismatch')
    lock=args.out_root/'registration.json'
    if args.stage=='collect' and args.case=='pull' and not lock.exists():
        args.out_root.mkdir(parents=True,exist_ok=True)
        write_new(lock,dict(**contract,asym_spec_sha256=sha256(SPEC),parent_spec_sha256=sha256(inherited),
                            settle_sha256=sha256(SETTLE/'decision.json')))
    locked=json.loads(lock.read_bytes())
    if (any(locked.get(k)!=v for k,v in contract.items()) or locked['asym_spec_sha256']!=sha256(SPEC)
            or locked['parent_spec_sha256']!=sha256(inherited) or locked['settle_sha256']!=sha256(SETTLE/'decision.json')):
        raise ValueError('physical registration changed')
    SPEC=lock
    if args.stage=='collect':
        if not args.case:
            ap.error('collection needs --case')
        args.rail='asym'
        with patch.multiple(acquisition,PLAN=lock,SOURCES=SOURCES,SeamRailWorld=World,RailCapture=wide.CaptureWide):
            raise SystemExit(acquisition.acquire(args))
    raise SystemExit(evaluate(args.out_root,args.stage,locked,hashes))
