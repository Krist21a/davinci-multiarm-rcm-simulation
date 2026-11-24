"""Consolidate fixed-case evidence and test for real through-depth samples in the work image."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from capture_surface_seam import ROOT, sha256, write_new
from ruler_readout import inks

TASK = ROOT/'validation/task'
DEV = TASK/'seam_slide_v2'
FACE = TASK/'seam_face_visibility_v1'
OUT = TASK/'seam_work_observability_v1'


def face_depth(verts, depth):
    """Pixel-centre, two-sided, perspective-correct depth of a material triangle at fixed nodes."""
    h,w = depth.shape
    f = .5*w/np.tan(np.deg2rad(35))
    if np.any(verts[:,2] <= 0):
        raise ValueError('unexpected triangle behind the camera')
    xy = np.array([w/2,h/2])-f*verts[:,:2]/verts[:,2:]
    basis = xy[1:]-xy[0]
    if abs(np.linalg.det(basis)) < 1e-10:
        return dict(samples=0, deep=0, foreground=0)
    lo = np.maximum(np.floor(xy.min(0)).astype(int), 0)
    hi = np.minimum(np.ceil(xy.max(0)).astype(int)+1, [w,h])
    y,x = np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
    b = (np.column_stack((x.ravel()+.5,y.ravel()+.5))-xy[0]) @ np.linalg.inv(basis)
    a = 1-b.sum(axis=1)
    keep = (a>=0) & (b>=0).all(axis=1)
    expected = 1/(a[keep]/verts[0,2]+b[keep]@(1/verts[1:,2]))
    delta = depth[y.ravel()[keep],x.ravel()[keep]]-expected
    return dict(samples=int(len(delta)), deep=int(np.sum(delta > .002)), foreground=int(np.sum(delta < -.0003)))


def picture(samples):
    img = Image.new('RGB',(900,520),'white')
    d = ImageDraw.Draw(img)
    d.text((8,4),'Real ECM RGB-D | the same closed working pose and one guarded 3 mm look-back',fill='black')
    for i,(name,phase,rgb,loc) in enumerate(samples):
        lab = inks(rgb)
        y,x=np.where(lab>=0)
        roi=(max(0,x.min()-8),max(0,y.min()-8),min(rgb.shape[1],x.max()+9),min(rgb.shape[0],y.max()+9))
        crop = Image.fromarray(rgb).crop(roi)
        k = min(440/crop.width,210/crop.height)
        w,h = int(crop.width*k),int(crop.height*k)
        col,row = (0 if phase=='work' else 1), (0 if name=='a' else 1)
        px,py = 450*col+6, 246*row+27
        d.text((px,py),f"{8+row} mm | {phase} | surviving face ink: {loc} connected px",fill='black')
        img.paste(crop.resize((w,h),Image.Resampling.NEAREST),(px,py+19))
    img.save(OUT/'work_and_backoff.png')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    registration = dict(scope='Post-hoc theoretical visibility of a new positive evidence rule. Never a control input.',
        criterion='On each independently severed cell, look for even one actual sensor pixel >2 mm behind each of '
                  'its two triangle projections at the actual nodes. Requiring an actual through-ray prevents inferring '
                  'a gap solely from missing dye. Grant the hypothetical reader exact material vertices; no erosion, '
                  'no tool masking, no assumed minimum area. A missing ray rules out this evidence source for this '
                  'frame only; not an impossibility result for other observers or motions. Fixed 70 degree sensor.',
        recipe_sha256=sha256(Path(__file__)))
    write_new(OUT/'criteria.json',registration)
    origin=json.loads((DEV/'decision.json').read_bytes())
    direct=json.loads((TASK/'seam_clear_observation_v1/decision.json').read_bytes())
    contact=json.loads((TASK/'seam_connected_evidence_v1/decision.json').read_bytes())
    views={}
    for suffix in ('run2','tail'):
        for name in ('a','b'):
            p=TASK/('seam_work_views_v1_'+suffix)/name/'views.json'
            v=json.loads(p.read_bytes())
            for f in v['views']:
                if name in views.get(str(f['index']),{}):
                    raise ValueError('duplicate camera candidate')
                views.setdefault(str(f['index']),{})[name]=dict(restored=f.get('restored',False),
                    dec=[x['decision'] for x in f.get('predictions',[])],source_sha256=sha256(p))
    if set(views)!=set(map(str,range(13))) or any(set(v)!=set(('a','b')) for v in views.values()):
        raise ValueError('camera records incomplete')
    summary, pictures = [],[]
    for name in ('a','b'):
        folder=DEV/'slide'/name
        o=json.loads((folder/'oracle.json').read_bytes())
        r=next(x for x in origin['runs'] if x['case']==name)
        f=[s for s in o['frames'] if s['phase']=='work'][-1]
        if (not r['mechanics_ok'] or r['oracle_sha256']!=sha256(folder/'oracle.json') or 'CAMERA_ADJUST' in o['visited']
                or not np.allclose(o['view']['target'],[4.5,0,.08],rtol=0,atol=1e-6)):
            raise ValueError('unverified match')
        token=f['key']+'.npz'
        high,struct=folder/'high'/token,folder/'structure'/token
        if sha256(high)!=f['high_sha256'] or sha256(struct)!=f['structure_sha256']:
            raise ValueError('source changed')
        probe_path=FACE/name/'probe.json'
        probe=json.loads(probe_path.read_bytes())
        if (not probe['complete'] or probe['ref_sha256']!=r['oracle_sha256'] or
                not all(e['restored'] for e in probe['exposures'])):
            raise ValueError('uncertified optical probe')
        with np.load(high) as sens,np.load(struct) as st:
            xyz=(st['nodes'].reshape(-1,3)-o['view']['applied_t'])@np.array(o['view']['applied_R'])
            rows=[]
            if sens['depth'].shape!=(1920,2560) or np.any(st['faces'][:f['prefix_mm']]):
                raise ValueError('wrong sensor scale or structural prefix')
            for j in range(f['prefix_mm']):
                a=24*j+11
                faces=[face_depth(xyz[ids],sens['depth']) for ids in ([a,a+1,a+25],[a,a+25,a+24])]
                rows.append(dict(row_index=j,faces=faces))
        for ph in ('work','backoff'):
            exp=next(e for e in probe['exposures'] if e['phase']==ph)
            image_path=high if ph=='work' else TASK/'seam_clear_observation_v1'/name/'high.npz'
            if sha256(image_path)!=exp['reference_sha256']:
                raise ValueError('phase mismatch')
            with np.load(image_path) as inp:
                pictures.append((name,ph,inp['rgb'],max(p['largest'] for p in exp['inks'])))
        roll_path=TASK/'seam_work_roll_v1'/name/'probe.json'
        roll=json.loads(roll_path.read_bytes())
        summary.append(dict(case=name,actual_mm=f['prefix_mm'],work_sha256=f['high_sha256'],
            rows=rows,zero_depth_face_rows=[r['row_index'] for r in rows if any(x['deep']==0 for x in r['faces'])],
            print_counterfactual=probe['exposures'],print_probe_sha256=sha256(probe_path),
            roll=dict(matched=roll['matched'],qualified=roll['pose_qualified'],reason=roll.get('reason'),sha256=sha256(roll_path))))
    picture(pictures)
    out=dict(criteria_sha256=sha256(OUT/'criteria.json'),decision='development gate failed; do not enable closure',
        captured_work_rows=sum(f['phase']=='work' for r in origin['runs'] for f in r['rows']),
        working_decisions_sha256=sha256(DEV/'decision.json'),views=views,cases=summary,
        closed_backoff_pair=direct['explicit_pair'],closed_backoff_decision_sha256=sha256(TASK/'seam_clear_observation_v1/decision.json'),
        connected_reader_development_pass=contact['passed'],connected_reader_tests=contact['synthetic'],
        connected_decision_sha256=sha256(TASK/'seam_connected_evidence_v1/decision.json'),
        confirmed=False,authorizes_vision=False,new_visual_runs=0)
    write_new(OUT/'decision.json',out)
    for row in summary:
        print(row['actual_mm'],'rows missing a positive through-ray in at least one face:',row['zero_depth_face_rows'])


if __name__=='__main__':
    main()
