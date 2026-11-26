"""Geometry-only check of one 2x work raster; not synthetic optical evidence."""
from pathlib import Path

import numpy as np

from screen_m5_v4_center_v1 import KEYS, OUT, PARENT, Sheet, check, load, measure, projected, rays
from screen_m5_v4_center_v1 import sha256, verify, witnesses, write

DEST = OUT.parent/'m5_v4_center_density_v1'
SPEC = dict(parent_decision='5d54ff13f6a62e9278ee4bac91cc4f52114c127a8eec8023d9883d57790b561c',
            factors=[1,2],scope='Scorer geometry only. Same six poses and 70 deg, same print, pixel centers, '
            'no image interpolation, no simulator, marker simulation, depth noise or motion. Compare native '
            'geometry counts and topology to sealed native observations. Only one doubled-size candidate: '
            'require every frame to meet geometry-only central witness gate before suggesting a real passive '
            'capture. Separate cap and face routes; satisfying a witness count does not establish useful room.')


def geometry(nodes,edges,shape,f):
    sheet = Sheet(nodes,edges)
    px = projected(nodes[9:12,10:14],shape,f).reshape(-1,2)
    lo = np.maximum(np.floor(px.min(0)).astype(int)-2,0)
    hi = np.minimum(np.ceil(px.max(0)).astype(int)+3,shape)
    rc = np.moveaxis(np.indices(hi-lo),0,-1).reshape(-1,2)+lo
    ray = rays(rc,shape,f)
    hits = [sheet.hits(r) for r in np.array_split(ray,max(1,int(np.ceil(len(ray)/64))))]
    z,ids = (np.concatenate(part) for part in zip(*hits,strict=True))
    rgb = np.zeros((len(rc),3),dtype=np.uint8)
    for j,i,h in set(map(tuple,ids.tolist())):
        if j not in (9,10) or i not in (10,11,12):
            continue
        mask = (ids==[j,i,h]).all(1)
        if i==11:
            rgb[mask] = [248,24+14*j,147+58*h]
            continue
        tri_idx = (((j,i),(j,i+1),(j+1,i+1)),((j,i),(j+1,i+1),(j+1,i)))[h]
        tri = nodes[tuple(np.array(tri_idx).T)]
        p = z[mask,None]*ray[mask]-tri[0]
        uv = np.linalg.lstsq((tri[1:]-tri[0]).T,p.T,rcond=None)[0].T
        bary = np.column_stack((1-uv.sum(1),uv))
        for k,(row,col) in enumerate(tri_idx):
            if row==10 and col in (11,12):
                rgb[np.flatnonzero(mask)[bary[:,k]>=.85]] = [14,164,147+58*(col-11)]
    evidence = witnesses(rgb.reshape(*(hi-lo),3))
    w = measure(evidence,z.reshape(hi-lo),focal=f,far=.3,sigma=0.,axis=None)
    return dict(ink=evidence[2],contacts=w['contacts'],cap_geometry=w['cap_witness'],face_geometry=w['face_witness'])


def run():
    parent = load(verify(OUT/'decision.json',SPEC['parent_decision']))
    reg = load(verify(OUT/'registration.json',parent['registration_sha256']))
    predictions = load(verify(OUT/'optical_predictions.json',parent['optical_sha256']))
    DEST.mkdir(exist_ok=False)
    write(DEST/'registration.json',dict(spec=SPEC,source_sha256=sha256(Path(__file__))))
    rows = {}
    for rec in reg['records']:
        h,case,name,review = (rec[k] for k in ('h','case','name','review'))
        folder = PARENT/case
        r = load(verify(folder/'run.json',reg['sources'][f'validation/task/{PARENT.name}/{case}/run.json']))
        with (np.load(verify(folder/r['captures'][name]['file'],h)) as f,
              np.load(verify(folder/r['scoring_sidecars'][name]['file'],rec['files'][1])) as s,
              np.load(verify(folder/r['review_sidecars'][review]['file'],rec['files'][2])) as e):
            nodes = (s['nodes_world']-f['cam_t'])@f['cam_R']
            group = [geometry(nodes,e['edges'],tuple(k*np.array(f['depth'].shape)),k*f['focal'].item())
                     for k in SPEC['factors']]
        native = predictions['frames'][h]
        same = all(group[0]['ink'][k]==native['ink'][k] for k in KEYS) and group[0]['contacts']==native['draws'][0]['contacts']
        rows[h] = dict(case=case,name=name,native_matches=same,doubled=group[1],native=group[0])
    recommend = all(r['native_matches'] and (r['doubled']['cap_geometry'] or r['doubled']['face_geometry'])
                    for r in rows.values())
    write(DEST/'decision.json',dict(registration_sha256=sha256(DEST/'registration.json'),rows=rows,
          geometry_only=True,all_have_candidate_witness=recommend,authorizes_motion=False))
    for r in rows.values():
        print(r['case'],r['name'],r['native_matches'],r['doubled'])


if __name__=='__main__':
    run()
