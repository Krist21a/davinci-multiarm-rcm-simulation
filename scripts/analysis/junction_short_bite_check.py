"""Check an archived smaller-advance intervention without running a new task or using its optics."""
import argparse
from pathlib import Path

import numpy as np

from junction_material_support import evidence, facets
from replay_bilateral_3d import ROOT, digest, load, write
from seam_topology import material_state


def read(folder,frame):
    p=folder/'structure'/(frame['key']+'.npz')
    if digest(p)!=frame['structure_sha256']:
        raise ValueError('structure digest mismatch')
    with np.load(p,allow_pickle=False) as d:
        edges,faces,nodes=(d[k] for k in ('edges','faces','nodes'))
    state=material_state(edges)
    if state['prefix_rows']!=frame['prefix_mm'] or not np.array_equal(faces,state['faces']):
        raise ValueError('invalid reference state')
    return (edges,faces,nodes,np.array(frame['tcp']))


def main(out):
    src=ROOT/'validation/task/seam_slide_v2/slide'
    hd=ROOT/'validation/task/m5_local_hd_dev_v1'
    spec=ROOT/'config/task/seam_slide_v2.json'
    plan=dict(scope='Scoring only. Compare existing 2-mm-cap d/e with their 3-mm-cap a/b/c reference in the archived slide mechanics. First check EXACT nodes, faces, constraints and TCP at every loaded/work checkpoint of a/b/c against the HD development trajectories. Check the same initial loaded state in d/e. At every actual 8-mm smaller-cap work checkpoint require surviving shared ordinal edges at all four last-open-cell reference corners. Report other actual lengths. Existence is necessary only; no substitution for the HD images or qualification of a control change.',
              sources={p.relative_to(ROOT).as_posix():digest(p) for p in
              [Path(__file__),Path(__file__).with_name('junction_material_support.py'),spec,
               *(src/c/'oracle.json' for c in 'abcde'),*(hd/c/'oracle.json' for c in 'abc')]})
    write(out/'short_bite_plan.json',plan)
    initial=None
    matches=[]
    rows=[]
    for case in 'abcde':
        folder=src/case
        old=load(folder/'oracle.json')
        if old['spec_sha256']!=digest(spec) or old['entry']['cap_mm']!=(3 if case in 'abc' else 2):
            raise ValueError('unexpected mechanical recipe')
        for fr in old['frames']:
            phase,mm=fr['phase'],fr['prefix_mm']
            if phase not in ('work','loaded'):
                continue
            data=read(folder,fr)
            if phase=='loaded' and initial is None:
                initial=data
            if case in 'abc':
                new=load(hd.joinpath(case,'oracle.json'))
                peer=next(f for f in new['frames'] if (f['phase'],f['bites'],f['prefix_mm'])==
                          (phase,fr['bites'],mm))
                ref=read(hd/case,peer)
            elif phase=='loaded':
                ref=initial
            else:
                ref=None
            if ref is not None:
                checks={k:np.array_equal(a,b) for k,a,b in zip(('edges','faces','nodes','tcp'),data,ref)}
                matches.append(dict(case=case,phase=phase,bites=fr['bites'],matches=checks))
            if phase=='work':
                alive=facets(data[0])
                pts=[evidence(alive,j,side) for j in (mm-1,mm) for side in (0,1)]
                rows.append(dict(case=case,mm=mm,bites=fr['bites'],points=pts,
                                 surviving=sum(bool(p['live_same_bank_ab_edges']) for p in pts)))
    matched=all(all(r['matches'].values()) for r in matches) and len(matches)==14
    eight=[r for r in rows if r['case'] in 'de' and r['mm']==8]
    write(out/'short_bite_check.json',dict(plan_sha256=digest(out/'short_bite_plan.json'),matches=matches,
          matched_context=matched,rows=rows,smaller_cap_preserves_four_at_eight=matched and
          {r['case'] for r in eight}=={'d','e'} and all(r['surviving']==4 for r in eight)))
    print('context matches:',matched,'work corners per length:',
          [(r['case'],r['mm'],r['surviving']) for r in rows])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True,type=Path)
    main(ap.parse_args().out)
