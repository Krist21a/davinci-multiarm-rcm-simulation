"""Isolate one normal-only boundary condition, not a new threshold or controller."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import rest_cell_pilot as previous
from capture_surface_seam import ROOT, add_arguments, sha256, write_new

SPEC=ROOT/'config/task/seam_planar_rim_v1.json'
OUT=ROOT/'validation/task/seam_planar_rim_v1'
CHAIN=previous.asym.asymmetric_source
INJECT='''
        for j=0,12 do for i=11,12 do
            local k=j*24+i+1
            if not T.pinned[k] then
                local p=T.x[k]
                T.x[k]=vsub(p,vmul(slideAxis,vdot(vsub(p,slideOrigin),slideAxis)))
            end
        end end
'''


def staged(parent):
    src=CHAIN(parent)
    site='\n        projectSlide()\n'
    if src.count(site)!=1:
        raise ValueError('unknown PBD solve stage')
    return src.replace(site,site+INJECT)


def main(args):
    base=previous.OUT/'registration.json'
    original=json.loads(base.read_bytes())
    amend=json.loads(SPEC.read_bytes())
    if sha256(ROOT/amend['inherits'])!=original['amendment_sha256']:
        raise ValueError('unknown metrology parent')
    contract=dict(**{k:v for k,v in original.items() if k!='id'},id=amend['id'],rim_spec=amend,
                  rim_sha256=sha256(SPEC),baseline_sha256=sha256(base))
    lock=args.out_root/'registration.json'
    if args.stage=='collect' and args.case=='pull' and not lock.exists():
        args.out_root.mkdir(parents=True,exist_ok=True)
        write_new(lock,contract)
    if json.loads(lock.read_bytes())!=contract:
        raise ValueError('registration changed')
    sources=(*previous.SOURCES,'scripts/analysis/planar_rim_pilot.py','config/task/seam_asymmetric_rest_v1.json')
    hashes={p:sha256(ROOT/p) for p in sources}
    if hashes['scripts/vision/cell_certificate_v1.py']!=contract['certificate_sha256'] or hashes['src/tissue/cell_certificate_print.lua']!=contract['print_sha256']:
        raise ValueError('optics drift')
    if args.stage=='collect':
        if not args.case:
            raise ValueError('missing case')
        args.rail='asym'
        with patch.object(previous.asym,'asymmetric_source',staged), patch.multiple(previous.acquisition,PLAN=lock,
                SOURCES=sources,SeamRailWorld=previous.World,RailCapture=previous.ReportCapture):
            return previous.acquisition.acquire(args)
    entries=contract['development_runs'][:1] if args.stage=='preflight' else contract['development_runs']
    for e in entries:
        r=json.loads((args.out_root/'asym'/e['id']/'oracle.json').read_bytes())
        rest=r.get('measured_cutline_rest_m') or []
        if len(rest)!=14 or any(not abs(v-.00276)<=1e-12 for v in rest):
            raise ValueError('invalid transverse rest lengths')
    with patch.object(previous.asym,'SPEC',lock):
        return previous.asym.evaluate(args.out_root,args.stage,contract,hashes)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--stage',required=True,choices=('collect','preflight','work'))
    ap.add_argument('--case',choices=('pull','a','b','c','d','e'))
    ap.add_argument('--out-root',type=Path,default=OUT)
    raise SystemExit(main(ap.parse_args()))
