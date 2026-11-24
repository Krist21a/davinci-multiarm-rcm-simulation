"""Acquire the indexed-print version of the fixed HD development checkpoints in a fresh folder."""
import argparse
import json
from pathlib import Path

from local_hd_screen import ROOT, SOURCES, add_arguments, collect, sha256, write_new

SPEC = ROOT/'config/task/m5_indexed_reference_v1.json'


def register(out):
    spec=json.loads(SPEC.read_bytes())
    parent=ROOT/spec['parent']
    if sha256(parent/'registration.json')!=spec['parent_registration_sha256']:
        raise ValueError('wrong parent registration')
    historical=ROOT/'validation/task/m5_junction_hd_dev_v1/material_support.json'
    if sha256(historical)!=spec['orphan_evidence_sha256']:
        raise ValueError('wrong orphan witness')
    lock=dict(criteria=spec,spec_sha256=sha256(SPEC),
              sources={p.relative_to(ROOT).as_posix():sha256(p) for p in [*SOURCES,Path(__file__)]},
              parent_oracles={r['name']:sha256(parent/r['name']/'oracle.json') for r in spec['runs']})
    out.mkdir(parents=True,exist_ok=True)
    path=out/'registration.json'
    if not path.exists():
        write_new(path,lock)
    if json.loads(path.read_bytes())!=lock:
        raise ValueError('acquisition sources or criteria changed')
    return lock


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--case',choices=('pull','a','b','c'),required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    raise SystemExit(collect(args,args.out,register(args.out)))
