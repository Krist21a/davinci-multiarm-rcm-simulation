"""Versioned correction to transverse strain units; do not overwrite historical trials."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import asymmetric_cell_pilot as asym
import capture_seam_rail as acquisition
from capture_surface_seam import ROOT, add_arguments, sha256, write_new
from cell_rest_truth import CellRestTruth
from seam_slide_world import SeamSlideWorld

SPEC=ROOT/'config/task/seam_asymmetric_rest_v1.json'
OUT=ROOT/'validation/task/seam_asymmetric_rest_v1'
READBACK=ROOT/'src/tissue/cell_rest_readback.lua'
SOURCES=(*asym.SOURCES,'scripts/analysis/rest_cell_pilot.py','scripts/task/cell_rest_truth.py',
         'src/tissue/cell_rest_readback.lua','config/task/seam_asymmetric_cell_v1.json')


class World(SeamSlideWorld):
    def start(self):
        def build(parent):
            return asym.asymmetric_source(parent)+'\n'+READBACK.read_text(encoding='utf-8')
        with patch('seam_slide_world.staged_source',build):
            super().start()

    def place_sheet(self):
        super().place_sheet()
        rest=np.asarray(self.call('cellTransverseRestLengths'),float)
        if rest.shape!=(14,) or not np.allclose(rest,.003*(1-self.pretension),atol=1e-12,rtol=0):
            raise ValueError('asymmetric rest readback inconsistent with the registered model')
        self.cutline_rest_m=rest.copy()
        self.cutline_rest_m.setflags(write=False)
        # A new experimental world, with the same existing truth separation and audit trail.
        old=self.truth
        self.truth=CellRestTruth(self)
        self.truth.strict=old.strict
        self.truth.reads,self.truth.reads_by_purpose=old.reads,old.reads_by_purpose


class ReportCapture(asym.wide.CaptureWide):
    def supervise(self,obs):
        if hasattr(self.w,'cutline_rest_m') and 'measured_cutline_rest_m' not in self.record:
            self.record['measured_cutline_rest_m']=self.w.cutline_rest_m.tolist()
        return super().supervise(obs)


def old_units():
    src=asym.OUT/'asym/pull/oracle.json'
    r=json.loads(src.read_bytes())
    if r['final_state']!='FAILED' or r.get('events') or any(x['cut_length_m']!=0 for x in r['observations']):
        raise ValueError('expected original pre-bite failure')
    return dict(source_sha256=sha256(src),not_a_rescored_trajectory=True,
        observations=[dict(tick=x['tick'],state=x['state'],recorded=x['tension'],true_equivalent=(1+x['tension'])/3-1)
                      for x in r['observations'] if not x['torn']])


def main(args):
    amend=json.loads(SPEC.read_bytes())
    inherited=asym.OUT/'registration.json'
    parent=json.loads(inherited.read_bytes())
    baseline=ROOT/amend['inherits']
    if sha256(baseline)!=parent['asym_spec_sha256']:
        raise ValueError('wrong fixture registration')
    full={**parent,'id':amend['id'],'rest_amendment':amend,'amendment_sha256':sha256(SPEC),
          'uncorrected_fixture_sha256':sha256(inherited),'prior_units':old_units()}
    lock=args.out_root/'registration.json'
    if args.stage=='collect' and args.case=='pull' and not lock.exists():
        args.out_root.mkdir(parents=True,exist_ok=True)
        write_new(lock,full)
    if json.loads(lock.read_bytes())!=full:
        raise ValueError('registration mismatch')
    hashes={p:sha256(ROOT/p) for p in SOURCES}
    if hashes['scripts/vision/cell_certificate_v1.py']!=full['certificate_sha256'] or hashes['src/tissue/cell_certificate_print.lua']!=full['print_sha256']:
        raise ValueError('optical registration mismatch')
    if args.stage=='collect':
        if not args.case:
            raise ValueError('collection needs a case')
        args.rail='asym'
        with patch.multiple(acquisition,PLAN=lock,SOURCES=SOURCES,SeamRailWorld=World,RailCapture=ReportCapture):
            return acquisition.acquire(args)
    for e in (full['development_runs'][:1] if args.stage=='preflight' else full['development_runs']):
        r=json.loads((args.out_root/'asym'/e['id']/'oracle.json').read_bytes())
        if (len(r.get('measured_cutline_rest_m',[]))!=14 or
                not np.allclose(r['measured_cutline_rest_m'],.00276,rtol=0,atol=1e-12)):
            raise ValueError('missing real solver rest-length readback')
    with patch.object(asym,'SPEC',lock):
        return asym.evaluate(args.out_root,args.stage,full,hashes)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--stage',choices=('collect','preflight','work'),required=True)
    ap.add_argument('--case',choices=('pull','a','b','c','d','e'))
    ap.add_argument('--out-root',type=Path,default=OUT)
    raise SystemExit(main(ap.parse_args()))
