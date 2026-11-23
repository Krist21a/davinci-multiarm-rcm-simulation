"""Same frozen slide mechanics; separately versioned, strictly veto-only optical reader."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import seam_slide_pilot as slide
import capture_seam_rail as capture
import ruler_readout
import verified_ruler
from capture_surface_seam import ROOT, add_arguments, sha256
from seam_slide_world import SeamSlideWorld

PLAN = ROOT/'config/task/seam_slide_v2.json'
SOURCES = (*slide.SOURCES, 'scripts/vision/verified_ruler.py', 'scripts/analysis/contact_veto_pilot.py')
OUT = ROOT/'validation/task/seam_slide_v2'


def verify_lock():
    p = json.loads(PLAN.read_bytes())
    if (sha256(Path(verified_ruler.__file__)) != p['veto_sha256'] or
            sha256(slide.SPEC) != p['fixture_protocol_sha256'] or
            sha256(slide.OUT/'selection.json') != p['prior_failed_selection_sha256']):
        raise ValueError('development registration mismatch')
    for mm, expect in ((0, 'UNKNOWN'), (8, 'BELOW'), (9, 'STOP'), (10, 'STOP')):
        im = np.zeros((100, 100, 3), np.uint8)
        z = np.full((100, 100), .15)
        for j in range(13):
            rows = slice(10+6*j, 16+6*j)
            r, k = 32+48*(j % 4), 35+46*(j//4)
            im[rows, 20:49] = (r, 235, k)
            im[rows, 49:79] = (r, k, 235)
            z[rows, 20:79] = .105
        im[10:10+6*mm, 47:52] = 0
        z[10:10+6*mm, 47:52] = .15
        if verified_ruler.classify(im, z)['decision'] != expect:
            raise ValueError('synthetic reader check failed')
    if verified_ruler.classify(np.zeros_like(im), np.full_like(z, .02))['decision'] != 'UNKNOWN':
        raise ValueError('shutter check failed')


def main(args):
    verify_lock()
    args.rail = 'slide'
    if args.stage == 'collect':
        if not args.case:
            raise ValueError('collection needs case')
        with patch.multiple(capture, PLAN=PLAN, SOURCES=SOURCES, SeamRailWorld=SeamSlideWorld):
            return capture.acquire(args)
    with patch.multiple(slide, SPEC=PLAN, SOURCES=SOURCES), patch.object(ruler_readout, 'classify', verified_ruler.classify):
        return slide.evaluate(args)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out-root', type=Path, default=OUT)
    ap.add_argument('--stage', choices=('collect', 'preflight', 'work'), required=True)
    ap.add_argument('--case', choices=('pull', 'a', 'b', 'c', 'd', 'e'))
    raise SystemExit(main(ap.parse_args()))
