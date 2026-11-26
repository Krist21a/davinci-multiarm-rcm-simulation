"""Post-hoc development diagnostic only; deliberately never open holdout or tissue images."""
from pathlib import Path

import numpy as np

from score_m5_v4_plane_v1 import F, OUT, checked, depth, load, sample, sha256, write


def run():
    decision = load(OUT/'decision.json')
    frozen = load(checked(OUT/'development.json',decision['development_sha256']))
    cap = load(checked(OUT/'capture.json',frozen['capture_sha256']))
    lock = load(checked(OUT/'registration.json',decision['registration_sha256']))
    rows = {}
    for i,pose in enumerate(lock['spec']['development']):
        if not any(pose[3:5]):
            continue
        key = f'development_{i:02d}'
        m = cap['frames']['development'][key]
        s = sample(checked(OUT/m['file'],m['sha256']),pose)
        u = 1/depth(s['rc'],s['n'],s['d'],[0.]*4)
        slope = np.max(abs(s['n'][:2]))/(F*abs(s['d']))
        ratio = (u-1/s['z'])/slope
        predicted = 1/(u-.5*slope)
        rows[key] = dict(ratio_px_p05_median_p95=np.percentile(ratio,[5,50,95]).tolist(),
                         half_slope_p95_abs_mm=float(np.percentile(abs(predicted-s['z']),95)))
    write(OUT/'posthoc_dev_slope.json',dict(rows=rows,decision_sha256=sha256(OUT/'decision.json'),
          source_sha256=sha256(Path(__file__)),fitted_parameters=0,preregistered=False,authorizes_motion=False))


if __name__=='__main__':
    run()
