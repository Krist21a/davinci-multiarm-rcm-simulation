"""Optical endpoint authority for the guided sheet, independent of tool-marker pose."""
import math

import numpy as np

from cell_certificate_v2 import classify


def verdict(report):
    n = report.get('prefix')
    if type(n) is not int or not 0 < n <= 10 or n != report.get('local_mm'):
        return 'UNKNOWN'
    if n in (9, 10) and report.get('decision') == 'STOP':
        return 'TARGET'
    if n < 9 and report.get('decision') == 'BELOW':
        return 'BELOW'
    return 'UNKNOWN'


def measure(rgb, depth, *, focal, far, sigma):
    if (rgb.shape != (1920, 2560, 3) or rgb.dtype != np.uint8
            or depth.shape != (1920, 2560) or depth.dtype.kind != 'f'
            or not all(math.isfinite(x) for x in (focal, far, sigma))
            or focal <= 0 or far <= 0 or sigma < 0
            or not math.isclose(2*math.atan(1280/focal), math.radians(20), abs_tol=1e-6)):
        return dict(verdict='UNKNOWN', reason='invalid review optics', prefix=None, local_mm=None)
    measured = np.where((depth > 0) & (depth <= np.float32(far)), depth, np.nan)
    report = classify(rgb, measured, principal=(1280, 960), focal=focal, sigma=sigma)
    return dict(report, verdict=verdict(report))
