"""Original ray/sphere estimator with a 200-evaluation budget for the hinge benchmark."""
import math

import numpy as np
from scipy.optimize import least_squares

from cut_perception import FIT_RMS_MAX_M, MIN_MARKER_PX, _kabsch, unit
from m5_v4_hinge_marker_v2 import reader_for


def sphere(points,radius,initial):
    ranges = np.linalg.norm(points,axis=1)
    if (len(ranges)<MIN_MARKER_PX or not np.all(np.isfinite(ranges)&(ranges>radius))
            or not np.isfinite(initial).all() or radius<=0):
        return None,dict(nfev=0,reason='invalid sphere support')
    rays = points/ranges[:,None]

    def residual(centre):
        foot = rays @ centre
        disc = radius**2 - np.sum((foot[:, None]*rays - centre)**2, axis=1)
        return np.r_[foot - np.sqrt(np.maximum(disc, 0)) - ranges, np.sqrt(np.maximum(-disc, 0))]

    fit = least_squares(residual,initial,loss='soft_l1',f_scale=.0005,max_nfev=200)
    rank = int(np.linalg.matrix_rank(fit.jac))
    valid = fit.success and np.isfinite(fit.x).all() and fit.x[2]>radius and rank==3
    return fit.x if valid else None,dict(nfev=fit.nfev,status=fit.status,rank=rank,pixels=len(points))


def read(reader,rgb,z):
    image = rgb.astype(float)
    names = ('distal','proximal','third')
    samples = [reader._marker(image,z,col) for col in reader.marker_colours]
    centers,uvs,errors,points = zip(*samples,strict=True)
    base = dict(fits={},**{k+'_uv':list(uv) if uv else None for k,uv in zip(names,uvs)})
    if any(c is None for c in centers):
        return dict(base,valid=False,reason='; '.join(f'{k} marker missing: {e}' for k,c,e in
                    zip(names,centers,errors) if c is None))
    R,t,rms = _kabsch(reader.marker_offsets,np.stack(centers))
    base['fit_rms_m'] = rms
    if not math.isfinite(rms) or rms>FIT_RMS_MAX_M:
        return dict(base,valid=False,reason='initial marker RMS veto')
    if reader.ray_marker_fit:
        fixed = []
        for name,pts,c in zip(names,points,centers,strict=True):
            value,base['fits'][name] = sphere(pts,reader.marker_radius,c)
            fixed.append(value)
        if any(p is None for p in fixed):
            return dict(base,valid=False,reason='unresolved ray/sphere fit')
        R,t,rms = _kabsch(reader.marker_offsets,np.stack(fixed))
        base.update(original_rms_m=base['fit_rms_m'],fit_rms_m=rms)
        if not math.isfinite(rms) or rms>FIT_RMS_MAX_M:
            return dict(base,valid=False,reason='ray marker rigid-fit RMS out of tolerance')
    if reader.tcp_local is None:
        return dict(base,valid=False,reason='tcp_local calibration missing')
    tip = R@reader.tcp_local+t
    result = dict(base,valid=True,reason='',tip=tip.tolist(),axis=R[:,1].tolist(),tip_uv=list(reader._project(tip)))
    if reader.jaw_axis_local is not None:
        result['jaw_axis'] = unit(R@reader.jaw_axis_local).tolist()
    if reader.opening_axis_local is not None:
        result['opening_axis'] = unit(R@reader.opening_axis_local).tolist()
    return result
