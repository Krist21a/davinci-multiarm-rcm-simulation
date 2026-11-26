"""Hinge-only three-marker fit; the original marker masks, calibration and gates stay frozen."""
import numpy as np
from scipy.optimize import least_squares

from cut_perception import CutPerception, FIT_RMS_MAX_M, MIN_MARKER_PX, _kabsch, unit


def residual(centre,rays,ranges,radius):
    foot = rays@centre
    perp = np.linalg.norm(foot[:,None]*rays-centre,axis=1)
    disc = radius**2-perp**2
    return np.r_[foot-np.sqrt(np.maximum(disc,1e-14))-ranges,np.maximum(perp-radius,0.)]


def sphere(points,radius,initial):
    ranges = np.linalg.norm(points,axis=1)
    if (len(ranges)<MIN_MARKER_PX or not np.all(np.isfinite(ranges)&(ranges>radius))
            or not np.isfinite(initial).all() or radius<=0):
        return None,dict(nfev=0,reason='invalid sphere support')
    fit = least_squares(residual,initial,args=(points/ranges[:,None],ranges,radius),
                        loss='soft_l1',f_scale=.0005,max_nfev=60)
    rank = int(np.linalg.matrix_rank(fit.jac))
    valid = fit.success and np.isfinite(fit.x).all() and fit.x[2]>radius and rank==3
    return fit.x if valid else None,dict(nfev=fit.nfev,status=fit.status,rank=rank,pixels=len(points))


def read(reader,rgb,z):
    image = rgb.astype(float)
    centers,points = [],[]
    info = dict(valid=False,fits={})
    names = ('distal','proximal','third')
    for key,color in zip(names,reader.marker_colours,strict=True):
        c,_,why,pts = reader._marker(image,z,color)
        if c is None:
            return dict(info,reason=key+': '+why)
        centers.append(c)
        points.append(pts)
    _,_,first = _kabsch(reader.marker_offsets,np.stack(centers))
    info['initial_rms_m'] = first
    if not np.isfinite(first) or first>FIT_RMS_MAX_M:
        return dict(info,reason='initial marker RMS veto')
    fixed = []
    for key,pts,c in zip(names,points,centers,strict=True):
        result,info['fits'][key] = sphere(pts,reader.marker_radius,c)
        fixed.append(result)
    if any(p is None for p in fixed):
        return dict(info,reason='unresolved ray/sphere fit')
    R,t,rms = _kabsch(reader.marker_offsets,np.stack(fixed))
    info['fit_rms_m'] = rms
    if not np.isfinite(rms) or rms>FIT_RMS_MAX_M:
        return dict(info,reason='ray marker rigid-fit RMS out of tolerance')
    return dict(info,valid=True,reason='',tip=(R@reader.tcp_local+t).tolist(),
                jaw_axis=unit(R@reader.jaw_axis_local).tolist(),opening_axis=unit(R@reader.opening_axis_local).tolist())


def reader_for(model,f,far):
    return CutPerception(1280,960,f,far,model.rcm,tcp_local=model.tip_Y,jaw_axis_local=model.tip_R[:,2],
                         opening_axis_local=model.tip_R[:,0],ray_marker_fit=True)
