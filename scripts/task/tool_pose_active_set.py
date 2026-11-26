"""Bounded IK fallback preserving the native target, tolerances and joint margins."""
import numpy as np
from scipy.optimize import lsq_linear

import tool_pose as native


def bounded_step(J,e,z,lo,hi):
    """Solve the damped linearized step in the joint/trust-region box."""
    lower = np.maximum(lo-z,-native._STEP_MAX)
    upper = np.minimum(hi-z,native._STEP_MAX)
    A = np.concatenate((J,native._DAMPING*np.eye(len(z))))
    b = np.r_[-e,np.zeros(len(z))]
    free = lower<upper
    step = lower.copy()
    if free.any():
        b = b-A[:,~free]@step[~free]
        step[free] = lsq_linear(A[:,free],b,bounds=(lower[free],upper[free]),method='bvls').x
    return step


def solve(fk,q0,limits,target_p,target_axis=None,target_yaw_axis=None,axis_col=2,yaw_col=1):
    first = native.solve(fk,q0,limits,target_p,target_axis,target_yaw_axis,axis_col=axis_col,yaw_col=yaw_col)
    if first['converged']:
        return first
    lo,hi = np.asarray(limits[0])+native._MARGIN,np.asarray(limits[1])-native._MARGIN
    if not np.all(hi>lo):
        return first
    scale = native._SCALE
    lo,hi = lo*scale,hi*scale
    z = np.clip(first['q']*scale,lo,hi)
    p = np.asarray(target_p,dtype=float)
    axis = None if target_axis is None else target_axis/np.linalg.norm(target_axis)
    hinge = None if target_yaw_axis is None else target_yaw_axis/np.linalg.norm(target_yaw_axis)
    tol_p,tol_axis = .0002,.02

    def evaluate(x):
        return native._residual(native._pose(fk,x/scale),p,axis,hinge,1.,tol_p/tol_axis,axis_col,yaw_col)

    e,dp,da,dh = evaluate(z)
    converged = False
    for i in range(1,31):
        J = np.empty((len(e),len(z)))
        for j,eps in enumerate(native._FD_EPS):
            left,right = z.copy(),z.copy()
            left[j],right[j] = max(lo[j],z[j]-eps),min(hi[j],z[j]+eps)
            J[:,j] = (evaluate(right)[0]-evaluate(left)[0])/(right[j]-left[j])
        step = bounded_step(J,e,z,lo,hi)
        for rate in (1.,.5,.25,.125):
            candidate = np.clip(z+rate*step,lo,hi)
            residual = evaluate(candidate)
            if np.linalg.norm(residual[0])<np.linalg.norm(e):
                z = candidate
                e,dp,da,dh = residual
                break
        else:
            break
        converged = dp<tol_p and all(a is None or a<tol_axis for a in (da,dh))
        if converged:
            break
    at_limit = [n for j,n in enumerate(native.Q_NAMES)
                if z[j]<=lo[j]+1e-9*scale[j] or z[j]>=hi[j]-1e-9*scale[j]]
    return dict(q=z/scale,pos_err_m=dp,axis_err_rad=da,yaw_axis_err_rad=dh,
                iterations=i,converged=bool(converged),at_limit=at_limit,active_set=True)
