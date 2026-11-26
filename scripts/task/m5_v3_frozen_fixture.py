"""Place the sealed nominal fixture without consulting a tool pose or FK."""
import numpy as np

from sim_world import GRASP_RADIUS_M
from workspace import H, NU, NV


def place_sheet(w, geometry):
    if not w.canonical_seam or w.roi_proxy is not None:
        raise ValueError('not the registered coupon')
    def point(v):
        p = np.asarray(v,dtype=float)
        if p.shape!=(3,) or not np.isfinite(p).all():
            raise ValueError('invalid fixture geometry')
        return p.copy()
    w.sheet_origin,w.u,w.v = (point(geometry[k]) for k in ('sheet_origin','u','v'))
    axes = np.stack((w.u,w.v))
    if not np.allclose(axes@axes.T,np.eye(2),rtol=0,atol=1e-12):
        raise ValueError('invalid fixture axes')
    w.grasp_points = {arm:point(geometry['grasp_points'][arm]) for arm in (w.primary,w.assist)}
    w.call('tissueCreate',dict(origin=w.sheet_origin.tolist(),u=w.u.tolist(),v=w.v.tolist(),
           nu=NU,nv=NV,spacing=H,pretension=w.pretension,anchor_sides=False,anchor_from_row=3,
           grasp_radius=GRASP_RADIUS_M,seam_mode='surface',guide_band_m=w.sheet_cfg['guide_band_m']))
    w.steps(30)
    w.cutline_rest_m = np.asarray(w.call('tissueTransverseRestLengths'),dtype=float)
