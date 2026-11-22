"""Verify a same-bite material chart against a fresh visual slit before reusing it."""
import numpy as np
from scipy import ndimage

from sim_world import SimWorld
from material_coordinate import COLORS, COLOR_TOL, MIN_PAIRS


def tracks(rgb,front_uv,forward_uv):
    """Coincident band transitions on the two distinct banks, in the unchanged camera."""
    im=np.asarray(rgb)
    # Restrict to the immutable flushed four-band print, not similarly shaped background.
    lab=np.full(im.shape[:2],-1,np.int8)
    for i,color in enumerate(COLORS):
        lab[np.sum((im-np.asarray(color)*255.)**2,axis=2)<=COLOR_TOL**2]=i
    origin=np.asarray(front_uv)
    fwd=np.asarray(forward_uv)-origin
    if not np.isfinite(fwd).all() or np.linalg.norm(fwd)<1:
        return {}
    lateral=np.array([-fwd[1],fwd[0]])/np.linalg.norm(fwd)
    scans={}
    for j in range(1,4):
        m=(lab==j)&ndimage.binary_dilation(lab==j-1)
        y,x=np.nonzero(m)
        projection=(x+.5-origin[0])*lateral[0]+(y+.5-origin[1])*lateral[1]
        for sign in (-1,1):
            # Exclude the slit and the tiny one-pixel colour contacts along its lip.
            keep=sign*projection>=3
            scans[j,sign]=set((y[keep]*im.shape[1]+x[keep]).tolist())
    return scans


def steady(a,b):
    return bool(len(a)==6 and a.keys()==b.keys() and all(
        len(s&b[k])>=max(MIN_PAIRS,.6*len(s),.6*len(b[k])) for k,s in a.items()))


class SameBiteWorld(SimWorld):
    _anchor_stamp=None
    _transfer_tick=None

    def _update_fix(self,est):
        cut=est.get('cut') or {}
        tissue=est.get('tissue') or {}
        mode=cut.get('mode')
        if (mode not in ('pre_cut','slit','bank_slit') or not cut.get('front_valid') or not tissue.get('valid')
                or not (est.get('scissors') or {}).get('valid')):
            return
        closed_before=hasattr(self,'_pre_closure_progress')
        if mode=='pre_cut' and (self.awaiting_new_slit or closed_before):
            return
        super()._update_fix(est)
        fix=self._fix
        if fix is None:
            return
        if fix['tick']==self.tick:
            p=fix['front']
            uv,_=self.project(p)
            fwd,_=self.project(p+.004*fix['direction'])
            self._anchor_stamp=None
            if uv is not None and fwd is not None:
                self._anchor_stamp=(fix['tick'],fix['epoch'],np.array(uv),np.array(fwd),
                                    tracks(self.image,uv,fwd))
            return
        if (not self._anchor_stamp or self.awaiting_new_slit or mode not in ('slit','bank_slit')
                or not 0<self.tick-fix['tick']<=self._fix_max_age or fix['epoch']!=self._fix_epoch
                or fix['progress']>=self.cfg['cut']['length_m'] or (cut.get('material_chart') or {}).get('valid')):
            return
        t,e,uv,fwd,old=self._anchor_stamp
        if (t,e)!=(fix['tick'],fix['epoch']):
            return
        p=np.asarray(cut.get('front'))
        d=np.asarray(cut.get('direction'))
        n=np.asarray(tissue.get('normal'))
        if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (p,d,n)):
            return
        drift=float(np.linalg.norm(p-fix['front']))
        cos_tol=np.cos(np.deg2rad(self.cfg['cut']['vision_axis_tolerance_deg']))
        if (drift>=.0005 or d@fix['direction']<cos_tol or n@fix['normal']<cos_tol
                or not steady(old,tracks(self.image,uv,fwd))):
            return
        self._transfer_tick=self.tick
        self.event_log.append(dict(event='same_bite_confirmed',tick=self.tick,anchor_tick=t,
                                   drift_m=drift,mode=mode,progress_m=fix['progress']))

    def _fix_valid(self):
        fix=self._fix
        return bool(fix is not None and not self.awaiting_new_slit and fix['epoch']==self._fix_epoch
                    and (fix['tick']==self.tick or (self._transfer_tick==self.tick
                         and 0<self.tick-fix['tick']<=self._fix_max_age)))

    def _closure_event(self):
        ev=super()._closure_event()
        cut=(self._last_est or {}).get('cut') or {}
        ev.update(direct_front=bool(cut.get('front_valid') and cut.get('mode') in ('pre_cut','slit','bank_slit')),
                  material_source='current' if self._fix['tick']==self.tick else 'confirmed_same_bite')
        return ev
