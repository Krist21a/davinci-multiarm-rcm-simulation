"""Scoring and veto only: compare all coupon edges and faces to the uncut mesh."""
import numpy as np

from seam_topology import material_state
from workspace import NU, NV


def _topology():
    ends = {}
    for j in range(NV-1):
        for i in range(NU-1):
            a = j*NU+i
            b,d,e = a+1,a+NU,a+NU+1
            # An edge shared by two faces damages both incident material rows.
            for edge in ((a,b),(b,e),(a,e),(a,d),(d,e)):
                ends[edge] = j+1
            if i!=NU//2-1:
                ends[b,d] = j+1  # non-face spring, still a structural loss
    keys = sorted(ends)
    return np.array(keys), np.array([ends[k] for k in keys])


KEYS, ROW_ENDS = _topology()


def _alive(triples):
    table = np.asarray(triples)
    if table.dtype.kind not in 'iu' or table.size!=3*len(KEYS):
        raise ValueError('incomplete constraint table')
    table = table.reshape(-1,3)
    keys = np.sort(table[:,:2],axis=1)
    order = np.lexsort((keys[:,1],keys[:,0]))
    if not np.array_equal(keys[order],KEYS) or not np.isin(table[:,2],[0,1]).all():
        raise ValueError('constraint identities or live bits differ from the registered mesh')
    return table[order,2].astype(bool)


class DamageBaseline:
    def __init__(self, uncut):
        self.alive = _alive(uncut)
        if not self.alive.all():
            raise ValueError('uncut baseline already damaged')
        self.alive.setflags(write=False)

    def score(self, current):
        lost = self.alive & ~_alive(current)
        result = material_state(current)
        result.update(farthest_partial_row_end=int(ROW_ENDS[lost].max(initial=0)),
                      newly_dead_edges=int(lost.sum()))
        return result


def within_envelope(state, goal):
    return all(state[k]<=min(10,goal+1) for k in ('farthest_row_end','farthest_partial_row_end'))


def exact_endpoint(state, goal):
    return all(state[k]==goal for k in ('prefix_rows','farthest_row_end','farthest_partial_row_end'))
