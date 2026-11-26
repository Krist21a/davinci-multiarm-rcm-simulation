"""Opt-in reader with the native gap veto restricted to its actual window."""
from types import FunctionType

import numpy as np

from cut_perception import AHEAD_WINDOW_M, GAP_DEPTH_M, CutPerception, _provable_ahead
from m5_guided_entry_bin_v1 import EntryBinPerception


def _window_provable_ahead(s,band,occ,b_start,edges,sa_px,mask,depth,pz,far):
    if not _provable_ahead(s,band,np.zeros_like(occ),b_start,edges,sa_px,mask,depth,pz,far):
        return False
    ahead = band & (sa_px>=s) & (sa_px<s+AHEAD_WINDOW_M+.0005)
    behind = ahead & ~mask & ((depth>pz+GAP_DEPTH_M)|~np.isfinite(depth)|(depth>=far*.999))
    bins = np.digitize(sa_px[behind],edges)-1
    bins = bins[(bins>=b_start)&(bins<len(occ))]
    # Preserve a mixed occupied bin's veto whenever it has actual in-window support.
    return not occ[bins].any()


def _with_window(method):
    # Keep the hash-pinned originals untouched; bind only the predicate in a private namespace.
    return FunctionType(method.__code__,dict(method.__globals__,_provable_ahead=_window_provable_ahead),
                        method.__name__,method.__defaults__,method.__closure__)


class AheadWindowPerception(EntryBinPerception):
    _cut = _with_window(CutPerception._cut)
    _gap_scan = _with_window(EntryBinPerception._gap_scan)
