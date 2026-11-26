"""Opt-in working reader that uses an exact index for the legacy entry guard."""
import numpy as np

from cut_perception import GAP_BAND_M, GAP_BIN_M, GAP_DEPTH_M, RUN_GAP_BINS, CutPerception, _provable_ahead

_ENTRY_START_BIN = 4  # (1 mm before entry + 1 mm after entry) / 0.5 mm


class EntryBinPerception(CutPerception):
    def _gap_scan(self, mask, sa_px, sb_px, pz, depth, line_lat, s_near, s_far):
        """Frozen native scan with only the start index made deterministic."""
        if GAP_BIN_M != .0005:
            raise ValueError('unregistered entry bin width')
        edges = np.arange(s_near - 0.001, s_far + 0.002, GAP_BIN_M)
        band = (np.abs(sb_px - line_lat) < GAP_BAND_M) & (sa_px >= edges[0]) & (sa_px < edges[-1])
        behind = (~mask & band & ((depth > pz + GAP_DEPTH_M) | ~np.isfinite(depth) |
                                  (depth >= self.far * 0.999)))
        idx = np.clip(np.digitize(sa_px[band], edges) - 1, 0, len(edges) - 2)
        occ = np.bincount(idx, weights=behind[band].astype(float), minlength=len(edges) - 1) >= 2
        b_start = min(_ENTRY_START_BIN, len(edges) - 2)
        first = np.flatnonzero(occ[b_start:])
        if len(first) == 0 or edges[b_start + int(first[0])] > s_near + 0.0025:
            return dict(run_end=None, run_len=0.0, slit_sel=np.zeros(mask.shape, bool),
                        recovery=False, reach_far=False,
                        band=band, occ=occ, b_start=b_start, edges=edges)
        lo = b_start + int(first[0])
        end_b, gap_b = lo, 0
        for b in range(lo + 1, len(edges) - 1):
            if occ[b]:
                end_b, gap_b = b, 0
            else:
                gap_b += 1
                if gap_b > RUN_GAP_BINS:
                    break
        run_end = float(edges[min(end_b + 1, len(edges) - 1)])
        reach_far = run_end >= s_far - 0.001
        slit_sel = behind & (sa_px >= edges[lo]) & (sa_px < run_end)
        recovery = (not reach_far and
                    _provable_ahead(run_end, band, occ, b_start, edges,
                                    sa_px, mask, depth, pz, self.far))
        return dict(run_end=run_end, run_len=run_end - float(edges[lo]),
                    slit_sel=slit_sel, recovery=recovery, reach_far=reach_far,
                    band=band, occ=occ, b_start=b_start, edges=edges)
