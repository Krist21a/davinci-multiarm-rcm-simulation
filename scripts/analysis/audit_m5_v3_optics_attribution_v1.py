"""Attribution on the sealed uncut RGB-D frames, separate from the two readers."""
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import ndimage

from replay_m5_v3_corridor_v2 import ROOT, DATA, digest, load
from cell_certificate_v1 import decode
from m5_v3_perception import _station, bank_patch, inks, side_stations


ID = 'm5_v3_optics_attribution_v1'
PLAN = ROOT/f'config/task/{ID}.json'
CAPTURE = DATA/'m5_v3_readonly_v4'
DEPTH_TOL_M = .001
HALF_WINDOW = 2
INSET = .15


def verified(path, sha256):
    if digest(path) != sha256:
        raise ValueError(f'changed input: {path}')
    return path


class Frame:
    def __init__(self, path, shape):
        with np.load(path, allow_pickle=False) as sample:
            rgb, z = sample['rgb'], sample['depth']
            self.f, far = (sample[k].item() for k in ('focal', 'far'))
        if rgb.shape != (*shape, 3) or z.shape != shape:
            raise ValueError('unexpected registered frame dimensions')
        self.bank = inks(rgb)
        self.cap, self.face = decode(rgb)
        yy, xx = np.nonzero((self.bank >= 0) | (self.cap >= 0) | (self.face >= 0))
        self.offset = np.array([max(0, yy.min()-5), max(0, xx.min()-5)])
        y0, x0 = self.offset
        crop = np.s_[y0:yy.max()+6, x0:xx.max()+6]
        self.rgb, self.z = rgb[crop], z[crop]
        self.bank, self.cap, self.face = self.bank[crop], self.cap[crop], self.face[crop]
        self.good = np.isfinite(self.z) & (self.z > 0) & (self.z < far*.999)
        self.pp = np.array(shape)/2.-self.offset
        self.far = far
        self.rgb_full, self.z_full = rgb, z

    def bank_masks(self):
        result = [[], []]
        for side in range(2):
            for j in range(13):
                v = np.zeros(self.z.shape, bool)
                ij = bank_patch(self.good & (self.bank == side*13+j))
                v[tuple(ij.T)] = True
                result[side].append(v)
        return result

    def points(self, yx):
        depth = self.z[tuple(yx.T)]
        return np.column_stack(((self.pp-yx-.5)*depth[:, None]/self.f, depth))

    def pixels(self, xyz):
        return self.pp - self.f*xyz[..., :2]/xyz[..., 2, None] - .5

    def region(self, pix, expected_z):
        y, x = pix.T
        valid = self.good[y, x]
        z = self.z[y, x]
        delta = z - expected_z
        reported = []
        for i, (b, c, f) in enumerate(zip(self.bank[y, x], self.cap[y, x], self.face[y, x], strict=True)):
            label = (f'face:{f//2}:{f%2}' if f >= 0 else f'cap:{c//2}:{c%2}' if c >= 0
                     else f'bank:{b%13}:{b//13}' if b >= 0 else 'unclassified')
            zone = ('invalid' if not valid[i] else 'nearer' if delta[i] < -DEPTH_TOL_M
                    else 'farther' if delta[i] > DEPTH_TOL_M else 'compatible')
            reported.append(f'{zone} {label}')
        colors, counts = np.unique(self.rgb[y, x], axis=0, return_counts=True)
        indices = np.argsort(-counts, kind='stable')[:6]
        return dict(samples=len(pix), split=dict(Counter(reported)),
                    projected_depth_mm=np.round(np.asarray(expected_z)*1000., 5).tolist()
                    if np.ndim(expected_z) == 0 else
                    np.round(np.percentile(expected_z, [0, 50, 100])*1000., 5).tolist(),
                    residual_mm=np.round(np.percentile(delta[valid], [0, 50, 100])*1000., 5).tolist()
                    if valid.any() else [],
                    raw_colors=[dict(rgb=colors[k].tolist(), pixels=int(counts[k])) for k in indices])

    def probe(self, xyz):
        xy = np.rint(self.pixels(xyz)).astype(int)
        offsets = np.arange(-HALF_WINDOW, HALF_WINDOW+1)
        pix = (np.array(np.meshgrid(offsets, offsets)).reshape(2, -1).T + xy)
        pix = pix[((pix >= 0) & (pix < self.z.shape)).all(1)]
        return dict(yx=(xy+self.offset).tolist(), **self.region(pix, xyz[-1]))

    def bank_proxy(self, j, masks):
        sides = []
        for bank in masks:
            around = [np.argwhere(bank[k]) for k in (j-1, j+1)]
            sides.append(np.mean([np.median(self.points(pts), axis=0) for pts in around], axis=0)
                         if all(len(pts) for pts in around) else None)
        return dict(left=self.probe(sides[0]) if sides[0] is not None else None,
                    right=self.probe(sides[1]) if sides[1] is not None else None,
                    between=self.probe((sides[0]+sides[1])*.5) if all(s is not None for s in sides) else None)

    def bank_stats(self, j, side, masks, seeds):
        m = self.good & (self.bank == j+side*13)
        cc, _ = ndimage.label(m)
        area = sorted(np.bincount(cc[m])[1:].tolist(), reverse=True) if m.any() else []
        p = _station(inks(self.rgb_full), self.z_full, seeds, j, side, self.f, self.far)
        return dict(pixels=int(m.sum()), largest_components=area[:5], patch_px=int(masks[side][j].sum()),
                    station_seed=seeds[j] is not None, station_fit=p is not None)

    def triangle(self, j, side):
        corners = ((2*j, 2*j+1, 2*j+3), (2*j, 2*j+3, 2*j+2))[side]
        members = [np.argwhere(self.good & (self.cap == c)) for c in corners]
        color = np.array([248, 24+14*j, 147+58*side])
        out = dict(cap_ids=corners, cap_pixels=[len(p) for p in members],
                   face_pixels=int((self.face == 2*j+side).sum()),
                   exact_color_px=int(np.all(self.rgb == color, axis=-1).sum()))
        if any(len(p) < 4 for p in members):
            return dict(out, status='insufficient cap samples for independent projection')
        cz = np.array([np.median(self.z[tuple(p.T)]) for p in members])
        yx = np.array([np.median(p, axis=0) for p in members])
        T = (yx[1:] - yx[0]).T
        area = abs(np.linalg.det(T))/2.
        out.update(cap_yx=(yx+self.offset).tolist(), projected_area_px2=float(area))
        if area < 1e-6:
            return dict(out, status='degenerate proxy')
        lo = np.maximum(0, np.floor(yx.min(0)).astype(int))
        hi = np.minimum(self.z.shape, np.ceil(yx.max(0)).astype(int)+1)
        pix = np.stack(np.meshgrid(*(np.arange(a, b) for a, b in zip(lo, hi, strict=True))),
                       axis=-1).reshape(-1, 2)
        w = (pix-yx[0]) @ np.linalg.inv(T).T
        w = np.column_stack((1-w.sum(1), w))
        inside = (w >= INSET).all(1)
        ref = 1/(w[inside] @ (1/cz))
        return dict(out, status='interior tested' if inside.any() else 'no inset pixel centers',
                    optical=self.region(pix[inside], ref) if inside.any() else None)


def audit():
    plan = load(PLAN)
    dec = verified(CAPTURE/'decision.json', plan['capture_decision_sha256'])
    reg = verified(CAPTURE/'registration.json', plan['capture_registration_sha256'])
    saved = load(dec)
    if saved['registration_sha256'] != digest(reg) or saved['failure'] or saved['cleanup_error']:
        raise ValueError('not the registered complete acquisition')
    inputs = [PLAN, dec, reg, Path(__file__).resolve()]
    inputs += [ROOT/p for p in ('scripts/vision/m5_v3_perception.py', 'scripts/vision/cell_certificate_v1.py',
                               'scripts/vision/ruler_readout.py', 'scripts/vision/m5_v3_corridor_v2.py',
                               'scripts/vision/m5_guided_bank_corridor_v1.py')]
    captures = {}
    for name, record in saved['captures'].items():
        file = verified(CAPTURE/record['file'], record['sha256'])
        inputs.append(file)
        if record['fov_deg'] not in (70, 20):
            raise ValueError('unregistered view')
        wide = record['fov_deg'] == 70
        f = Frame(file, (960,1280) if wide else (1920,2560))
        masks = f.bank_masks()
        rows = {}
        if wide:
            seeds = [[None if q is None else q+f.offset for q in side_stations(side, k)]
                     for k, side in enumerate(masks)]
        for j in sorted(plan['suspect_rows']+plan['comparison_rows']):
            evidence = {}
            if j in plan['suspect_rows']:
                evidence['flanking_probes'] = f.bank_proxy(j,masks)
            if wide:
                evidence['banks'] = [f.bank_stats(j,s,masks,seeds[s]) for s in range(2)]
            else:
                evidence['faces'] = [f.triangle(j,s) for s in range(2)]
            rows[str(j)] = evidence
        captures[name] = dict(closed=record['closed'], fov_deg=record['fov_deg'], rows=rows)
    if set(captures) != {'work_closed', 'work_open', 'narrow_before_backoff', 'narrow_entry_clear'}:
        raise ValueError('incomplete capture set')
    return dict(plan=plan, authorizes_motion=False, captures=captures,
                sha256={p.relative_to(ROOT).as_posix():digest(p) for p in inputs})


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    evidence = audit()
    with args.out.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(evidence, out, indent=2, allow_nan=False)
        out.write('\n')
