"""Rasterize known passive cap/face material in an oblique pinhole camera, for development tests."""
import hashlib

import numpy as np

from cell_certificate import OPEN, PRESENT, UNKNOWN, classify


def depths(z):
    seed = int(hashlib.sha256(z.tobytes()).hexdigest()[:16], 16)
    yield 0., z
    yield .0005, z + np.random.default_rng(seed).normal(0., .0005, z.shape)


def contains(polygon, point):
    q = np.asarray(polygon)
    v = np.roll(q, -1, axis=0)-q
    w = point-q
    return bool(np.all(v[:, 0]*w[:, 1]-v[:, 1]*w[:, 0] >= -1e-8))


def render(n, kind='plain'):
    size, f = 480, 1280/np.tan(np.deg2rad(10.))
    pp = np.full(2, size/2)
    a, tilt = .29, np.deg2rad(59.)
    u = np.array([np.cos(a), np.sin(a), 0.])
    v = np.array([-np.sin(a)*np.cos(tilt), np.cos(a)*np.cos(tilt), np.sin(tilt)])
    normal, origin = np.cross(u, v), np.array([0., 0., .135])
    yy, xx = np.indices((size, size))
    ray = np.stack([(xx+.5-pp[0])/f, (yy+.5-pp[1])/f, np.ones(xx.shape)], axis=-1)
    sheet = (normal @ origin)/(ray @ normal)
    points = ray*sheet[..., None]-origin
    x, y = points @ u*1000+1.5, points @ v*1000+6.5
    im = np.zeros((*xx.shape, 3), np.uint8)
    z = np.full(xx.shape, .18)
    for j in range(13):
        t = y-j
        row = (t >= 0) & (t < 1)
        for s, w in enumerate((x+1, x-3)):
            bank = row & (w >= 0) & (w < 1)
            r, g = 32+48*(j % 4), 35+46*(j//4)
            im[bank], z[bank] = (r, g if s else 235, 235 if s else g), sheet[bank]
            # In the two bank triangles the print at a tagged vertex occupies
            # barycentric weight >= 0.85, as on the real specimen.
            first = w >= t
            if s == 0:
                weights = (np.where(first, w-t, 0), np.minimum(w, t))
            else:
                weights = (1-np.maximum(w, t), np.where(first, 0, t-w))
            for off, weight in enumerate(weights):
                if not (kind == 'far_cap' and (j+off, s) == (12, 1)):
                    im[bank & (weight >= .85)] = (14, 24+14*(j+off), 147+58*s)
        for h, tri in enumerate((x/3 >= t, x/3 < t)):
            mask = row & (x >= 0) & (x < 3) & tri
            live = j >= n
            if kind in ('half0', 'half1') and (j, h) == (n, int(kind[-1])):
                live = False
            if kind == 'bridge' and (j, h) == (4, 1):
                live = True
            if live:
                im[mask], z[mask] = (248, 24+14*j, 147+58*h), sheet[mask]
            if kind == 'target_cover' and (j, h) == (8, 1):
                im[mask], z[mask] = 100, .095
    if kind == 'hide_entry':
        im[y < 2.5], z[y < 2.5] = 90, .095
    if kind == 'shutter':
        im[:], z[:] = 90, .095
    def project(i, j):
        q = origin+((i-1.5)*u+(j-6.5)*v)*.001
        return pp+f*q[:2]/q[2]
    if kind in ('remote', 'deep_speck', 'near_speck'):
        ix, iy = (12, 12) if kind == 'remote' else np.floor(project(2.2, 8.25)).astype(int)
        im[iy, ix] = (248, 24+14*8, 147)
        if kind == 'near_speck':
            z[iy, ix] = sheet[iy, ix]
    return im, z, f, pp, np.array([project(3*s, j) for j in range(14) for s in (0, 1)])


CASES = ([(n, 'plain', n, n) for n in (3, 6, 8, 9, 10)]
         + [(0, 'plain', None, None), (11, 'plain', None, None),
            (8, 'half0', 8, 8), (8, 'half1', 8, 8),
            (9, 'bridge', None, None), (9, 'target_cover', None, None),
            (9, 'shutter', None, None), (9, 'remote', 9, 9), (9, 'deep_speck', 9, 9),
            (9, 'near_speck', None, None), (9, 'hide_entry', 9, None), (9, 'far_cap', 9, 9)])


def controls():
    results = []
    for n, kind, want_local, want_prefix in CASES:
        im, z, f, pp, uv = render(n, kind)
        for sigma, depth in depths(z):
            r = classify(im, depth, focal=f, principal=pp, sigma=sigma)
            outside = [i for i, c in enumerate(r.get('corners', []))
                       if c and not contains(c['polygon'], uv[i])]
            decision = UNKNOWN if want_prefix is None else 'STOP' if want_prefix >= 9 else 'BELOW'
            wrong_faces = []
            for j, pair in enumerate(r.get('states', [])):
                for h, state in enumerate(pair):
                    live = ((j >= n and not (kind == 'half'+str(h) and j == n))
                            or (kind == 'bridge' and (j, h) == (4, 1)))
                    if (state == OPEN and live) or (state == PRESENT and not live):
                        wrong_faces.append([j, h])
            results.append(dict(n=n, case=kind, sigma=sigma, decision=r['decision'], local=r['local_mm'],
                                prefix=r['prefix'], escaped=outside, wrong_faces=wrong_faces,
                                passed=(not outside and not wrong_faces and r['local_mm'] == want_local
                                        and r['prefix'] == want_prefix and r['decision'] == decision)))
    return results
