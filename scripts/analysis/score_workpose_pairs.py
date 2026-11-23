"""Compare RGB-D at matched one-action work poses; truth overlays are scoring-only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from probe_workpose_pairs import PROTOCOL, ROOT
from cut_perception import _sheet_mask


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(z, points):
    pc = (np.asarray(points) - z['cam_t']) @ z['cam_R']
    h, w = z['depth'].shape
    if not np.isfinite(pc).all() or np.any(pc[..., 2] <= 0):
        raise ValueError('invalid scoring projection')
    return [w / 2, h / 2] - float(z['focal']) * pc[..., :2] / pc[..., 2:3]


def regions(z):
    front = z['truth_front']
    uv = project(z, front)
    direction = z['truth_direction']
    nodes = z['nodes']
    i = np.argmin(np.linalg.norm(nodes[:, nodes.shape[1] // 2] - front, axis=1))
    lat = nodes[i, -1] - nodes[i, 0]
    lat = lat - (lat @ direction) * direction
    lat /= np.linalg.norm(lat)
    corners = project(z, [front + d * direction + b * lat for d, b in
                          ((0., -.0015), (.0025, -.0015), (.0025, .0015), (0., .0015))])
    lo = np.floor(corners.min(axis=0)).astype(int)
    hi = np.ceil(corners.max(axis=0)).astype(int)
    h, w = z['depth'].shape
    if not (1 <= lo[0] <= hi[0] < w - 1 and 1 <= lo[1] <= hi[1] < h - 1):
        raise ValueError('front patch crosses the sensor border')
    u, v = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
    edges = np.roll(corners, -1, axis=0) - corners
    cross = ((v[..., None] - corners[:, 1]) * edges[:, 0]
             - (u[..., None] - corners[:, 0]) * edges[:, 1])
    inside = (cross >= 0).all(axis=-1) | (cross <= 0).all(axis=-1)
    cx, cy = np.rint(uv).astype(int)
    return uv, corners, (v[inside], u[inside]), np.s_[cy-1:cy+2, cx-1:cx+2]


def measurements(z, reg):
    has_sheet = _sheet_mask(z['rgb'].astype(float))[reg]
    without = _sheet_mask(z['no_cutter_rgb'].astype(float))[reg]
    d = (z['no_cutter_depth'] - z['depth'])[reg]
    if not d.size or not np.isfinite(d).all():
        raise ValueError('missing raw depth evidence')
    return dict(pixels=d.size, cutter_nearer=int((d > .00001).sum()),
                sheet=int(has_sheet.sum()), sheet_without_cutter=int(without.sum()))


def tile(image, center, polygon=None):
    # Keep the raw pixel grid visible; the outline stops short of the front marker.
    cx, cy = np.rint(center).astype(int)
    x, y = cx - 22, cy - 22
    pane = Image.fromarray(image).crop((x, y, x+45, y+45)).resize((225, 225), Image.Resampling.NEAREST)
    dr = ImageDraw.Draw(pane)
    if polygon is not None:
        xy = [tuple((p - [x, y]) * 5 + 2) for p in polygon]
        dr.line(xy + [xy[0]], fill='#00cabb', width=1)
    u, v = (center - [x, y]) * 5 + 2
    for a, b in ((-12, -5), (5, 12)):
        dr.line((u+a, v, u+b, v), fill='#ffdb00', width=2)
        dr.line((u, v+a, u, v+b), fill='#ffdb00', width=2)
    return pane


def score(folder):
    protocol = json.loads(PROTOCOL.read_bytes())
    out = []
    for phase, spec in protocol['cases'].items():
        canvas = Image.new('RGB', (944, 116 + 266 * len(spec['distances_m'])), '#152023')
        dr = ImageDraw.Draw(canvas)
        dr.text((14, 10), f'PSM2  |  {phase.upper()}   |   checkpoint {spec["tick"]} -> {spec["tick"]+1}', fill='white')
        dr.text((14, 34), 'Yellow: model front (scoring only)   Teal: forward patch   Depth delta: brighter = more hidden by cutter', fill='white')
        for col, header in enumerate(('Before / RGB', 'After / RGB', 'After / cutter hidden', 'After / depth delta (0-3 mm)')):
            dr.text((12 + 234 * col, 69), header, fill='white')
        baseline = None
        for i, distance in enumerate(spec['distances_m']):
            d = folder / phase / f'{i:02d}'
            r = json.loads((d / 'result.json').read_bytes())
            if (r['protocol_sha256'] != digest(PROTOCOL) or r['reference_sha256'] != spec['sha256']
                    or r['phase'] != phase or r['tick'] != spec['tick'] or r['distance_m'] != distance
                    or (i and r['before'] != baseline)):
                raise ValueError(f'unpaired branch: {d}')
            baseline = r['before']
            for name, sha in r['files'].items():
                if digest(d / name) != sha:
                    raise ValueError(f'frame hash mismatch: {d / name}')
            y = 94 + i * 266
            row = dict(phase=phase, distance_m=distance, action_accepted=r['motion_accepted'],
                       source_result_sha256=digest(d / 'result.json'))
            with np.load(d / 'before.npz', allow_pickle=False) as pre:
                canvas.paste(tile(pre['rgb'], project(pre, pre['truth_front'])), (8, y))
            if r['motion_accepted']:
                with np.load(d / 'after.npz', allow_pickle=False) as post:
                    uv, corners, ahead, near = regions(post)
                    diff = np.clip((post['no_cutter_depth'] - post['depth']) / .003, 0, 1)
                    delta = np.repeat((255 * diff).astype('uint8')[..., None], 3, axis=-1)
                    for c, data in enumerate((post['rgb'], post['no_cutter_rgb'], delta), start=1):
                        canvas.paste(tile(data, uv, corners if c == 3 else None), (8 + c*234, y))
                    row.update(front_neighborhood=measurements(post, near), ahead_patch=measurements(post, ahead),
                               actual_tcp_displacement_m=float(np.linalg.norm(post['tcp'][:, 3] - np.array(baseline['tcp'])[:, 3])))
                obs, cut = r['after_observation'], r['after_cut']
                row.update(mode=cut.get('mode'), reason=cut.get('reason'), progress_m=obs['cut_progress_m'],
                           scissors_visible=obs['scissors_visible'], fix_visible=obs['cut_site_visible'],
                           aligned=r['closure_aligned'], sweep_budget_m=r['sweep_budget_m'],
                           along_tip_minus_front_m=r['along_tip_minus_front_m'], cut_length_m=obs['cut_length_m'],
                           torn=obs['torn'], faults=obs['faults'])
            text = f"{distance*1000:g} mm   |   accepted={r['motion_accepted']}   |   " + (
                f"front={row['fix_visible']}  scissors={row['scissors_visible']}   |   {row['reason'] or row['mode']}"
                if r['motion_accepted'] else r['refusal'])
            dr.text((12, y+231), text, fill='white')
            out.append(row)
        with (folder / f'{phase}_pairs.png').open('xb') as f:
            canvas.save(f, format='PNG')
    with (folder / 'pairs.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(dict(protocol=protocol, scorer_sha256=digest(Path(__file__)), rows=out),
                  f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder', type=Path)
    score(ap.parse_args().folder)
