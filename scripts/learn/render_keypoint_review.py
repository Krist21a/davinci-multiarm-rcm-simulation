"""Render unmarked + crosshair review panels from the frozen keypoint annotation audit.

Mechanical plate generation only: rows are taken verbatim from audit['visual_examples']
in original order (no sampling/dedup/reorder). Each panel is 2x2 cells of 640x508 on a
1280x1016 RGB canvas -- a 28 px black title strip ('#N  ep##### / tick ###') over the
unmodified 640x480 frame. The overlay copy adds only crosshairs: cyan on the rounded tip
uv_px, lime on the rounded front uv_px, in bounds only; missing points draw nothing.
panels.json maps image titles to source frames. No classes or statuses are rendered.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TITLE_H = 28
CELL_W, CELL_H = 640, 480 + TITLE_H
CYAN, LIME, WHITE, BLACK = (0, 255, 255), (0, 255, 0), (255, 255, 255), (0, 0, 0)


def crosshair(draw, uv, color, ox, oy):
    if uv is None:
        return
    u, v = map(round, uv)
    if not (0 <= u < CELL_W and 0 <= v < CELL_H - TITLE_H):
        return
    u, v = u + ox, v + oy + TITLE_H
    segs = [((u - 12, v), (u - 4, v)), ((u + 4, v), (u + 12, v)),
            ((u, v - 12), (u, v - 4)), ((u, v + 4), (u, v + 12))]
    for seg in segs:
        draw.line(seg, fill=BLACK, width=3)
    for seg in segs:
        draw.line(seg, fill=color, width=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='fresh output directory')
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f'refusing to reuse existing directory: {args.out}')

    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    root = Path(audit['root'])
    rows = [(cat, row) for cat, group in audit['visual_examples'].items()
            for row in group]
    args.out.mkdir(parents=True)
    panels = []
    for panel, start in enumerate(range(0, len(rows), 4), start=1):
        clean = Image.new('RGB', (2 * CELL_W, 2 * CELL_H))
        overlaid = Image.new('RGB', (2 * CELL_W, 2 * CELL_H))
        for slot, (cat, row) in enumerate(rows[start:start + 4]):
            seq = start + slot + 1
            ox, oy = (slot % 2) * CELL_W, (slot // 2) * CELL_H
            with np.load(root / row['rgbd'], allow_pickle=False) as z:
                rgb = z['rgb']
            img = Image.fromarray(rgb)
            for canvas, mark in ((clean, False), (overlaid, True)):
                d = ImageDraw.Draw(canvas)
                d.text((ox + 4, oy + 7),
                       f"#{seq}  ep{row['index']:05d} / tick {row['tick']:03d}",
                       fill=WHITE)
                tile = img.copy() if mark else img
                canvas.paste(tile, (ox, oy + TITLE_H))
                if mark:
                    dd = ImageDraw.Draw(canvas)
                    crosshair(dd, (row.get('tip') or {}).get('uv_px'), CYAN, ox, oy)
                    crosshair(dd, (row.get('front') or {}).get('uv_px'), LIME, ox, oy)
            panels.append({'panel': seq, 'plate': panel, 'category': cat, 'rgbd': row['rgbd']})
        clean.save(args.out / f'review_{panel:02d}_clean.png')
        overlaid.save(args.out / f'review_{panel:02d}_overlay.png')
    with (args.out / 'panels.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(panels, f, indent=1)
        f.write('\n')
    print(json.dumps({'rows': len(rows), 'panels': len(rows) // 4 + bool(len(rows) % 4),
                      'out': str(args.out.resolve())}))


if __name__ == '__main__':
    main()
