"""Render close-ups of the three PSM instrument tips of scenes/davinci_si_instruments.ttt into docs/si_instrument_tips.png.

Anatomy and the table are hidden for the pictures only; the scene file is not saved. Each tip is shown with its
jaws closed and opened by the panel's 0.5 rad.
"""
import argparse
import math

import numpy as np
from PIL import Image, ImageDraw

import instrument_guard as guard
from simtools import INSTRUMENT_SCENE, REPO, add_arguments, aliases, connect, load_scene, np_to_matrix, relative

OUT = REPO / 'docs' / 'si_instrument_tips.png'
LABELS = {'psm1': 'PSM1  420172 Maryland Bipolar', 'psm2': 'PSM2  420179 Monopolar Curved Scissors',
          'psm3': 'PSM3  420093 ProGrasp'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    args = p.parse_args()
    client, sim = connect(args)
    load_scene(client, sim, args, INSTRUMENT_SCENE, 'render_tips.ttt')
    by = aliases(sim)
    for n, h in by.items():
        if sim.getObjectType(h) == sim.sceneobject_shape and not n.startswith('psm'):
            sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
    W, H = 480, 360
    tiles = []
    for arm in ('psm1', 'psm2', 'psm3'):
        row = []
        for q in (0.0, 0.25):
            for n in (10, 11):
                sim.setJointPosition(by[f'{arm}_joint{n}'], q)
            T, _ = guard.wrist_frame(sim, by, arm)
            x, y, z = T[:3, 0], T[:3, 1], T[:3, 2]       # distal, jaw axis, pitch axis
            tcp = np.array(sim.getObjectPosition(by[f'{arm}_tcp'], -1))
            focus = 0.5 * (T[:3, 3] + tcp)
            eye = focus + 1.9 * np.linalg.norm(tcp - T[:3, 3]) * unit(y + 0.35 * z - 0.25 * x)
            fwd = unit(focus - eye)
            up = unit(np.cross(np.cross(fwd, x), fwd))
            R = np.column_stack([np.cross(up, fwd), up, fwd])
            s = sim.createVisionSensor(1 + 2, [W, H, 0, 0], [0.005, 0.3, math.radians(45), .01, 0, 0, .95, .95, .95, 0, 0])
            sim.setObjectMatrix(s, -1, np_to_matrix(R, eye))
            sim.handleVisionSensor(s)
            img, _ = sim.getVisionSensorImg(s)
            sim.removeObjects([s])
            row.append(Image.frombytes('RGB', (W, H), bytes(img)).transpose(Image.FLIP_TOP_BOTTOM))
        tiles.append(row)
    sheet = Image.new('RGB', (2 * W, 3 * H + 3 * 28), 'white')
    d = ImageDraw.Draw(sheet)
    for r, (arm, row) in enumerate(zip(('psm1', 'psm2', 'psm3'), tiles)):
        top = r * (H + 28)
        d.text((8, top + 8), LABELS[arm] + '   (left: closed, right: open 0.5 rad)', fill='black')
        for c, im in enumerate(row):
            sheet.paste(im, (c * W, top + 28))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    print('saved', relative(OUT))


def unit(v):
    return v / np.linalg.norm(v)


if __name__ == '__main__':
    main()
