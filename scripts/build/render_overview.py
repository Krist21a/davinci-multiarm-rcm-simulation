"""Render an overview of scenes/davinci_si_instruments.ttt (whole system and a close view of the ports)
into docs/overview_si_instruments.png and docs/ports_si_instruments.png. The scene file is not saved."""
import argparse
import math

import numpy as np
from PIL import Image

from simtools import INSTRUMENT_SCENE, REPO, add_arguments, aliases, connect, load_scene, np_to_matrix, relative


VIEW = (2.6, -1.2, 1.7)

def unit(v):
    return v / np.linalg.norm(v)


def shot(sim, eye, target, out, W=1280, H=800, fov=50, far=6.0):
    fwd = unit(target - eye)
    up = unit(np.array([0, 0, 1.0]) - np.dot([0, 0, 1.0], fwd) * fwd)
    R = np.column_stack([np.cross(up, fwd), up, fwd])
    s = sim.createVisionSensor(1 + 2, [W, H, 0, 0], [0.01, far, math.radians(fov), .02, 0, 0, .12, .12, .14, 0, 0])
    sim.setObjectMatrix(s, -1, np_to_matrix(R, eye))
    sim.handleVisionSensor(s)
    img, _ = sim.getVisionSensorImg(s)
    sim.removeObjects([s])
    Image.frombytes('RGB', (W, H), bytes(img)).transpose(Image.FLIP_TOP_BOTTOM).save(out)
    print('saved', relative(out))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--try-views', action='store_true', help='write small candidate views docs/_view_*.png instead')
    args = p.parse_args()
    client, sim = connect(args)
    load_scene(client, sim, args, INSTRUMENT_SCENE, 'overview.ttt')
    by = aliases(sim)
    for n in ('TaskBead', 'TaskTarget', 'psm2_TipMarker'):
        if n in by:
            sim.setObjectInt32Param(by[n], sim.objintparam_visibility_layer, 0)
    ports = np.array([sim.getObjectPosition(by[a + '_RCM_reference'], -1) for a in ('psm1', 'psm2', 'psm3', 'ecm')])
    c = ports.mean(axis=0)
    views = {'a': (2.6, -1.2, 1.7), 'b': (-2.2, -1.9, 1.6), 'c': (2.4, 1.9, 1.8), 'd': (-2.4, 1.6, 1.7)}
    if args.try_views:
        for k, v in views.items():
            shot(sim, c + np.array(v), c + np.array([0.0, 0.2, -0.2]), REPO / 'docs' / f'_view_{k}.png', W=640, H=400, fov=45)
        return
    shot(sim, c + np.array(VIEW), c + np.array([0.0, 0.2, -0.2]), REPO / 'docs' / 'overview_si_instruments.png', fov=45)
    shot(sim, c + np.array([0.30, 0.22, 0.32]), c + np.array([0.0, 0.02, -0.03]), REPO / 'docs' / 'ports_si_instruments.png', fov=45)


if __name__ == '__main__':
    main()
