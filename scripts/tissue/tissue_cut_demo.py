"""Tissue model check in scenes/davinci_si_instruments.ttt (src/tissue/tissue.lua).

A. Grasper (PSM1, Maryland): a sheet is laid through its jaw line. Opening and closing the jaws must cut
   nothing; tissueGrasp pins nodes; retracting the instrument stretches the sheet; release frees it.
B. Scissors (PSM2, curved scissors): tissueGrasp is refused. PSM2 is brought into the endoscope view, a
   pre-tensioned sheet is laid in front of the blades, and the blades open, advance and close four times.
   The cut must separate the sheet along the blade line, with no torn constraints.
Writes validation/tissue/cut_demo.json and endoscope frames (validation/tissue/*.png, docs/tissue_cut_demo.png).
The scene is loaded from a staged copy and never saved.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'build'))
from simtools import (ENDO_CONFIG, INSTRUMENT_SCENE, REPO, TISSUE_VALIDATION, add_arguments, aliases, connect,  # noqa: E402
                      load_scene, matrix_to_np, read_json, relative, sha256, stop_simulation, write_json)

H = 0.001                       # node spacing [m]
PRETENSION = 0.08               # rest length 8 % below the laid-out spacing: the sheet is held under tension
SEPARATED = 1.25                # a row is cut through when the two nodes beside the blade line are > 1.25 H apart


def unit(v):
    return v / np.linalg.norm(v)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('--cycles', type=int, default=4, help='open-advance-close cycles of the scissors')
    p.add_argument('--advance-m', type=float, default=0.003, help='insertion advance per cycle')
    args = p.parse_args()
    rcm = read_json(ENDO_CONFIG / 'rcm_config.json')
    task = read_json(ENDO_CONFIG / 'task.json')
    endo = read_json(ENDO_CONFIG / 'endoscope.json')
    index = {a['arm']: i for i, a in enumerate(rcm, 1)}

    client, sim = connect(args)
    load_scene(client, sim, args, INSTRUMENT_SCENE, 'tissue_cut_demo.ttt')
    by = aliases(sim)
    ctrl = by['RCMController']
    cam = by[endo['camera']['alias']]
    W, Hh = endo['camera']['resolution']

    def call(name, *a):
        return sim.callScriptFunction(name, ctrl, *a)

    def steps(n):
        for _ in range(n):
            sim.step()

    def frame(arm):
        j9, j10 = (matrix_to_np(sim.getObjectMatrix(by[f'{arm}_joint{n}'], -1)) for n in (9, 10))
        x = unit(j10[:, 3] - j9[:, 3])
        return dict(pivot=j10[:, 3], x=x, axis=j10[:, 2], tcp=np.array(sim.getObjectPosition(by[f'{arm}_tcp'], -1)))

    def sheet(origin, u, v, nu, nv, sides=True):
        return call('tissueCreate', dict(origin=origin.tolist(), u=u.tolist(), v=v.tolist(), nu=nu, nv=nv, spacing=H,
                                         pretension=PRETENSION, anchor_sides=sides))

    def nodes(nu):
        return np.array(call('tissueNodes')).reshape(-1, nu, 3)

    def image():
        sim.handleVisionSensor(cam)
        raw, _ = sim.getVisionSensorImg(cam)
        return Image.frombytes('RGB', (W, Hh), bytes(raw)).transpose(Image.FLIP_TOP_BOTTOM)

    def closeup(centre, normal, up, dist=0.045):
        """Picture straight onto the sheet (the endoscope sees it nearly edge-on)."""
        from simtools import np_to_matrix
        fwd = -unit(normal)
        upv = unit(up - np.dot(up, fwd) * fwd)
        R = np.column_stack([np.cross(upv, fwd), upv, fwd])
        s = sim.createVisionSensor(1 + 2, [W, Hh, 0, 0], [0.005, 0.3, np.radians(40), .01, 0, 0, .1, .1, .12, 0, 0])
        sim.setObjectMatrix(s, -1, np_to_matrix(R, centre - fwd * dist))
        sim.handleVisionSensor(s)
        raw, _ = sim.getVisionSensorImg(s)
        sim.removeObjects([s])
        return Image.frombytes('RGB', (W, Hh), bytes(raw)).transpose(Image.FLIP_TOP_BOTTOM)

    def move(arm, d_ins, n=15):
        yaw, pitch, ins = call('getRCMStats')[index[arm] - 1]['target']
        ok = bool(call('setRCMTarget', index[arm], yaw, pitch, ins + d_ins))
        steps(n)
        return ok

    sim.setStepping(True)
    sim.setBoolParam(sim.boolparam_realtime_simulation, False)
    sim.startSimulation()
    sim.step()
    call('homeRCM')
    steps(20)
    result = {'scene': relative(INSTRUMENT_SCENE), 'scene_sha256': sha256(INSTRUMENT_SCENE), 'spacing_m': H,
              'pretension': PRETENSION}

    # A. Grasper: sheet through the jaw line of PSM1 (the jaws lie in the sheet).
    f = frame('psm1')
    nuA, nvA = 10, 36          # 35 mm long: the 21 mm jaws end about 10 mm before the attached far edge
    # Only the far edge is attached (tissue still fixed to the body there); the grasper pulls it towards the port.
    sheet(f['pivot'] - 0.004 * f['x'], f['axis'], f['x'], nuA, nvA, sides=False)
    steps(5)
    ok_open = bool(call('setGrip', index['psm1'], 0.5))
    steps(5)
    ok_close = bool(call('setGrip', index['psm1'], 0.0))
    steps(5)
    cut_by_grasper = call('tissueStats')['cut']
    grabbed = call('tissueGrasp', index['psm1'])
    strain0 = call('tissueStats')['max_strain']
    pulled = move('psm1', -0.0015, 20)
    sA = call('tissueStats')
    released = call('tissueRelease', index['psm1'])
    steps(20)
    result['grasper'] = dict(arm='psm1', jaws_opened=ok_open, jaws_closed=ok_close, constraints_cut_by_closing=cut_by_grasper,
                             nodes_grasped=grabbed, retract_1p5mm_accepted=pulled, max_strain_before_pull=strain0,
                             max_strain_after_pull=sA['max_strain'], torn=sA['torn'], nodes_released=released,
                             max_strain_after_release=call('tissueStats')['max_strain'])
    move('psm1', 0.0015, 20)
    call('tissueRemove')

    # B. Scissors: grasp refused; cut a pre-tensioned sheet in the endoscope view.
    i2 = index['psm2']
    for ready in task['planning']['command_sequence_checked'][:2]:
        call('setRCMTarget', i2, *map(float, ready))
    steps(30)
    f = frame('psm2')
    nuB, nvB = 16, 14
    # Sheet centre line on the blade line (pivot -> blade tip), between node columns 7 and 8; the sheet plane holds
    # the blade line and the jaw axis, so the opened blades straddle it. Near edge 2 mm in front of the blade tip.
    # v = advance direction (the shaft); the curved blade tip runs along the sheet centre line as it advances.
    v = f['x']
    u = unit(f['axis'] - np.dot(f['axis'], v) * v)
    sheet_origin = f['tcp'] + 0.002 * v
    sheet(sheet_origin, u, v, nuB, nvB)
    steps(30)
    centre = sheet_origin + 0.5 * (nvB - 1) * H * v
    normal = np.cross(u, v)
    before = image()
    before_close = closeup(centre, normal, v)
    refused = call('tissueGrasp', i2)
    n0 = nodes(nuB)
    cycles = []
    for c in range(args.cycles):
        o = bool(call('setGrip', i2, 0.5))
        steps(5)
        a = move('psm2', args.advance_m, 15)
        cut_before = call('tissueStats')['cut']
        fc = frame('psm2')
        cl = bool(call('setGrip', i2, 0.0))
        steps(10)
        # Blade line in sheet coordinates (u across the cut, v along it, from the sheet centre line / near edge).
        line = [[float(np.dot(q - sheet_origin, u)), float(np.dot(q - sheet_origin, v))] for q in (fc['pivot'], fc['tcp'])]
        cycles.append(dict(opened=o, advanced=a, closed=cl, constraints_cut=call('tissueStats')['cut'] - cut_before,
                           blade_line_uv_m=line))
    move('psm2', -args.cycles * args.advance_m, 30)
    steps(40)
    after = image()
    after_close = closeup(centre, normal, v)
    sB = call('tissueStats')
    n1 = nodes(nuB)
    mid = nuB // 2
    gap0 = np.linalg.norm(n0[:, mid] - n0[:, mid - 1], axis=1)
    gap1 = np.linalg.norm(n1[:, mid] - n1[:, mid - 1], axis=1)
    # The curved blades cut a slightly slanted, overlapping path: per row, the widest gap between neighbouring nodes.
    row_gap = np.linalg.norm(n1[:, 1:] - n1[:, :-1], axis=2)
    widest = row_gap.argmax(axis=1).tolist()
    gap1 = row_gap.max(axis=1)
    separated = []
    for j in range(nvB):                       # rows cut through, counted from the near edge
        if gap1[j] <= SEPARATED * H:
            break
        separated.append(j)
    result['scissors'] = dict(arm='psm2', grasp_result=refused, cycles=cycles, constraints_cut=sB['cut'], torn=sB['torn'],
                              constraints=sB['constraints'], max_strain_end=sB['max_strain'],
                              gap_along_cut_m=dict(before=gap0.tolist(), after=gap1.tolist()),
                              separated_rows=separated, cut_depth_m=len(separated) * H, widest_gap_column_per_row=widest,
                              cut_events=sB['cut_events'],
                              torn_events=[dict(e, a_ij=[e['a'] % nuB, e['a'] // nuB], b_ij=[e['b'] % nuB, e['b'] // nuB])
                                           for e in sB['torn_events']])
    stop_simulation(client, sim)

    g, s = result['grasper'], result['scissors']
    result['checks'] = {
        'grasper_does_not_cut': g['constraints_cut_by_closing'] == 0,
        'grasper_grasps_tissue': g['nodes_grasped'] > 0 and g['max_strain_after_pull'] > g['max_strain_before_pull'],
        'scissors_cannot_grasp': s['grasp_result'] == -1,
        'scissors_cut_separates_sheet': len(s['separated_rows']) >= 3 and 0 in s['separated_rows'],
        'no_tearing': g['torn'] == 0 and s['torn'] == 0,
    }
    result['ok'] = all(result['checks'].values())
    result['notes'] = ('Kinematic position-based sheet; the instruments do not collide with it. Cutting removes the constraints '
                       'within 0.3 mm of the closed blade line. No mass, gravity, contact force, bleeding or energy effect.')
    TISSUE_VALIDATION.mkdir(parents=True, exist_ok=True)
    before.save(TISSUE_VALIDATION / 'cut_before.png')
    after.save(TISSUE_VALIDATION / 'cut_after.png')
    before_close.save(TISSUE_VALIDATION / 'cut_before_closeup.png')
    after_close.save(TISSUE_VALIDATION / 'cut_after_closeup.png')
    sheet_img = Image.new('RGB', (2 * W, 2 * Hh + 48), 'white')
    d = ImageDraw.Draw(sheet_img)
    d.text((8, 6), 'Close-up onto the sheet, PSM2 curved scissors: before (left) and after (right) four cuts', fill='black')
    sheet_img.paste(before_close, (0, 24))
    sheet_img.paste(after_close, (W, 24))
    d.text((8, Hh + 30), 'Endoscope view of the same moments (the sheet is seen almost edge-on)', fill='black')
    sheet_img.paste(before, (0, Hh + 48))
    sheet_img.paste(after, (W, Hh + 48))
    sheet_img.save(REPO / 'docs' / 'tissue_cut_demo.png')
    write_json(TISSUE_VALIDATION / 'cut_demo.json', result)
    print('checks', result['checks'], 'cut', s['constraints_cut'], 'depth mm', s['cut_depth_m'] * 1000, flush=True)
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
