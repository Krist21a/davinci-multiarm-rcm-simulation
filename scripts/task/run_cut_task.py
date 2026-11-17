"""Run the cutting-task state machine (cut_fsm.py) in scenes/davinci_si_instruments.ttt.

  python run_cut_task.py --mode cut_only --scenario nominal
  python run_cut_task.py --mode cut_only --scenario occlusion_transient
  python run_cut_task.py --mode cut_only --scenario occlusion_persistent
  python run_cut_task.py --mode full --scenario nominal

Writes validation/task/sim_<mode>_<scenario>.json and one endoscope frame per state change
(validation/task/sim_<mode>_<scenario>/). The scene is loaded from a staged copy and never saved.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'build'))
from cut_fsm import CutTaskFSM  # noqa: E402
from sim_world import SimWorld  # noqa: E402
from simtools import INSTRUMENT_SCENE, REPO, add_arguments, connect, relative, sha256, stop_simulation, write_json  # noqa: E402

OUT = REPO / 'validation' / 'task'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('--mode', choices=['cut_only', 'full'], default='cut_only')
    p.add_argument('--scenario', choices=['nominal', 'occlusion_transient', 'occlusion_persistent'], default='nominal')
    p.add_argument('--sheet', action='store_true', help='also write a contact sheet to docs/fsm_<mode>_<scenario>.png')
    args = p.parse_args()
    cfg = json.loads((REPO / 'config' / 'task' / 'cut_task.json').read_text(encoding='utf-8'))
    client, sim = connect(args)
    world = SimWorld(client, sim, args, cfg, mode=args.mode, event=None if args.scenario == 'nominal' else args.scenario)
    fsm = CutTaskFSM(world, cfg, cut_only=args.mode == 'cut_only')
    shots_at = []                                   # endoscope image seen when each transition was decided
    base_go = fsm.go

    def go(new, reason, obs=None):
        shots_at.append(world.image)
        base_go(new, reason, obs)
    fsm.go = go
    try:
        fsm.run()
        final = world.observe()
        stats = world.call('tissueStats')
    finally:
        stop_simulation(client, sim)
    name = f'sim_{args.mode}_{args.scenario}'
    fdir = OUT / name
    fdir.mkdir(parents=True, exist_ok=True)
    for old in fdir.glob('*.png'):
        old.unlink()
    shots = []
    for k, (e, im) in enumerate(zip([e for e in fsm.log if 'to' in e], shots_at)):
        im = (im or Image.new('RGB', (world.W, world.H))).copy()
        ImageDraw.Draw(im).text((8, 8), f"tick {e['tick']}  {e['frm']} -> {e['to']}", fill=(255, 255, 255))
        im.save(fdir / f"{k:02d}_{e['to']}.png")
        shots.append((e, im))
    if args.sheet and shots:
        cols, tw, th = 4, 320, 240
        rows = (len(shots) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * tw, rows * (th + 18)), 'white')
        d = ImageDraw.Draw(sheet)
        for k, (e, im) in enumerate(shots):
            x, y = (k % cols) * tw, (k // cols) * (th + 18)
            d.text((x + 4, y + 3), f"{k + 1}. {e['to']}  ({e['reason'][:38]})", fill='black')
            sheet.paste(im.resize((tw, th)), (x, y + 18))
        sheet.save(REPO / 'docs' / f'fsm_{args.mode}_{args.scenario}.png')
    rep = fsm.report()
    write_json(OUT / f'{name}.json', dict(
        scene=relative(INSTRUMENT_SCENE), scene_sha256=sha256(INSTRUMENT_SCENE), mode=args.mode, scenario=args.scenario,
        config=cfg, roles=getattr(world, 'roles_report', None), **rep,
        final_observation=dict(cut_length_m=final.cut_length_m, tension=final.tension, max_strain=final.max_strain, torn=final.torn),
        tissue=dict(cut=stats.get('cut'), torn=stats.get('torn'), cut_events=stats.get('cut_events'), torn_events=stats.get('torn_events')),
        camera_moves=world.camera_moves,
        notes='cut_length_m comes from the tissue node geometry (rows separated along the cut line), not from vision. '
              'Visibility uses the rendered endoscope depth buffer.'))
    print(name, rep['final_state'], rep['reason'])
    print(' > '.join(e['to'] for e in fsm.log if 'to' in e))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
