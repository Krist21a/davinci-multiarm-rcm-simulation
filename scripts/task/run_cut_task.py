"""Run the cutting-task state machine (cut_fsm.py) in scenes/davinci_si_instruments.ttt.

  python run_cut_task.py --mode cut_only --scenario nominal
  python run_cut_task.py --mode cut_only --scenario occlusion_transient
  python run_cut_task.py --mode cut_only --scenario occlusion_persistent
  python run_cut_task.py --mode full --scenario nominal
  python run_cut_task.py --mode full --scenario slip_regrasp
  python run_cut_task.py --mode full --scenario slip_handover
  python run_cut_task.py --mode full --scenario relax
  python run_cut_task.py --mode full --scenario relax_large
  python run_cut_task.py --mode full --scenario handover_tension

Writes validation/task/sim_<mode>_<scenario>.json and one endoscope frame per state change
(validation/task/sim_<mode>_<scenario>/). The scene is loaded from a staged copy and never saved.
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'build'))
from cut_fsm import CutTaskFSM, S, completion_issues  # noqa: E402
from sim_world import SimWorld  # noqa: E402
from simtools import ARTICULATED_SCENE as INSTRUMENT_SCENE  # noqa: E402
from simtools import REPO, add_arguments, connect, relative, sha256, stop_simulation, write_json  # noqa: E402

OUT = REPO / 'validation' / 'task'

# Injected-event ticks are world ticks (one per observe() call), chosen from the nominal full-mode
# timeline: CUT runs at world ticks ~9-15, so slips land inside it after the first closure, a lone
# relax drops tension mid-CUT, and the post-handover relax comes a few ticks after the takeover.
SLIP_TICK = 13
RELAX_TICK = 11
HANDOVER_RELAX_TICK = 24


def scenario_events(cfg, scenario):
    """Injected-event spec per scenario, in the SimWorld/FakeWorld dict semantics (sim_world._events)."""
    r = cfg.get('roles', {})
    primary, assist = r.get('retract_primary', 'psm1'), r.get('retract_assist', 'psm3')
    return {
        'nominal': {},
        'occlusion_transient': {'occlude': (6, 8)},
        'occlusion_persistent': {'occlude_until_camera': 6},
        'slip_regrasp': {'slip': {primary: SLIP_TICK}},
        'slip_handover': {'slip': {primary: SLIP_TICK, assist: SLIP_TICK}, 'regrasp_fails': [primary]},
        'relax': {'relax': RELAX_TICK},
        'relax_large': {'relax': RELAX_TICK, 'relax_m': 0.0015},   # exceeds the recovery budget on purpose
        'handover_tension': {'slip': {primary: SLIP_TICK, assist: SLIP_TICK},
                             'regrasp_fails': [primary], 'relax': HANDOVER_RELAX_TICK},
    }[scenario]


def write_report(path, report):
    """Publish a complete JSON file, never a partially written replacement of an old run."""
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        write_json(temporary, report)
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass                              # do not mask a write/replace failure


def save_frames(args, name, world, fsm, shots_at):
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('--mode', choices=['cut_only', 'full'], default='cut_only')
    p.add_argument('--scenario', choices=['nominal', 'occlusion_transient', 'occlusion_persistent',
                                          'slip_regrasp', 'slip_handover', 'relax', 'relax_large',
                                          'handover_tension'],
                   default='nominal')
    p.add_argument('--scene', type=Path, default=INSTRUMENT_SCENE, help='instrument scene to load without saving')
    p.add_argument('--sheet', action='store_true', help='also write a contact sheet to docs/fsm_<mode>_<scenario>.png')
    args = p.parse_args()
    run_id = uuid.uuid4().hex
    name = f'sim_{args.mode}_{args.scenario}'
    report = dict(run_id=run_id, success=False, mode=args.mode, scenario=args.scenario,
                  error=None, cleanup_error=None, errors=[], acceptance_issues=[], controller_sha256=None)
    client = sim = world = fsm = None
    shots_at = []                              # endoscope image seen when each transition was decided
    accepted = False

    def record_error(stage, exc):
        message = f'{stage}: {type(exc).__name__}: {exc}'
        report['errors'].append(message)
        if report['error'] is None:
            report['error'] = message
        return message

    stage = 'configuration'
    try:
        cfg = json.loads((REPO / 'config' / 'task' / 'cut_task.json').read_text(encoding='utf-8'))
        report['config'] = cfg
        stage = 'source provenance'
        scene = args.scene.resolve()
        report.update(scene=relative(scene), scene_sha256=sha256(scene),
                      source_sha256={relative(path): sha256(path) for path in
                                     [HERE / 'cut_fsm.py', HERE / 'sim_world.py', HERE / 'workspace.py',
                                      HERE / 'ground_truth.py', HERE / 'visibility.py',
                                      Path(__file__).resolve(), REPO / 'config' / 'task' / 'cut_task.json']})
        stage = 'connect'
        client, sim = connect(args)
        # Construction starts the simulator; even partial initialization must reach cleanup/reporting.
        stage = 'world initialization'
        events = scenario_events(cfg, args.scenario)
        world = SimWorld(client, sim, args, cfg, mode=args.mode, events=events)
        report.update(controller_sha256=getattr(world, 'controller_sha256', None),
                      roles=getattr(world, 'roles_report', None), events=events)
        stage = 'FSM initialization'
        fsm = CutTaskFSM(world, cfg, cut_only=args.mode == 'cut_only')
        base_go = fsm.go

        def go(new, reason, obs=None):
            shots_at.append(world.image)
            base_go(new, reason, obs)
        fsm.go = go
        stage = 'FSM run'
        fsm.run()
        stage = 'final observation'
        final = world.observe()
        stage = 'final tissue statistics'
        stats = world.call('tissueStats')
        stage = 'final acceptance'
        issues = completion_issues(final, cfg)
        report.update(acceptance_issues=issues,
                      final_observation=dict(cut_length_m=final.cut_length_m, tension=final.tension, max_strain=final.max_strain,
                                             torn=final.torn, faults=final.faults,
                                             overcut_m=max(0.0, final.cut_length_m - cfg['cut']['length_m'])),
                      tissue=dict(cut=stats.get('cut'), torn=stats.get('torn'), cut_events=stats.get('cut_events'),
                                  torn_events=stats.get('torn_events')),
                      camera_moves=world.camera_moves,
                      notes='cut_length_m comes from the tissue node geometry (rows separated along the cut line), not from vision. '
                            'Visibility uses the rendered endoscope depth buffer.')
        accepted = fsm.state == S.DONE and not issues
    except Exception as exc:
        record_error(stage, exc)
    finally:
        if sim is not None:
            try:
                stop_simulation(client, sim)
            except Exception as exc:
                report['cleanup_error'] = f'{type(exc).__name__}: {exc}'

    if world is not None:
        try:
            instrumentation = dict(refused_candidates=world.refused_failures,
                                   event_log=world.event_log,
                                   truth_reads=world.truth.reads,
                                   timing=world.timing_summary())
            # A duck-typed world without instrumentation contributes nothing rather than failing the report.
            if all(isinstance(v, (dict, list)) for v in instrumentation.values()):
                report.update(instrumentation)
        except Exception as exc:
            record_error('world instrumentation', exc)
    if fsm is not None:
        try:
            report.update(fsm.report())
        except Exception as exc:
            record_error('FSM report', exc)
        try:
            save_frames(args, name, world, fsm, shots_at)
        except Exception as exc:
            record_error('screenshots', exc)
    report['success'] = bool(accepted and report['error'] is None and report['cleanup_error'] is None)
    path = OUT / f'{name}.json'
    try:
        write_report(path, report)
    except Exception as exc:
        report['success'] = False
        report['report_error'] = record_error('report write', exc)
        # A serialization failure must still leave a current, small failure report if storage works.
        failure = {key: report[key] for key in ('run_id', 'success', 'mode', 'scenario', 'error', 'cleanup_error', 'report_error')}
        try:
            write_report(path, failure)
        except Exception as fallback_exc:
            print(f'cannot publish report: {type(fallback_exc).__name__}: {fallback_exc}; {json.dumps(failure)}', file=sys.stderr)
    print(name, report.get('final_state', 'ERROR'), report['error'] or report['cleanup_error'] or report.get('reason', ''))
    print(' > '.join(e['to'] for e in report.get('log', []) if 'to' in e))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
