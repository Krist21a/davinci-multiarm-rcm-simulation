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
import hashlib
import json
import math
import sys
import uuid
from pathlib import Path

import numpy as np
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


# Development flags form the registered prefix chain; only these three implications are
# real prerequisites. Everything else is a boolean type check.
PERCEPTION_FLAG_CHAIN = [('material_pattern', '1_5'), ('local_material_chart', '1_6'),
                         ('front_recovery', '1_7'), ('predictive_view', '1_8'),
                         ('exclude_tcp_marker', '1_9'), ('recovery_view_after_cut', '1_9_1'),
                         ('ray_marker_fit', '1_10')]


def resolve_perception_flags(cfg):
    """Apply defaults and enforce the real prerequisite chain for the development flags."""
    perc = cfg.setdefault('perception', {})
    perc.setdefault('material_pattern', cfg['cut'].get('control_source') == 'vision')
    if type(perc['material_pattern']) is not bool:
        raise ValueError('perception.material_pattern must be a boolean')
    perc.setdefault('local_material_chart', perc['material_pattern'])
    if type(perc['local_material_chart']) is not bool or (perc['local_material_chart'] and not perc['material_pattern']):
        raise ValueError('local_material_chart must be boolean and requires material_pattern')
    perc.setdefault('front_recovery', perc['local_material_chart'])
    if type(perc['front_recovery']) is not bool:
        raise ValueError('front_recovery must be a boolean')
    tool_pose_mode = cfg['cut'].get('align_mode', 'shaft') == 'tool_pose'
    perc.setdefault('predictive_view', perc['front_recovery'] and tool_pose_mode)
    if type(perc['predictive_view']) is not bool or (perc['predictive_view'] and not tool_pose_mode):
        raise ValueError('predictive_view must be boolean and requires tool_pose')
    perc.setdefault('exclude_tcp_marker', perc['predictive_view'])
    if type(perc['exclude_tcp_marker']) is not bool:
        raise ValueError('exclude_tcp_marker must be a boolean')
    perc.setdefault('recovery_view_after_cut', False)
    if type(perc['recovery_view_after_cut']) is not bool:
        raise ValueError('recovery_view_after_cut must be a boolean')
    perc.setdefault('markers', True)
    if type(perc['markers']) is not bool:
        raise ValueError('markers must be a boolean')
    perc.setdefault('ray_marker_fit', False)
    if type(perc['ray_marker_fit']) is not bool or (perc['ray_marker_fit'] and not perc['markers']):
        raise ValueError('ray_marker_fit requires markers')
    return perc


def protocol_provenance(perc):
    """(effective flag dict, registered version) or (flags, None) for custom development.

    Only an exact prefix of the historical flag chain may claim a protocol version."""
    flags = {flag: bool(perc.get(flag)) for flag, _ in PERCEPTION_FLAG_CHAIN}
    chain = [flag for flag, _ in PERCEPTION_FLAG_CHAIN]
    enabled = [flag for flag in chain if flags[flag]]
    if enabled != chain[:len(enabled)]:
        return flags, None
    return flags, (PERCEPTION_FLAG_CHAIN[len(enabled) - 1][1] if enabled else '1_4')


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
    p.add_argument('--trace', action='store_true',
                   help='record per-tick tension, cut length and a tissue-node hash in the report')
    p.add_argument('--align-mode', choices=['shaft', 'tool_pose'], default=None,
                   help='override cut.align_mode for this run (default: config value)')
    p.add_argument('--control-source', choices=['truth', 'vision'], default=None,
                   help='override cut.control_source for this run (default: config value)')
    p.add_argument('--perception', default=None,
                   help='JSON object merged into cfg["perception"] (mode/markers/depth_noise_m/camera_pose_error/seed)')
    p.add_argument('--initial-view', default=None, metavar='YAW,PITCH',
                   help='absolute initial camera angles in degrees, applied after task homing '
                        '(same convention as CAMERA_POSES)')
    p.add_argument('--initial-view-insertion-m', type=float,
                   help='optional absolute ECM insertion at --initial-view, subject to the existing guard')
    p.add_argument('--name', default=None, help='report name override (default sim_<mode>_<scenario>)')
    p.add_argument('--occlusion-probe', metavar='DATASET', help='registered M5a v6 passive RGB-D probe capture')
    args = p.parse_args()
    if args.initial_view_insertion_m is not None and (
            args.initial_view is None or not math.isfinite(args.initial_view_insertion_m)):
        p.error('--initial-view-insertion-m requires --initial-view and a finite value')
    if args.initial_view is not None:
        try:
            args.initial_view = [float(v) for v in args.initial_view.split(',')]
            if len(args.initial_view) != 2 or not all(map(math.isfinite, args.initial_view)):
                raise ValueError
        except ValueError:
            p.error('--initial-view must be finite YAW,PITCH in degrees')
    run_id = uuid.uuid4().hex
    args.run_id = run_id
    name = args.name or f'sim_{args.mode}_{args.scenario}'
    args.report_name = name                          # recorded in the perception frame-dump meta.json
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
        if args.align_mode:
            cfg['cut']['align_mode'] = args.align_mode
        if args.control_source:
            cfg['cut']['control_source'] = args.control_source
        if args.perception:
            cfg.setdefault('perception', {}).update(json.loads(args.perception))
        perc = resolve_perception_flags(cfg)
        if args.occlusion_probe:
            v6 = REPO / 'config' / 'perception' / 'm5a_amendment_v6.json'
            probes = json.loads(v6.read_bytes())['occlusion_probes']
            spec = next((s for s in probes['datasets'].values() if s['dataset'] == args.occlusion_probe), None)
            perc = cfg.get('perception', {})
            pose_error = perc.get('camera_pose_error') or {}
            if (spec is None or args.mode != 'full' or args.scenario != spec['scenario']
                    or args.initial_view != spec.get('initial_view') or cfg['cut']['align_mode'] != spec['align_mode']
                    or args.initial_view_insertion_m is not None
                    or cfg['cut'].get('control_source', 'truth') != 'truth' or perc.get('mode') != 'shadow'
                    or perc.get('material_pattern') or not perc.get('markers', True) or perc.get('depth_noise_m', 0)
                    or any(perc.get(f) for f in ('local_material_chart', 'front_recovery', 'predictive_view',
                                                 'exclude_tcp_marker', 'recovery_view_after_cut', 'ray_marker_fit'))
                    or pose_error.get('rot_deg', 0) or pose_error.get('trans_m', 0)
                    or perc.get('dump_dir') != f'validation/perception/frames/{args.occlusion_probe}'):
                raise ValueError('occlusion probe arguments differ from the registered capture')
            args.occlusion_probe = dict(spec, **{k: probes[k] for k in ('ticks', 'distance_m', 'radius_m')})
            report['occlusion_probe_protocol_sha256'] = sha256(v6)
        report['config'] = cfg
        report['initial_view'] = args.initial_view
        if args.initial_view_insertion_m is not None:
            report['initial_view_insertion_m'] = args.initial_view_insertion_m
        report['control_source'] = cfg['cut'].get('control_source', 'truth')
        # Report protocol flag provenance whenever any perception setting is nondefault —
        # including truth-control runs without material_pattern, which previously reported
        # nothing and hid custom flag combinations.
        flags, version = protocol_provenance(perc)
        pose_err = perc.get('camera_pose_error') or {}
        nondefault = (any(flags[f] for f in flags) or not perc.get('markers', True)
                      or perc.get('depth_noise_m', 0)
                      or pose_err.get('rot_deg', 0) or pose_err.get('trans_m', 0))
        if report['control_source'] == 'vision' or nondefault:
            if version is not None:
                protocol_path = REPO / 'config' / 'task' / f'm5b_protocol_v{version}.json'
                report['protocol_amendment'] = json.loads(protocol_path.read_text(encoding='utf-8'))
                report['protocol_amendment_sha256'] = sha256(protocol_path)
            else:
                # A non-prefix combination is custom development: report the effective flags
                # explicitly and never a misleading protocol version.
                report['protocol_amendment'] = dict(
                    custom_development=True,
                    note='non-prefix flag combination; no registered protocol version applies',
                    perception_flags=flags)
        stage = 'source provenance'
        scene = args.scene.resolve()
        report.update(scene=relative(scene), scene_sha256=sha256(scene),
                      source_sha256={relative(path): sha256(path) for path in
                                     [HERE / 'cut_fsm.py', HERE / 'sim_world.py', HERE / 'workspace.py',
                                      HERE / 'ground_truth.py', HERE / 'visibility.py', HERE / 'tool_pose.py', HERE / 'cutter_view.py',
                                      HERE.parent / 'vision' / 'cut_perception.py',
                                      HERE.parent / 'vision' / 'material_coordinate.py',
                                      Path(__file__).resolve(), REPO / 'config' / 'task' / 'cut_task.json']})
        stage = 'connect'
        client, sim = connect(args)
        # Construction starts the simulator; even partial initialization must reach cleanup/reporting.
        stage = 'world initialization'
        events = scenario_events(cfg, args.scenario)
        world = SimWorld(client, sim, args, cfg, mode=args.mode, events=events)
        if perc['exclude_tcp_marker']:
            report['tip_marker_optics'] = world.tip_marker_optics
        report.update(controller_sha256=getattr(world, 'controller_sha256', None),
                      roles=getattr(world, 'roles_report', None), events=events)
        stage = 'FSM initialization'
        fsm = CutTaskFSM(world, cfg, cut_only=args.mode == 'cut_only')
        base_go = fsm.go

        def go(new, reason, obs=None):
            shots_at.append(world.image)
            base_go(new, reason, obs)
        fsm.go = go
        if args.trace:
            trace = []
            observe = world.observe

            def traced_observe(**kwargs):
                obs = observe(**kwargs)
                nodes_sha = hashlib.sha256(np.asarray(world.call('tissueNodes'), dtype=float)
                                           .tobytes()).hexdigest()[:16]
                trace.append(dict(tick=world.tick, tension=obs.tension, cut_length_m=obs.cut_length_m,
                                  nodes_sha256=nodes_sha,
                                  protected_mm=getattr(world, 'last_protected', None)))
                return obs
            world.observe = traced_observe
            report['trace'] = trace
        stage = 'FSM run'
        fsm.run()
        stage = 'final observation'
        final = world.observe(inject_events=False)
        stage = 'final tissue statistics'
        stats = world.call('tissueStats')
        if cfg.get('perception', {}).get('mode') == 'shadow' and cfg['perception'].get('dump_dir'):
            stage = 'frame capture manifest'
            world.finish_capture()
        if args.initial_view is not None:
            report['initial_view_readback'] = world.initial_view_readback
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
                                   truth_reads_by_purpose=world.truth.reads_by_purpose,
                                   timing=world.timing_summary(),
                                   perception=getattr(world, 'perc_log', []),
                                   # Model distances in the synthetic scene, NOT clinical
                                   # safety distances.
                                   protected_min_mm=world.protected_min()
                                   if hasattr(world, 'protected_min') else {})
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
