"""Seal and score one independent offline confirmation of the fixed tip gate."""
import argparse
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

import collect
from tip_baseline import baseline
from tip_data import (CALIBRATION, LOCK, NEGATIVES, POSITIVE, REPO, SCENARIOS, VARIANTS,
                      _open_capture, input_depth, read_label, read_pixels, sha, write_json)
from tip_marker_gate_dev import gates, patch_features, suppress
from tip_score import summarize
from throughput_probe import SIM_EXE, port_open

PLAN = REPO / 'config/learn/tip_marker_gate_confirm_v1.json'
MANIFEST_ROOT = REPO / 'validation/task/learned_perception_v1'
LEARNED_DIR = 'learned_perception_v1'


def source_digests(plan):
    names = (PLAN, Path(__file__), CALIBRATION, LOCK,
             *(REPO / p for p in (
                 'scripts/learn/tip_data.py', 'scripts/learn/tip_score.py',
                 'scripts/learn/tip_baseline.py', 'scripts/learn/throughput_probe.py',
                 'scripts/learn/audit_keypoints.py', 'scripts/learn/phase_a_index.py',
                 'scripts/task/visibility.py', 'scripts/vision/cut_perception.py',
                 plan['parent']['protocol'], plan['parent']['gate_source'],
             )))
    return {p.relative_to(REPO).as_posix(): sha(p.read_bytes()) for p in names}


def acquisition_args(plan, data_root):
    c = plan['cohort']
    return ['scripts/learn/collect.py', '--split', c['sampler_namespace'],
            '--start', str(c['start']), '--count', str(c['count']), '--workers', str(c['workers']),
            '--base-port', str(c['base_port']), '--data-root', data_root.as_posix(),
            '--tag', c['tag'], '--no-aux']


def assert_fresh(data_root, plan):
    tag = plan['cohort']['tag']
    paths = (data_root / LEARNED_DIR / tag, MANIFEST_ROOT / tag, MANIFEST_ROOT / f'{tag}.json')
    if any(p.exists() for p in paths):
        raise ValueError('collector tag has already been attempted; no rerun')


def parameters(plan):
    c = plan['cohort']
    proto = collect.protocol()
    for start in range(412, 10000, 40):
        rows = [collect.episode_params('dev', i, proto) for i in range(start, start + 40)]
        scenario = Counter(r['scenario'] for r in rows)
        if scenario['occlusion_transient'] >= 16 and all(
                scenario[n] >= 5 for n in SCENARIOS if n != 'occlusion_transient'):
            if (start, len(rows), scenario, c['sampler_namespace']) != (
                    c['start'], c['count'], c['expected_scenarios'], 'dev'):
                raise ValueError('locked cohort differs from the first eligible block')
            return [dict(r, seed=str(r['seed'])) for r in rows]
    raise ValueError('no eligible block')


def register(data_root, seal):
    plan = json.loads(PLAN.read_bytes())
    assert_fresh(data_root, plan)
    parent = plan['parent']
    dev = data_root / LEARNED_DIR / parent['development_report']
    parent_files = ((dev, parent['development_report_sha256']),
                    (REPO / parent['protocol'], parent['protocol_sha256']),
                    (REPO / parent['gate_source'], parent['gate_source_sha256']))
    if any(sha(p.read_bytes()) != digest for p, digest in parent_files):
        raise ValueError('development artifact mismatch')
    decision = json.loads(dev.read_bytes())
    if not decision['gates']['depth_rule']['passes'] or decision['learned_development_benefit']:
        raise ValueError('development did not select the fixed rule')
    request = parameters(plan)
    reserved = json.loads(LOCK.read_bytes())['splits']['independent_evaluation']
    old = [json.loads(p.read_bytes()) for p in MANIFEST_ROOT.glob('*/ep[0-9]*.json')]
    indices = {r['index'] for r in old + reserved}
    seeds = {str(r['seed']) for r in old + reserved}
    if any(r['index'] in indices or r['seed'] in seeds for r in request):
        raise ValueError('confirmation overlaps earlier or reserved episodes')
    if len({r['seed'] for r in request}) != len(request):
        raise ValueError('duplicate seed within confirmation')
    frozen = {
        'protocol': plan, 'episodes': request, 'collector_argv': acquisition_args(plan, data_root),
        'source_sha256': source_digests(plan), 'collector_sha256': collect.source_hashes(collect.protocol()),
        'simulator_exe_sha256': sha(SIM_EXE.read_bytes()),
        'score_versions': {'python': platform.python_version(), 'numpy': np.__version__,
                           'scipy': scipy.__version__},
    }
    seal.mkdir()
    write_json(seal / 'registration.json', frozen)
    print(f"registered {len(request)} episodes; registration SHA256 {sha((seal / 'registration.json').read_bytes())}")
    print(json.dumps(frozen['collector_argv']))


def verify_seal(seal, data_root):
    reg = json.loads((seal / 'registration.json').read_bytes())
    plan = reg['protocol']
    if (reg['source_sha256'] != source_digests(plan)
            or reg['collector_sha256'] != collect.source_hashes(collect.protocol())
            or reg['simulator_exe_sha256'] != sha(SIM_EXE.read_bytes())
            or reg['collector_argv'] != acquisition_args(plan, data_root)
            or reg['episodes'] != parameters(plan)
            or reg['score_versions'] != {'python': platform.python_version(), 'numpy': np.__version__,
                                         'scipy': scipy.__version__}):
        raise ValueError('frozen registration no longer matches sources, recipe or environment')
    return reg


def preflight(data_root, seal):
    reg = verify_seal(seal, data_root)
    plan = reg['protocol']
    assert_fresh(data_root, plan)
    p = plan['cohort']['base_port']
    if any(port_open(port) for port in range(p, p + plan['cohort']['workers'])):
        raise ValueError('a dedicated RPC port is already in use')
    print('seal, fresh collector outputs and worker ports OK')


def joined_frames(data_root, reg):
    c = reg['protocol']['cohort']
    tag = c['tag']
    image_root = data_root / LEARNED_DIR / tag
    summary_raw = (MANIFEST_ROOT / f'{tag}.json').read_bytes()
    summary = json.loads(summary_raw)
    episodes = [e for w in summary['workers'] for e in w['episodes']]
    manifests = {e['index']: e for e in episodes}
    wanted = [e['index'] for e in reg['episodes']]
    source = reg['collector_sha256']
    if (summary['indices'] != wanted or len(episodes) != len(wanted) or sorted(manifests) != wanted
            or summary.get('episodes') != len(wanted) or any(w.get('error') for w in summary['workers'])
            or summary['tag'] != tag or summary['split'] != c['sampler_namespace']
            or summary['aux_views'] != 0 or summary['hd_twin'] or summary['base_port'] != c['base_port']
            or summary['source_sha256'] != source):
        raise ValueError('collector summary is incomplete or discordant')
    rows = []
    for e in reg['episodes']:
        index = e['index']
        name = f'ep{index:05d}'
        manifest = manifests[index]
        disk = image_root / name
        raw = (disk / 'manifest.json').read_bytes()
        if (raw != (MANIFEST_ROOT / tag / f'{name}.json').read_bytes()
                or json.loads(raw) != manifest):
            raise ValueError(f'{name}: manifest copies differ')
        if (manifest['status'] not in ('done', 'failed') or manifest['error']
                or manifest.get('cleanup_error') or int(e['seed']) != manifest['seed']
                or any(e[k] != val for k, val in manifest['sampled'].items())
                or e['aux_seed'] != manifest['aux_seed']
                or manifest['tag'] != tag or manifest['split'] != c['sampler_namespace']
                or manifest['aux_views'] != 0 or manifest['hd_twin']
                or manifest['source_sha256'] != source):
            raise ValueError(f'{name}: refused, infrastructure error or incorrect sampling')
        n = manifest['ticks']
        files = {f['name']: f for f in manifest['files']}
        names = [f'tick_{tick:03d}.npz' for tick in range(1, n + 1)]
        if (n <= 0 or len(manifest['trace']) != n or len(manifest['files']) != n
                or manifest['frames_by_kind'] != dict(main=n, hd=0, aux=0)
                or set(files) != set(names) or {f.name for f in disk.glob('*.npz')} != set(names)):
            raise ValueError(f'{name}: wrong frame or trace counts')
        for tick, fname, trace in zip(range(1, n + 1), names, manifest['trace'], strict=True):
            if trace['tick'] != tick or (disk / fname).stat().st_size != files[fname]['bytes']:
                raise ValueError(f'{name}: frame, manifest and trace disagree')
            rows.append({'episode': index, 'seed': e['seed'], 'scenario': e['scenario'],
                         'episode_status': manifest['status'], 'tick': tick, 'rgbd': f'{name}/{fname}',
                         'sha256': files[fname]['sha256'], 'trace_length_m': trace['cut_length_m'],
                         'manifest_sha256': sha(raw)})
    if len(rows) != summary['frames']:
        raise ValueError('total frame count differs from collector summary')
    return image_root, rows, sha(summary_raw), summary['counts_by_status']


def timing(root, row):
    with _open_capture(root, row) as cap:
        if ('label_cut_length' in cap) != ('label_nodes' in cap):
            raise ValueError('partial structural label')
        length = float(cap['label_cut_length'].item()) if 'label_cut_length' in cap else None
    if not math.isclose(0. if length is None else length, row['trace_length_m'], rel_tol=0, abs_tol=1e-9):
        raise ValueError('structural timing label and trace disagree')
    return 'pre_sheet' if length is None else 'loaded_uncut' if length == 0 else 'cut'


def paired_forecasts(root, row, tcp):
    marker = baseline(root, row, tcp)
    _, depth, intr = read_pixels(root, row)
    fixed = {'rgbd': row['rgbd']}
    for v in VARIANTS:
        features = patch_features(input_depth(depth, intr[3], v, row), intr, marker[v])
        kept = features is not None and features[-1] >= -3
        fixed[v] = dict(suppress(marker[v], kept), patch_available=features is not None,
                        max_centre_relative_depth_mm=None if features is None else features[-1])
    return marker, fixed


def evaluate(data_root, seal, reg):
    root, rows, summary_sha, statuses = joined_frames(data_root, reg)
    tcp = json.loads(CALIBRATION.read_bytes())['locator']['tcp_local_m']
    baseline_rows, fixed_rows = [], []
    pairs = {v: {} for v in VARIANTS}
    for row in rows:
        left, right = paired_forecasts(root, row, tcp)
        row['timing'] = timing(root, row)
        row['label'] = read_label(root, row)
        key = row['label']['status']
        for v in VARIANTS:
            given, kept = left[v]['xyz_m'] is not None, right[v]['xyz_m'] is not None
            if kept and not given:
                raise RuntimeError('fixed rule must not add a prediction')
            pairs[v].setdefault(key, Counter())['retained' if kept else 'suppressed' if given
                                                else 'baseline_abstained'] += 1
        baseline_rows.append(left)
        fixed_rows.append(right)
    predictions = seal / 'predictions.json'
    write_json(predictions, {'rows': rows, 'baseline': baseline_rows, 'fixed_rule': fixed_rows})
    metric0, metric1 = summarize(rows, baseline_rows), summarize(rows, fixed_rows)
    blocked = sorted({r['episode'] for r in rows if r['label']['status'] == 'depth_blocked'})
    scenarios = Counter(r['scenario'] for r in rows if r['label']['status'] == POSITIVE)
    coverage = {
        'at_least_four_blocked_episodes': len(blocked) >= 4,
        'positive_support_in_every_scenario': all(scenarios[s] for s in SCENARIOS),
        'marker_outputs_on_blocked_in_each_variant':
            all(metric0[v]['all']['accepted_blocked'] > 0 for v in VARIANTS),
    }
    qualified = gates(metric1)
    status = 'incomplete' if not all(coverage.values()) else 'pass' if qualified['passes'] else 'fail'
    return {
        'confirmation': status, 'collector_summary_sha256': summary_sha, 'episode_status': statuses,
        'predictions_sha256': sha(predictions.read_bytes()), 'episodes': len(reg['episodes']),
        'blocked_episodes': blocked, 'scenario_positive_frames': scenarios, 'coverage_checks': coverage,
        'baseline': metric0, 'fixed_rule': metric1, 'fixed_rule_gates': qualified,
        'paired_decisions_by_label': pairs,
    }


def score(data_root, seal):
    reg = verify_seal(seal, data_root)
    plan = reg['protocol']
    summary = MANIFEST_ROOT / f"{plan['cohort']['tag']}.json"
    if not summary.exists() or 'episodes' not in json.loads(summary.read_bytes()):
        raise ValueError('collector has not written a finalized summary')
    digest = sha((seal / 'registration.json').read_bytes())
    write_json(seal / 'score_attempt.json', {'registration_sha256': digest, 'protocol': plan})
    try:
        report = evaluate(data_root, seal, reg)
    except Exception as exc:
        report = {'confirmation': 'incomplete', 'error': f'{type(exc).__name__}: {exc}'}
    write_json(seal / 'report.json', dict(report, registration_sha256=digest, protocol=plan))
    print(f"confirmation: {report['confirmation']}")
    if 'error' in report:
        print(report['error'])
    return int(report['confirmation'] == 'incomplete')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('register', 'preflight', 'score'))
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--seal', type=Path, required=True)
    args = p.parse_args()
    return {'register': register, 'preflight': preflight, 'score': score}[
        args.action](args.data_root, args.seal)


if __name__ == '__main__':
    sys.exit(main())
