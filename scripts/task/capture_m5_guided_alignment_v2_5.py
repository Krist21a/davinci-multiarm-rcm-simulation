"""Passive, scoring-only capture on the frozen v2.5 development trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np

import run_m5_guided_benchmark_v2_5 as candidate
import run_m5_guided_view_preflight_v2_5 as pre
from run_cut_task import REPO, add_arguments, sha256
from simtools import matrix_to_np

ID = 'm5_guided_alignment_capture_v2_5'
ARCHIVE = REPO/'validation/task/m5_guided_view_preflight_v2_5'
DECISION_SHA256 = '7fd95f41ae404e6a35bf6d393a522fc343ab270dbc5221413dc3b8c0b858d9b4'
STAGES = {
    5: 'align1', 7: 'advance1', 8: 'closure1', 9: 'after1',
    11: 'align2', 13: 'advance2', 14: 'closure2', 15: 'after2',
    21: 'align3', 23: 'advance3', 24: 'closure3', 25: 'after3',
}


def instrument(w):
    pre.instrument(w)
    observe = w.observe
    manifest = {}

    def sample(**kwargs):
        obs = observe(**kwargs)
        tick = w.tick
        if tick not in STAGES:
            return obs
        before = w.sim.getSimulationTime()
        scis = (w._last_est or {}).get('scissors') or {}
        cut = (w._last_est or {}).get('cut') or {}
        fix = w._fix if w._fix_valid() else None
        nan3 = np.full(3,np.nan)
        with w.truth.purpose('scoring'):
            nodes = w.truth.nodes()
            edges = np.asarray(w.truth._read('constraints',lambda:w.call('tissueConstraints')))
            tcp = matrix_to_np(w.truth._read('tcp_frame',lambda:w.sim.getObjectMatrix(
                w.by[w.cutter+'_tcp'],-1)))
        R,t = w.camera()
        if w.tick!=tick or w.sim.getSimulationTime()!=before:
            raise ValueError('passive capture changed time or tick')
        path = w.args.evidence_out/f'{tick:03d}_{STAGES[tick]}.npz'
        with path.open('xb') as f:
            np.savez_compressed(
                f,rgb=np.asarray(w.image),depth=w.depth,cam_R=R,cam_t=t,focal=w.f,far=w.far,
                tcp=tcp,nodes=nodes,edges=edges,sheet_u=w.u,sheet_v=w.v,jaw_axis_col=w.TOOL_AXIS_COL,
                visual_tip=scis.get('tip',nan3) if scis.get('valid') else nan3,
                visual_jaw=scis.get('jaw_axis',nan3) if scis.get('valid') else nan3,
                visual_front=cut.get('front',nan3) if cut.get('front_valid') else nan3,
                fix_front=fix['front'] if fix else nan3,fix_direction=fix['direction'] if fix else nan3,
                fix_normal=fix['normal'] if fix else nan3,fix_tick=fix['tick'] if fix else -1,
                tick=tick,sim_time=before)
        manifest[STAGES[tick]] = dict(file=path.name,sha256=sha256(path),tick=tick)
        if tick==max(STAGES):
            candidate.parent.put(w.args.evidence_out/'alignment_frames.json',manifest)
        return obs

    w.observe = sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--view',choices=('original','plus3'),required=True)
    for key in ('out','evidence-out'):
        p.add_argument('--'+key,type=Path,required=True)
    a = p.parse_args()
    decision = ARCHIVE/'decision.json'
    if sha256(decision)!=DECISION_SHA256:
        raise ValueError('development decision changed')
    frozen = json.loads(decision.read_bytes())
    reference = ARCHIVE/a.view/'nominal.json'
    view_reg = ARCHIVE/a.view/'registration.json'
    prior = json.loads(reference.read_bytes())
    sources = json.loads(view_reg.read_bytes())['sources']
    if (prior['registration_sha256']!=sha256(view_reg) or
            frozen['views'][a.view]['cases']['nominal']['report_sha256']!=sha256(reference) or
            any(sha256(REPO/k)!=h for k,h in sources.items())):
        raise ValueError('unsealed reference trajectory')
    plan = json.loads(pre.PARENT.read_bytes())
    plan['initial_view'] = frozen['views'][a.view]['pose']
    a.case = 'nominal'
    a.out,a.evidence_out = a.out/a.view,a.evidence_out/a.view
    files = (Path(__file__).resolve(),decision,reference,view_reg,*(REPO/k for k in sources))
    base = candidate.parent.register(a.out,ID+'/'+a.view,files)
    return candidate.run_case(a,plan,base,sha256(a.out/'registration.json'),instrument=instrument)


if __name__=='__main__':
    raise SystemExit(main())
