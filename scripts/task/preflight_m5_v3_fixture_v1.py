"""Guarded, non-cutting reachability preflight for the selected two-axis fixture."""
import argparse
from pathlib import Path
import json
import uuid

import numpy as np

from run_cut_task import REPO, add_arguments, connect, sha256, stop_simulation
from sim_world import SimWorld, workspace, GRASP_RADIUS_M
from workspace import NU, NV, H
import tool_pose_active_set

ID = 'm5_v3_fixture_preflight_v1'
SPEC = REPO/f'config/task/{ID}.json'
GRID = REPO/'validation/task/m5_v3_distal_workspace_v2'
BASE = REPO/'config/task/cut_task.json'

def load(p):
    return json.loads(p.read_bytes())

def write(p,obj):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,indent=2,allow_nan=False,default=lambda v:np.asarray(v).tolist())
        f.write('\n')

class Fixture(SimWorld):
    def place_sheet(self):
        assert self.canonical_seam and self.roi_proxy is None
        f = self.frame(self.cutter)
        g = workspace.layout(f['pivot'],f['x'],f['axis'],f['tcp'])
        self.u,self.v = g['u'],g['v']
        d = np.asarray(self.args.fixture_translation_m)
        self.sheet_origin = g['origin'] + d
        self.grasp_points = {k:p+d for k,p in g['grasp_points'].items()}
        self.call('tissueCreate',dict(origin=self.sheet_origin.tolist(),u=self.u.tolist(),
            v=self.v.tolist(),nu=NU,nv=NV,spacing=H,pretension=self.pretension,
            anchor_sides=False,anchor_from_row=3,grasp_radius=GRASP_RADIUS_M,
            seam_mode='surface',guide_band_m=self.sheet_cfg['guide_band_m']))
        self.steps(30)
        self.cutline_rest_m = np.asarray(self.call('tissueTransverseRestLengths'),dtype=float)

    def open_scissors(self):
        raise RuntimeError('preflight forbids cutter jaw commands')

    close_scissors = open_scissors


def run(a):
    p = load(SPEC)
    reg,decision = GRID/'registration.json',GRID/'decision.json'
    if (sha256(reg)!=p['workspace_registration_sha256'] or sha256(decision)!=p['workspace_decision_sha256']
            or load(decision)['registration_sha256']!=sha256(reg)):
        raise ValueError('wrong workspace seal')
    c = load(decision)['selected']
    if ((c['delta_yaw_deg'],c['delta_pitch_deg'])!=(2.,1.5) or not c['passes']
            or p['minimum_headroom_deg']!=1.5
            or not all(min(v['minimum_soft_headroom'][:2])>=1.5 for v in c['by_view'].values())):
        raise ValueError('not the approved two-axis candidate')
    a.fixture_translation_m = c['translation_m']
    a.scene,a.initial_view,a.initial_view_insertion_m = REPO/p['scene'],p['view'][:2],p['view'][2]
    a.run_id,a.scenario = uuid.uuid4().hex,'nominal'
    cfg = load(BASE)
    cfg['sheet'] = p['fixture']
    cfg['perception'].update(mode='shadow',markers=True,exclude_tcp_marker=True)
    a.out.mkdir(exist_ok=False)
    inputs = (SPEC,Path(__file__).resolve(),reg,decision,BASE,a.scene,
              *(REPO/h for h in ('scripts/task/sim_world.py','scripts/task/workspace.py',
                'scripts/task/tool_pose.py','scripts/task/tool_pose_active_set.py',
                'scripts/build/simtools.py','src/endoscope/rcm_controller.lua',
                'src/instruments/articulation.lua','src/tissue/tissue.lua','src/tissue/surface_seam.lua',
                'config/instruments/rcm_config.json','config/endoscope/task.json')))
    lock=a.out/'registration.json'
    write(lock,dict(plan=p,candidate=c,sources={k.relative_to(REPO).as_posix():sha256(k) for k in inputs}))
    r=dict(passed=False,authorizes_motion=False,registration_sha256=sha256(lock),steps=[],failure=None,cleanup_error=None)
    client=sim=w=None
    try:
        client,sim = connect(a)
        w=Fixture(client,sim,a,cfg,mode='full')
        ok,why=w.check_roles()
        if not ok or not w.home() or not w.jaws_closed():
            raise RuntimeError('roles/home/closed readback refused: '+why)
        initial_edges=np.asarray(w.call('tissueConstraints'))
        for arm in (w.primary,w.assist):
            ok,why=w.grasp(arm,w.grasp_points[arm])
            r['steps'].append(dict(stage='grasp',arm=arm,accepted=ok,reason=why))
            if not ok:
                raise RuntimeError(arm+' grasp: '+why)
        for arm in (w.primary,w.assist):
            if not w.retract(arm,p['pull_per_arm_m']):
                raise RuntimeError(arm+' load refused')
        def seen(stage):
            o=w.observe(inject_events=False)
            est=w._last_est or w._estimate()
            vis=bool(est['scissors']['valid'] and est['cut']['valid'])
            closed=w.jaws_closed()
            unchanged=np.array_equal(initial_edges,np.asarray(w.call('tissueConstraints')))
            holds=all(o.grasped.get(arm,0)>=cfg['grasp']['min_nodes'] for arm in (w.primary,w.assist))
            safe=bool(holds and closed and unchanged and not o.faults and not o.torn and o.cut_length_m==0
                      and cfg['tension']['low_strain']<=o.tension<=cfg['tension']['high_strain'])
            r['steps'].append(dict(stage=stage,visible=vis,closed=closed,structure_unchanged=unchanged,
                                   tension=o.tension,faults=list(o.faults),grasped=o.grasped,safe=safe))
            return safe,vis
        safe,visible=seen('loaded')
        if not (safe and visible):
            raise RuntimeError('loaded no-cut or vision gate refused')
        arm=w.cutter
        ready,_=w._tool_q(arm)
        path=[ready]
        for mm in p['waypoints_mm']:
            q0,ws=w._tool_q(arm)
            lo,hi=w._tool_limits(arm,ws)
            x=w.sheet_origin+.001*mm*w.v
            solve=tool_pose_active_set.solve(w._tool_fk(arm),q0,(lo,hi),x,w.v,w.u,
                                            axis_col=w.TOOL_AXIS_COL,yaw_col=w.TOOL_YAW_COL)
            q=solve['q']
            room=4.5-np.abs(q[:2])
            r['steps'].append(dict(stage='ik',mm=mm,q=q,headroom_deg=room,converged=solve['converged'],
                                   pos_err_m=solve['pos_err_m'],axis_err_rad=solve['axis_err_rad'],
                                   opening_err_rad=solve['yaw_axis_err_rad']))
            if not solve['converged'] or min(room)<p['minimum_headroom_deg'] or not w.jaws_closed():
                raise RuntimeError(f'closed workspace pose refused at {mm} mm')
            if not w.tool_command(arm,q):
                raise RuntimeError(f'guarded command refused at {mm} mm')
            actual,_=w._tool_q(arm)
            if not np.allclose(actual,q,atol=1e-6,rtol=0):
                raise RuntimeError('tool readback mismatch')
            path.append(q)
            safe,visible=seen(f'work_{mm}mm')
            if not (safe and visible):
                raise RuntimeError(f'waypoint evidence refused at {mm} mm')
        for q in reversed(path[:-1]):
            if not w.jaws_closed() or not w.tool_command(arm,q):
                raise RuntimeError('guarded reverse refused')
        safe,_=seen('returned')
        r['passed']=bool(safe and w.at_ready_pose())
        if not r['passed']:
            raise RuntimeError('ready pose not verified')
    except Exception as e:
        r['failure']=f'{type(e).__name__}: {e}'
        if w:
            w.hold()
    finally:
        if sim:
            try:
                if stop_simulation(client,sim) is False:
                    raise RuntimeError('simulator did not stop')
            except Exception as e:
                r['cleanup_error']=f'{type(e).__name__}: {e}'
                r['passed']=False
        write(a.out/'decision.json',r)
    print(r['passed'],r['failure'],r['cleanup_error'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
