"""Offline yaw/pitch sensitivity to a translated non-anatomical specimen."""
import argparse
from pathlib import Path

import numpy as np

from replay_cell_v2 import DATA, ROOT, digest, load, write
from tool_pose import _MARGIN

PLAN = ROOT/'config/task/m5_v3_yaw_screen_v1.json'
ARCHIVE = DATA/'m5_guided_alignment_capture_v2_5'


def checked(path, pin):
    if digest(path) != pin:
        raise ValueError('changed seal: '+str(path))
    return path


def angles(points, rcm, basis):
    forward, yaw, pitch = np.moveaxis((points-rcm)@basis.T,-1,0)
    if not np.all(forward>0):
        raise ValueError('points must be distal to the RCM')
    return np.degrees(np.stack((np.arctan2(yaw,forward),np.arctan2(pitch,forward)),axis=-1))


def ranges(values, soft):
    low, high = values.reshape(-1,2).min(0), values.reshape(-1,2).max(0)
    return dict(yaw_deg=[float(low[0]),float(high[0])],pitch_deg=[float(low[1]),float(high[1])],
                min_yaw_headroom_deg=float(soft[0]-max(abs(low[0]),abs(high[0]))),
                min_pitch_headroom_deg=float(soft[1]-max(abs(low[1]),abs(high[1]))))


def screen(out, frames):
    plan = load(PLAN)
    canonical = checked(ARCHIVE/'decision.json',plan['alignment_capture_sha256'])
    report = load(canonical)
    rcm_source = checked(ROOT/plan['rcm'],plan['rcm_sha256'])
    arm, = (a for a in load(rcm_source) if a['arm']=='psm2')
    native_source = ROOT/'scripts/task/tool_pose.py'

    rcm = np.asarray(arm['rcm_world'])
    basis = np.array([arm[k] for k in ('axis_home','transverse_x','transverse_y')])
    if not np.allclose(basis@basis.T,np.eye(3),atol=1e-8):
        raise ValueError('expected an orthonormal calibrated RCM basis')
    soft = np.asarray([arm['limits'][k] for k in ('yaw_deg','pitch_deg')])-_MARGIN[:2]
    if tuple(soft)!=(4.5,4.5):
        raise ValueError('unexpected shaft soft limits')
    sources = {p.relative_to(ROOT).as_posix():digest(p) for p in
               (PLAN, Path(__file__).resolve(), canonical, rcm_source, native_source,
                ROOT/'config/rcm_config.json', ROOT/'src/endoscope/rcm_controller.lua',
                ROOT/'scripts/analysis/replay_cell_v2.py')}
    images = []
    view_data = []
    for view in plan['views']:
        entry = report['views'][view]
        folder = ARCHIVE/view
        reg = checked(folder/'registration.json',entry['registration_sha256'])
        item = checked(folder/'nominal.json',entry['capture_sha256'])
        r = load(item)
        if r['registration_sha256']!=digest(reg) or r['error'] or r['cleanup_error']:
            raise ValueError('unmatched trajectory: '+view)
        pinned = load(reg)['sources']
        # The historical registration lists config/rcm_config.json, not the
        # instrument-scene fallback used by SimWorld. Pin the latter in this screen.
        for p in (native_source,ROOT/'src/endoscope/rcm_controller.lua',
                  ROOT/'config/rcm_config.json'):
            checked(p,pinned[p.relative_to(ROOT).as_posix()])
        sources.update({p.relative_to(ROOT).as_posix():digest(p) for p in (reg,item)})
        data = {}
        for k in ('align1','closure1','closure2','closure3'):
            tag = entry['frames'][k]
            p = checked(frames/ARCHIVE.name/view/'nominal'/tag['file'],tag['sha256'])
            images.append(dict(view=view,stage=k,tick=tag['tick'],frame=p.relative_to(frames).as_posix(),
                               sha256=tag['sha256']))
            with np.load(p,allow_pickle=False) as f:
                if f['tick'].item()!=tag['tick']:
                    raise ValueError('capture tick changed')
                data[k] = {a:f[a] for a in ('nodes','tcp')}
        view_data.append((view,r,data))
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    lock = out/'registration.json'
    write(lock,dict(protocol=plan,sources=sources,frames=images,basis_world=basis.tolist(),
                    rcm_world_m=rcm.tolist(),soft_limits_deg=soft.tolist()))

    result = {}
    for view,r,data in view_data:
        row = data['align1']['nodes'][:11,11:13,:].mean(1)
        travel = row[-1]-row[0]
        travel /= np.linalg.norm(travel)
        targets = np.concatenate((row,[p-s*travel for p in (row[0],row[-1]) for s in (.001,.003)]))
        reach = np.linalg.norm(targets-rcm,axis=1)
        trials = np.vstack((np.zeros((1,3)),.0005*basis,-.0005*basis))
        cal = []
        for k, tick in zip(('closure1','closure2','closure3'),(8,14,24),strict=True):
            observed, = (img for img in images if img['view']==view and img['stage']==k)
            e, = (e for e in r['events'] if e['event']=='guided_v21_set_tool_pose' and
                  e['accepted'] and e['state']=='CUT' and e['tick']==observed['tick']-1)
            if observed['tick'] != tick:
                raise ValueError('closure was not at the registered tick')
            tcp = data[k]['tcp'][:,3]
            ray = angles(tcp,rcm,basis)
            cal.append(dict(tick=tick,previous_command_tick=e['tick'],
                            command_yaw_pitch_deg=e['q'][:2],tcp_ray_deg=ray.tolist(),
                            command_minus_ray_deg=(e['q'][:2]-ray).tolist()))
        offsets = np.array([c['command_minus_ray_deg'] for c in cal])
        shifts = []
        for mm in plan['shift_mm']:
            shifted = targets + mm*.001*basis[1]
            ray = angles(shifted,rcm,basis)
            perturbed = angles(shifted[:,None,:]+trials,rcm,basis)
            rays = dict(nominal=ranges(ray,soft),perturbed=ranges(perturbed,soft))
            nominal = ranges(ray[:,None,:]+offsets,soft)
            pert = ranges(perturbed[:,:,None,:]+offsets,soft)
            lo,hi = pert['yaw_deg']
            qualified = (-2.5<=lo<=hi<=1. and pert['min_pitch_headroom_deg']>=0)
            shifts.append(dict(shift_mm=mm,translation_world_m=(mm*.001*basis[1]).tolist(),ray=rays,
                               offset_screen=dict(nominal=nominal,perturbed=pert),
                               coarse_candidate=qualified))
        result[view] = dict(target_count=len(targets),perturbations_per_target=len(trials),
                            rcm_to_specimen_point_m=[float(reach.min()),float(reach.max())],
                            calibration=cal,calibration_offset_range_deg=ranges(offsets,soft),
                            rows_world_m=row.tolist(),shifts=shifts)
    common = [mm for mm in plan['shift_mm'] if all(
        next(s['coarse_candidate'] for s in v['shifts'] if s['shift_mm']==mm) for v in result.values())]
    decision = dict(registration_sha256=digest(lock),views=result,common_coarse_candidates_mm=common,
                    verified_set_tool_pose=False,verified_collision_free_path=False,authorizes_motion=False)
    write(out/'decision.json',decision)
    for view,v in result.items():
        print(view,v['rcm_to_specimen_point_m'],v['calibration_offset_range_deg'])
        for s in v['shifts']:
            print(s['shift_mm'],s['offset_screen']['perturbed'],s['coarse_candidate'])
    print('common ray candidates (not full IK):',common)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    screen(a.out,a.frames)
