"""Native endoscope RGB-D metrology in a stopped disposable scene; no robot commands."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from run_m5_v3_outer_positive_v5 import REPO, add_arguments, check, connect, dedicated, listening, load, sha256, write
from simtools import ARTICULATED_SCENE, aliases, load_scene

ID = 'm5_v4_plane_calibration_v1'
PLAN = REPO/f'config/task/{ID}.json'
OUT = REPO/f'validation/task/{ID}'
AUDIT = REPO/'validation/task/m5_v4_bend_rim_audit_v1'
REG = REPO/f'validation/safety/regressions_{ID}.json'
W,H = 1280,960
F = W*.5/np.tan(np.deg2rad(35.))
COLORS = np.array([[29,209,197],[222,47,169]])


def grid():
    j,i = np.indices((7,9))
    return .004*np.column_stack(((i-4).ravel(),(j-3).ravel(),np.zeros(i.size)))


def register():
    spec = load(PLAN)
    check(sha256(ARTICULATED_SCENE)==spec['scene_sha256'] and
          sha256(AUDIT/'decision.json')==spec['audit_sha256'],'source experiment drift')
    tests = load(REG)
    check(tests['success'] and not tests['skipped'] and tests['tests_run']>=497,'offline suite refused')
    paths = [PLAN,Path(__file__).resolve(),REPO/'scripts/analysis/score_m5_v4_plane_v1.py',
             REPO/'scripts/tests/test_m5_v4_plane_v1.py',REPO/'scripts/build/simtools.py',
             REPO/'scripts/task/run_m5_v3_outer_positive_v5.py',REPO/'scripts/analysis/audit_m5_v4_bend_rim_v1.py']
    sources = {}
    for p in paths:
        k = p.relative_to(REPO).as_posix()
        sources[k] = sha256(p)
        check(sources[k]==tests['source_sha256'][k],'source not covered by regressions: '+k)
    for p in (REG,ARTICULATED_SCENE,AUDIT/'decision.json',AUDIT/'support.json',AUDIT/'registration.json'):
        sources[p.relative_to(REPO).as_posix()] = sha256(p)
    check(sources[(AUDIT/'support.json').relative_to(REPO).as_posix()]==load(AUDIT/'decision.json')['support_sha256'],
          'unbound transfer pixels')
    OUT.mkdir(exist_ok=False)
    write(OUT/'registration.json',dict(spec=spec,sources=sources))
    return spec


def acquire(s,cam,folder,name,pose):
    M = np.asarray(s.getObjectMatrix(cam,-1)).reshape(3,4)
    z,col,row,rx,ry,rz = pose
    center = z*.001*np.array([(W*.5-.5-col)/F,(H*.5-.5-row)/F,1.])
    expected = grid()@Rotation.from_euler('xyz',[rx,ry,rz],degrees=True).as_matrix().T+center
    handles,refs,indices = [],[],[]
    try:
        for parity,color in enumerate(COLORS):
            tris = []
            for j in range(6):
                for i in range(8):
                    if (i+j)%2==parity:
                        a = 9*j+i
                        tris.extend([[a,a+10,a+1],[a,a+9,a+10]])
            ids,remap = np.unique(np.ravel(tris),return_inverse=True)
            vertices = expected[ids]@M[:,:3].T+M[:,3]
            h = s.createMeshShape(0,0,vertices.ravel().tolist(),remap.ravel().tolist())
            handles.append(h)
            for prop in (s.colorcomponent_ambient_diffuse,s.colorcomponent_specular):
                s.setShapeColor(h,None,prop,[0.,0.,0.])
            s.setShapeColor(h,None,s.colorcomponent_emission,(color/255.).tolist())
            s.setObjectInt32Param(h,s.shapeintparam_static,1)
            s.setObjectInt32Param(h,s.shapeintparam_respondable,0)
            verts,faces = s.getShapeMesh(h)[:2]
            A = np.asarray(s.getObjectMatrix(h,-1)).reshape(3,4)
            actual = (np.asarray(verts).reshape(-1,3)@A[:,:3].T+A[:,3]-M[:,3])@M[:,:3]
            dist = np.linalg.norm(actual[:,None]-expected[None,:],axis=-1)
            assoc = dist.argmin(1)
            check(np.max(dist.min(1))<2e-6 and
                  sorted(tuple(sorted(t)) for t in assoc[np.reshape(faces,(-1,3))])==
                  sorted(tuple(sorted(t)) for t in tris),'mesh readback differs')
            refs.extend(actual)
            indices.extend(assoc)
        ref = np.asarray(refs)
        readback = np.stack([ref[np.asarray(indices)==i].mean(0) for i in range(63)])
        s.handleVisionSensor(cam)
        rgb,sz = s.getVisionSensorImg(cam)
        depth,sz2 = s.getVisionSensorDepth(cam,1)
        check(list(sz)==list(sz2)==[W,H] and np.array_equal(M,np.reshape(s.getObjectMatrix(cam,-1),(3,4))),
              'exposure geometry changed')
        path = folder/(name+'.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream,rgb=np.frombuffer(bytes(rgb),np.uint8).reshape(H,W,3)[::-1],
                                depth=np.frombuffer(bytes(depth),'<f4').reshape(H,W)[::-1],
                                cam=M,vertices_camera=readback,mesh_vertices_camera=ref,mesh_indices=indices,
                                focal=F,pose=np.asarray(pose))
        return dict(file=path.name,sha256=sha256(path))
    finally:
        if handles:
            s.removeObjects(handles)


def batch(a,spec):
    with dedicated(a):
        c,s = connect(a)
        cam = None
        try:
            check(s.getSimulationState()==s.simulation_stopped,'not a stopped instance')
            with tempfile.TemporaryDirectory(dir=a.stage_dir) as tmp:
                a.stage_dir = Path(tmp)
                load_scene(c,s,a,ARTICULATED_SCENE,'plane_reference.ttt')
            orig = aliases(s)['EndoscopeCamera']
            before = s.getObjectMatrix(orig,-1)
            stamp = s.getSimulationTime()
            cam, = s.copyPasteObjects([orig],0)
            check(cam!=orig,'clone not created')
            s.setObjectParent(cam,-1,True)
            s.setObjectMatrix(cam,-1,[1,0,0,0,0,1,0,0,0,0,1,3])
            param = s.visionfloatparam_perspective_angle
            s.setObjectFloatParam(cam,param,float(np.deg2rad(70)))
            s.setIntArray2Property(cam,'resolution',[W,H])
            params = [s.visionfloatparam_near_clipping,s.visionfloatparam_far_clipping]
            clipping = [s.getObjectFloatParam(cam,k) for k in params]
            check(clipping==[s.getObjectFloatParam(orig,k) for k in params]
                  and np.allclose(clipping,[.002,.3],atol=1e-6,rtol=0) and list(s.getVisionSensorRes(cam))==[W,H]
                  and abs(s.getObjectFloatParam(cam,param)-np.deg2rad(70))<1e-6,'wrong clone optics')
            result = dict(registration_sha256=sha256(OUT/'registration.json'),clipping=clipping,frames={})
            for split in ('development','holdout'):
                result['frames'][split] = {}
                for i,pose in enumerate(spec[split]):
                    name = f'{split}_{i:02d}'
                    result['frames'][split][name] = acquire(s,cam,OUT,name,pose)
                    check(s.getSimulationTime()==stamp and s.getSimulationState()==s.simulation_stopped and
                          s.getObjectMatrix(orig,-1)==before,'scene or original camera moved')
            return result
        finally:
            try:
                if cam is not None:
                    s.removeObjects([cam])
            finally:
                c.socket.close(linger=0)
                c.context.term()


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.set_defaults(host='127.0.0.1',port=24763)
    p.add_argument('--launch-simulator',type=Path,required=True)
    args = p.parse_args()
    check(not listening() and args.launch_simulator.is_file(),'simulator or dedicated port unavailable')
    spec = register()
    args.out = OUT
    try:
        captured = batch(args,spec)
    except Exception as exc:
        write(OUT/'error.json',dict(reason=str(exc),port_free=not listening()))
        raise
    captured['port_free'] = not listening()
    write(OUT/'capture.json',captured)
    print('closed stationary capture', {k:len(v) for k,v in captured['frames'].items()},flush=True)
