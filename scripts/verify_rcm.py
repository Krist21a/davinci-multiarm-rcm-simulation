"""Independently measure live shaft-to-fixed-pivot residuals without moving joints."""
import argparse
import json
import math
import time
from pathlib import Path


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def cross(a,b):
    return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='localhost')
    p.add_argument('--port',type=int,default=23000)
    p.add_argument('--samples',type=int,default=100)
    p.add_argument('--interval',type=float,default=.05)
    p.add_argument('--tolerance',type=float,default=1e-6,help='Maximum residual in metres')
    p.add_argument('--start',action='store_true')
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.samples<1 or args.interval<0 or args.tolerance<=0:p.error('Invalid sampling or tolerance value')
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    client=RemoteAPIClient(host=args.host,port=args.port)
    sim=client.require('sim')
    config=json.loads((Path(__file__).resolve().parents[1]/'config/rcm_config.json').read_text(encoding='utf-8'))
    handles=[sim.getObject('/'+a['arm']+'_joint8') for a in config]
    refs=[sim.getObject('/'+a['arm']+'_RCM_reference') for a in config]
    sim.getObject('/RCMController')  # Fail before starting if this is another scene.
    if sim.getSimulationState()==sim.simulation_stopped:
        if not args.start:raise SystemExit('RCM scene is stopped; start it or pass --start.')
        sim.startSimulation()
    maxima=[0.0]*3;reference_drift=[0.0]*3;records=[]
    for k in range(args.samples):
        for i,arm in enumerate(config):
            m=sim.getObjectMatrix(handles[i],sim.handle_world)
            origin=[m[3],m[7],m[11]];axis=[m[2],m[6],m[10]]
            delta=[r-o for r,o in zip(arm['rcm_world'],origin)]
            residual=norm(cross(delta,axis))/norm(axis)
            reference=sim.getObjectPosition(refs[i],sim.handle_world)
            drift=norm([a-b for a,b in zip(reference,arm['rcm_world'])])
            maxima[i]=max(maxima[i],residual);reference_drift[i]=max(reference_drift[i],drift)
            records.append({'sample':k,'arm':arm['arm'],'rcm_error_m':residual,'reference_drift_m':drift})
        if k+1<args.samples:time.sleep(args.interval)
    result={'passed':max(maxima+reference_drift)<=args.tolerance,'samples_per_arm':args.samples,'tolerance_m':args.tolerance,'max_rcm_error_m':maxima,'max_fixed_reference_drift_m':reference_drift}
    print(json.dumps(result,indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(dict(result,samples=records),indent=2),encoding='utf-8')
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
