"""Matched unloaded entry calibration; report a localization/IK refusal as such."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import capture_ruler_endpoint as acquisition
from simtools import add_arguments, sha256

SPEC = acquisition.ROOT/'config/task/ruler_entry_mount_v1.json'


def main(args):
    from ruler_entry_mount import EntryMountWorld

    holder={}

    class Recorder(EntryMountWorld):
        def home(self):
            ok=super().home()
            holder['world']=self
            return ok

        def localize(self):
            result=super().localize()
            # Post-hoc diagnostic: record the same gate that ran, no substitution.
            projected,pc=self.project(self.cut_front())
            visible,offset=self.point_visible(self.cut_front())
            record=dict(tick=self.tick,visible=visible,offset=offset,uv=None if projected is None else list(projected))
            if projected is not None:
                x,y=np.rint(projected).astype(int)
                if 1<=x<self.W-1 and 1<=y<self.H-1:
                    record['nearest_centre_minus_target_mm']=float(1000*(self.depth[y-1:y+2,x-1:x+2].min()-pc[2]))
            holder.setdefault('entry',[]).append(record)
            return result

    args.goal=int(args.goal)
    args.cap=3
    spec=json.loads(SPEC.read_bytes())
    args.pivot_local=spec['pivot_local_uvw_m']
    with patch.object(acquisition,'RulerWorld',Recorder):
        rc=acquisition.main(args)
    p=args.out/'oracle.json'
    r=json.loads(p.read_bytes())
    w=holder.get('world')
    with (args.out/'mount.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(dict(spec_sha256=sha256(SPEC),script_sha256=sha256(Path(__file__)),oracle_sha256=sha256(p),
                       geometry_sha256=sha256(acquisition.ROOT/'scripts/task/ruler_entry_mount.py'),
                       entry=holder.get('entry',[]),events=[] if w is None else w.event_log),f,indent=2,
                  default=lambda v: np.asarray(v).tolist(),allow_nan=False)
        f.write('\n')
    return rc


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--goal',required=True,type=int,choices=(8,9))
    ap.add_argument('--out',required=True,type=Path)
    raise SystemExit(main(ap.parse_args()))
