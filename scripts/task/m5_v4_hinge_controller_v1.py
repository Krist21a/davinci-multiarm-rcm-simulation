"""Flat hinge-coupon controller. No structural sensor, simulator or scorer imports."""
import numpy as np

import m5_v3_controller_v7 as v7
from m5_v3_corridor_v5 import measure as full_review
from m5_v3_perception_v4 import inks, intact_preflight
from m5_v4_frontier_v1 import measure as frontier
from m5_v4_hinge_work_v1 import chart, margin


def receipt(f,prior,sigma):
    if prior is None:
        ok = intact_preflight(f.rgb,f.depth,far=f.far)['passed']
        return ('UNCUT',0) if ok else ('UNKNOWN',None)
    a = frontier(f.rgb,f.depth,far=f.far,sigma=sigma,prefix=prior)
    b = full_review(f.rgb,f.depth,far=f.far,sigma=sigma)
    key = (a['verdict'],a['prefix_mm'])
    return key if key==(b['verdict'],b['prefix_mm']) else ('UNKNOWN',None)


class World(v7.World):
    def work(self,closed,prefix_mm):
        f = self.capture(False,closed)
        self.io.work_prefix(prefix_mm)
        g = chart(f.rgb,f.depth,f.focal,f.far,self.sigma,prefix_mm)
        self.io.work_estimate(None if g is None or f.tip is None else (
            f.R@g.direction,f.R@g.limit+f.t-f.tip))
        if g is None or f.tip is None:
            self.work_refusal = f'work tick {f.tick}: '+('missing chart/inverse support' if g is None else 'missing marker')
            return None
        R,t = np.asarray(f.R),v7.vector(f.t,3)
        v7.require(np.allclose(R.T@R,np.eye(3),atol=1e-6) and np.linalg.det(R)>0,'invalid camera matrix')
        g = v7.perception.WorkChart(g.prefix_mm,R@g.front+t,v7.unit(R@g.direction),v7.unit(R@g.opening),
                                    R@g.aim+t,R@g.limit+t)
        return v7.Work(g,v7.vector(f.tip,3),v7.unit(f.jaw),v7.unit(f.hinge),f.tick,self.epoch)

    def sweep_room(self,p):
        return super().sweep_room(p)+v7.sensor_margin(self.sigma)-margin(self.sigma)


class FSM(v7.FSM):
    def __init__(self,w):
        super().__init__(w)
        self.alternate_spent = False
        self.late_waypoint = False

    def run(self):
        w,io = self.world,self.world.io
        w.safe(True)
        v7.require(io.ready(),'not ready at start')
        self.trail = [w.q()]
        try:
            for _ in range(400):
                io.trace(self.state,self.prefix,self.bites,w.epoch)
                if self.state=='ALIGN':
                    if self.prefix>=6 and not self.late_waypoint:
                        v7.require(len(self.trail)==2 and self.align_count==0,'late approach not at entrance')
                        self.align_count += 1
                        p = w.work(True,self.prefix)
                        g = w.current(p)
                        q = w.solve(g.front-.002*g.direction,g.direction,g.opening)
                        w.move(q,True)
                        self.trail.append(q)
                        self.late_waypoint = True
                        continue
                    self._bite(w)
                    if self.bites==0 and len(self.trail)==2 and self.align_count==1:
                        io.entrance(self.trail[1])
                        f = w.capture(True,True)
                        val,n = receipt(f,None,w.sigma)
                        io.log_receipt(None,val,n)
                        v7.require(val=='UNCUT','missing uncut preflight')
                elif self.state=='REVIEW':
                    v7.require(len(self.trail)>=4 and np.allclose(w.q(),self.trail[-1],atol=1e-6,rtol=0),
                               'invalid review trail')
                    for q in reversed(self.trail[1:-1]):
                        w.move(q,True)
                    self.trail = self.trail[:2]
                    io.at_review(self.bites)
                    self.state = io.hold_status()
                    if self.state!='READ':
                        w.epoch += 1
                elif self.state in ('RECOVER_TENSION','RECOVER_GRASP'):
                    io.recover(self.state)
                    w.epoch += 1
                    w.safe(True)
                    self.state = 'READ'
                elif self.state=='OCCLUDED':
                    io.wait_closed()
                    w.epoch += 1
                    self.state = 'READ'
                elif self.state=='READ':
                    f = w.capture(True,True)
                    val,n = receipt(f,self.prefix,w.sigma)
                    io.log_receipt(self.prefix,val,n)
                    if val=='UNKNOWN' and not (inks(f.rgb)>=0).any() and io.wait_available():
                        self.state = 'OCCLUDED'
                        continue
                    if val=='UNKNOWN' and not self.alternate_spent:
                        self.alternate_spent = True
                        io.alternative()
                        w.epoch += 1
                        alt = w.capture(True,True)
                        av,an = receipt(alt,self.prefix,w.sigma)
                        io.log_receipt(self.prefix,av,an)
                        raise v7.Refused('primary UNKNOWN; alternate diagnostic only: '+str((av,an)))
                    v7.require(type(n) is int and self.prefix<n<=10 and val==('TARGET' if n>=9 else 'BELOW'),
                               'missing fresh optical progress')
                    self.prefix = n
                    if val=='TARGET':
                        w.move(self.trail[0],True)
                        v7.require(io.ready(),'closed ready readback refused')
                        self.state = 'DONE'
                        return
                    self.state,self.align_count = 'ALIGN',0
                else:
                    raise v7.Refused('invalid state')
            raise v7.Refused('episode budget exceeded')
        except (ValueError,RuntimeError,ArithmeticError) as err:
            self.state,self.reason = 'FAILED',str(err)
            io.hold()
