"""Flat v3 development policy. Deliberately no live CoppeliaSim entry point."""
from dataclasses import dataclass
import math

import numpy as np

from sim_world import sweep_projection
import m5_v3_perception_v4 as perception
from m5_v3_rigid_fk import CutterFK
import tool_pose_active_set as ik


class Refused(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise Refused(message)


def vector(a, length):
    a = np.asarray(a,dtype=float)
    require(a.shape==(length,) and np.isfinite(a).all(),'invalid readback')
    return a


def unit(a):
    a = vector(a,3)
    require(np.linalg.norm(a)>1e-9,'invalid axis')
    return a/np.linalg.norm(a)


@dataclass(frozen=True)
class RGBD:
    tick: int
    rgb: np.ndarray
    depth: np.ndarray
    focal: float
    far: float
    R: np.ndarray = None
    t: np.ndarray = None
    # Same-frame fitted marker pose in world coordinates; never use TCP truth here.
    tip: np.ndarray = None
    jaw: np.ndarray = None
    hinge: np.ndarray = None


@dataclass(frozen=True)
class Work:
    chart: perception.WorkChart
    tip: np.ndarray
    jaw: np.ndarray
    hinge: np.ndarray
    tick: int
    epoch: int


class World:
    """Guarded motion adapter. IO exposes readback, guarded verbs and RGB-D, not structural labels.

    Methods on IO: safe(), closed(), ready(), q(), limits() -> (lo, hi),
    capture(review: bool) -> RGBD, move(q), set_open(), cut(), camera(view), hold().
    Commands return bool.
    Capture is passive, synchronous, and stamped with a strictly increasing tick.
    """

    def __init__(self, io, goal_mm, model, sigma=0.):
        require(type(goal_mm) is int and goal_mm in (9,10),'unregistered endpoint')
        require(isinstance(model,CutterFK),'rigid FK calibration required')
        require(sigma in (0.,.0005),'unregistered depth noise')
        self.io,self.goal_mm,self.model,self.sigma = io,goal_mm,model,sigma
        self.epoch,self.tick,self.camera_count = 0,-1,0

    def safe(self, closed):
        require(self.io.safe() is True and self.io.closed() is closed,'mechanical or jaw interlock')

    def q(self):
        return vector(self.io.q(),6).copy()

    def capture(self, review, closed):
        self.safe(closed)
        f = self.io.capture(review)
        self.safe(closed)
        require(type(f.tick) is int and self.tick<f.tick<400,'stale or out-of-budget frame')
        self.tick = f.tick
        return f

    def work(self, closed, prefix_mm):
        f = self.capture(False,closed)
        g = perception.work(f.rgb,f.depth,focal=f.focal,far=f.far,sigma=self.sigma,
                            prefix_mm=prefix_mm,goal_mm=self.goal_mm)
        if g is None:
            return None
        R,t = np.asarray(f.R,dtype=float),vector(f.t,3)
        require(R.shape==(3,3) and np.isfinite(R).all() and np.allclose(R.T@R,np.eye(3),atol=1e-6)
                and np.linalg.det(R)>0.,'invalid camera pose')
        point = lambda p:R@vector(p,3)+t
        g = perception.WorkChart(g.prefix_mm,point(g.front),unit(R@g.direction),unit(R@g.opening),
                                 point(g.aim),point(g.limit))
        return Work(g,vector(f.tip,3),unit(f.jaw),unit(f.hinge),f.tick,self.epoch)

    def current(self, p):
        require(p is not None and (p.tick,p.epoch)==(self.tick,self.epoch),'stale working geometry')
        return p.chart

    def solve(self, p, axis, opening):
        lo,hi = self.io.limits()
        answer = ik.solve(self.model.pose,self.q(),(lo,hi),p,axis,opening,axis_col=2,yaw_col=0)
        require(answer['converged'],'bounded pose solve refused')
        q = vector(answer['q'],6)
        require((4.5-np.abs(q[:2])>=1.5).all(),'dual-axis workspace reserve refused')
        return q

    def move(self, q, closed):
        self.safe(closed)
        self.epoch += 1
        require(self.io.move(q) is True,'guarded pose command refused')
        self.safe(closed)
        require(np.allclose(self.q(),q,rtol=0,atol=1e-6),'pose readback disagreed')

    def look(self, target):
        self.safe(True)
        require(self.camera_count<3,'camera budget exhausted')
        self.camera_count += 1
        self.epoch += 1
        require(self.io.camera(target) is True,'guarded camera refused')
        self.safe(True)

    def open(self):
        self.safe(True)
        self.epoch += 1
        require(self.io.set_open() is True,'guarded opening refused')
        self.safe(False)

    def sweep_room(self, p):
        g = self.current(p)
        R = np.column_stack((p.hinge,np.cross(p.jaw,p.hinge),p.jaw))
        require(np.allclose(R.T@R,np.eye(3),atol=1e-6) and np.linalg.det(R)>0.,
                'invalid measured tool pose')
        bounds = [sweep_projection(R@a,unit(R@b),np.zeros(3),g.direction)
                  for a,b in self.model.blade_local()]
        return float((g.limit-p.tip)@g.direction)-max(bounds)-.0003-.00025

    def close(self, p):
        self.safe(False)
        self.current(p)
        require(aligned(p,.0015),'fresh optical closure alignment refused')
        require(self.sweep_room(p)>=-1e-12,'fresh sweep envelope refused closure')
        self.epoch += 1
        require(self.io.cut() is True,'guarded cutting closure refused')
        self.safe(True)

    def review(self):
        f = self.capture(True,True)
        return perception.review(f.rgb,f.depth,far=f.far,sigma=self.sigma)


def aligned(p, tolerance, before_advance=False):
    g = p.chart
    c = math.cos(math.radians(5.))
    diff = g.front-p.tip
    return (p.jaw@g.direction>=c and abs(p.hinge@g.opening)>=c
            and np.linalg.norm(np.cross(diff,p.jaw))<=tolerance
            and (not before_advance or abs(diff@g.direction-.001)<=.0004))


class FSM:
    """Closed alignment -> preplanned advance -> fresh closure -> optical review."""

    def __init__(self, world, views=()):
        self.world,self.views = world,tuple(views)
        self.state,self.prefix,self.reason = 'ALIGN',0,None
        self.trail,self.align_count,self.bites = [],0,0

    def step(self):
        if self.state in ('DONE','FAILED'):
            return self.state
        w = self.world
        try:
            w.safe(True)
            if not self.trail:
                require(w.io.ready() is True,'immutable ready pose unavailable')
                self.trail.append(w.q())
            if self.state=='ALIGN':
                self._bite(w)
            else:
                self._review(w)
        except (Refused, ValueError, RuntimeError, ArithmeticError) as exc:
            self.state,self.reason = 'FAILED',str(exc)
            w.io.hold()  # never close just to park after an open-jaw refusal
        return self.state

    def _bite(self, w):
        require(self.align_count<10,'alignment budget exhausted')
        self.align_count += 1
        p = w.work(True,self.prefix)
        if p is None and w.camera_count<len(self.views):
            w.look(self.views[w.camera_count])
            return
        g = w.current(p)
        require(g.prefix_mm==self.prefix,'current material front disagrees with last review')
        if len(self.trail)==1:
            q = w.solve(g.front-.005*g.direction,g.direction,g.opening)
            require(not np.allclose(q,self.trail[0],rtol=0,atol=1e-6),'entrance clear and ready must differ')
            w.move(q,True)
            self.trail.append(q)
            return
        if not aligned(p,.00028 if w.sigma else .0004,True):
            q = w.solve(g.front-.001*g.direction,g.direction,g.opening)
            require(len(self.trail)<15,'reverse trail exhausted')
            w.move(q,True)
            self.trail.append(q)
            return
        require(len(self.trail)>=3 and len(self.trail)<16,'no guarded approach and reverse trail')
        require(np.allclose(w.q(),self.trail[-1],rtol=0,atol=1e-6)
                and not np.allclose(self.trail[-1],self.trail[0],rtol=0,atol=1e-6),
                'review backoff must be a recorded non-ready pose')
        require(self.bites<5,'bite budget exhausted')
        projection = float(p.jaw@g.direction)
        amount = min(.003 if self.prefix==0 else .002,
                     .001*(w.goal_mm-self.prefix),
                     float((g.aim-p.tip)@g.direction)/projection,
                     w.sweep_room(p)-.00015)
        require(math.isfinite(amount) and amount>=.0002,'bounded advance refused')
        q = w.solve(w.model.pose(w.q())[:,3]+amount*p.jaw,p.jaw,p.hinge)
        old_tick = p.tick
        w.open()  # no grip command until bounded advance IK is known
        w.move(q,False)
        self.trail.append(q)
        current = w.work(False,self.prefix)
        fresh = w.current(current)
        require(current.tick>old_tick and fresh.prefix_mm==self.prefix,
                'fresh optical closure alignment refused')
        w.close(current)
        self.bites += 1
        self.state = 'REVIEW'

    def _review(self, w):
        require(len(self.trail)>=4 and np.allclose(w.q(),self.trail[-1],rtol=0,atol=1e-6),
                'invalid closed review path')
        for q in reversed(self.trail[1:-1]):
            w.move(q,True)
        self.trail = self.trail[:2]
        verdict = w.review()
        n = verdict['prefix_mm']
        require(type(n) is int and self.prefix<n<=w.goal_mm,'no fresh growing optical boundary')
        require(verdict['verdict']==('TARGET' if n>=9 else 'BELOW'),'optical boundary is not a permit')
        if n==w.goal_mm:
            w.move(self.trail[0],True)
            require(w.io.ready() is True,'closed ready pose not verified')
            self.state = 'DONE'
            return
        self.prefix,self.align_count,self.state = n,0,'ALIGN'
