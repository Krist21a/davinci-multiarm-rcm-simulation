"""Collision-pair rules for the endoscope arm (ECM) and the in-memory collision probe."""
import xml.etree.ElementTree as ET

from simtools import URDF

# Lua probe that is created in memory only; it is never saved into a scene file.
PROBE_ALIAS = 'BuildCollisionProbe'
PROBE_CODE = """sim=require('sim')
function sysCall_init() end
function scan(ps)
 local hits={}
 for i,p in ipairs(ps) do
  if sim.checkCollision(p[1],p[2])>0 then
   local n,segments=sim.checkCollisionEx(p[1],p[2]);hits[#hits+1]={index=i,segments=segments or {}}
  end
 end
 return hits
end
function setJoints(hs,qs) for i,h in ipairs(hs) do sim.setJointPosition(h,qs[i]) end end
"""


def urdf_adjacency():
    root = ET.parse(URDF).getroot()
    return {frozenset((j.find('parent').get('link'), j.find('child').get('link'))) for j in root.findall('joint')}


def link_of(shape):
    return shape[:-len('_visual')] if shape.endswith('_visual') else shape


def ecm_pair_candidates(shape_names):
    """All pairs involving a moving ECM visual shape, before zero-pose contact exclusion."""
    names = sorted(set(shape_names))
    ecm = [n for n in names if n.startswith('ecm_') and n.endswith('_visual')]
    anatomy = [n for n in names if n.startswith('Patient_')]
    tables = [n for n in names if n == 'TaskSurface' or n.startswith('LegSupport_')]
    psm = [n for n in names if n.startswith('psm') and n.endswith('_visual')]
    adjacent = urdf_adjacency()
    labels, categories = [], []

    def add(a, b, category):
        labels.append([a, b])
        categories.append(category)

    for n in ecm:
        for m in anatomy:
            add(n, m, 'ecm_anatomy')
        for m in tables:
            add(n, m, 'ecm_table')
        for m in psm:
            add(n, m, 'ecm_psm')
        if frozenset((link_of(n), 'cart')) not in adjacent:
            add(n, 'cart_visual', 'ecm_cart')
    for i, n in enumerate(ecm):
        for m in ecm[i + 1:]:
            if frozenset((link_of(n), link_of(m))) in adjacent:
                continue
            add(n, m, 'ecm_same_arm')
    return labels, categories


def scope_port_allowed(pair, points, port, scope_shape, radius):
    """Only the scope crossing the skin close to its own port is an intended contact."""
    import numpy as np
    if pair != [scope_shape, 'Patient_Skin'] or len(points) == 0:
        return False, None
    r = float(np.linalg.norm(np.asarray(points) - np.asarray(port), axis=1).max())
    return r <= radius, r


def wait_initialized(sim, script, timeout=10.0):
    """Customization scripts initialise asynchronously after createScript."""
    import time
    t0 = time.time()
    while True:
        try:
            sim.callScriptFunction('sysCall_init', script)
            return
        except Exception:
            if time.time() - t0 > timeout:
                raise
            if sim.getSimulationState() != sim.simulation_stopped:
                sim.step()          # in stepping mode new scripts only initialise on the next step
            else:
                time.sleep(0.1)


def create_probe(sim):
    h = sim.createScript(sim.scripttype_customization, PROBE_CODE, 0, 'lua')
    sim.setObjectAlias(h, PROBE_ALIAS)
    wait_initialized(sim, h)
    return h


def scan(sim, probe, handle_pairs):
    import numpy as np
    out = []
    for hit in sim.callScriptFunction('scan', probe, handle_pairs):
        out.append((hit['index'] - 1, np.array(hit['segments'], dtype=float).reshape(-1, 3)))
    return out
