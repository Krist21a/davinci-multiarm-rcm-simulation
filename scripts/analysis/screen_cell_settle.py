"""Production Lua sheet-only comparison before asking a simulator to grasp a new specimen."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from capture_surface_seam import ROOT, sha256, write_new
from seam_slide_world import staged_source
from seam_wide_cell_world import wide_source

sys.path.insert(0,str(ROOT/'scripts/tests'))
from test_lua_safety import FIXTURE, LEGACY_SOURCES as SOURCES, LuaRuntime

OUT=ROOT/'validation/task/seam_cell_settle_v1'
WIDE=ROOT/'validation/task/seam_wide_cell_v1/wide/pull/oracle.json'
ANCHOR='vmul(spec.u,(i-(nu-1)/2)*h)'
LEFT='vmul(spec.u,((i<=11 and (-11.5+i*(9/11))) or (i-11.5))*h)'


def asymmetric_source(parent):
    base=staged_source(parent)
    if base.count(ANCHOR)!=1:
        raise ValueError('unexpected original grid')
    return base.replace(ANCHOR,LEFT)+'\n'+(ROOT/'src/tissue/cell_certificate_print.lua').read_text(encoding='utf-8')


def settle(out):
    out.mkdir(parents=True,exist_ok=False)
    criteria=dict(scope='Isolated ungrasped PBD in original Lua and deterministic sim stub. Not IK, a live trace, '
                        'a cutting experiment or a camera. Ready-frame orientation reconstructed from the recorded TCP.',
        cases=['narrow slide (reference)','5 mm symmetric (observed live startup tear)',
               '3 mm asymmetric (left host at -2.5 mm, unchanged original right at +0.5 mm)'],
        gate='Use the same frame, pretension 8%, fixed far edge, 12 sweeps, tear ratio, guide, and 30 ordinary '
             'settling ticks as placement. No perturbation, no grasp, no changing widths or trying other seeds. '
             'Report every new tear with material node coordinates and strain. Consider the asymmetric candidate '
             'for a new live preflight only if narrow is undamaged, symmetric reproduces the live tear count, and '
             'asymmetric remains uncut and untorn. Do not attribute a particular live bond without a live tear log.',
        source_sha256=sha256(Path(__file__)),live_reference=sha256(WIDE))
    write_new(out/'criteria.json',criteria)
    ref=json.loads(WIDE.read_bytes())
    tcp=np.array(ref['ready_tcp'])
    o,v,u=tcp[:,3]+.002*tcp[:,2],tcp[:,2],-tcp[:,0]
    vec=lambda a:'{'+','.join(format(n,'.17g') for n in a)+'}'
    init='{nu=24,nv=14,spacing=.001,pretension=.08,anchor_sides=false,anchor_from_row=3,grasp_radius=.0025,origin=%s,u=%s,v=%s}'%(vec(o),vec(u),vec(v))
    parent='\n'.join(p.read_text(encoding='utf-8') for p in (FIXTURE,*SOURCES,ROOT/'src/tissue/ruler_print.lua',ROOT/'src/tissue/surface_seam.lua'))
    runtime=LuaRuntime()
    runs=[]
    for name,source in (('narrow',staged_source(parent)),('symmetric',wide_source(parent)),('asymmetric',asymmetric_source(parent))):
        path=out/(name+'.json')
        # Lua on Windows cannot reliably open the non-ASCII absolute repo path. Use the tested relative path.
        rel=path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        runtime.run(source+'''
sysCall_init();auto=false
tissueCreate('''+init+''')
local raw=assert(io.open('''+json.dumps(rel)+''',"wb"))
raw:write('{"steps":[')
local num=0
for tick=1,30 do
    tissueStep()
    local s=tissueStats()
    if tick>1 then raw:write(',') end
    raw:write(string.format('{"tick":%d,"peak":%.17g,"torn":%d,"cut":%d,"new":[',tick,s.max_strain,s.torn,s.cut))
    for i=num+1,#s.torn_events do
        local r=s.torn_events[i]
        if i>num+1 then raw:write(',') end
        raw:write(string.format('{"a":%d,"b":%d,"strain":%.17g}',r.a,r.b,r.strain))
    end
    num=#s.torn_events
    raw:write(']}')
end
raw:write(']}\\n');raw:close()
''')
        series=json.loads(path.read_bytes())['steps']
        runs.append(dict(name=name,trace_sha256=sha256(path),final=series[-1],events=[
            dict(step=s['tick'],a=[ev['a']%24,ev['a']//24],b=[ev['b']%24,ev['b']//24],strain=ev['strain'])
            for s in series for ev in s['new']]))
    actual=ref['observations'][-1]
    passed=(actual['state']=='LOCALIZE' and not ref.get('pull_motion_m') and not ref.get('events')
            and runs[1]['final']['torn']==actual['torn']==1 and runs[0]['final']['torn']==runs[2]['final']['torn']==0
            and all(not r['final']['cut'] for r in runs))
    write_new(out/'decision.json',dict(criteria_sha256=sha256(out/'criteria.json'),results=runs,
              asymmetric_eligible_only_for_live_preflight=passed,visual_control=False))
    print('eligible for preflight',passed,[(r['name'],r['events']) for r in runs])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=OUT)
    settle(ap.parse_args().out)
