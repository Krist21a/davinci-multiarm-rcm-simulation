"""Show actual nominal vision-control observations and distinguish a no-growth retry from DONE."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def plot(folder):
    report=json.loads((folder/'nominal.json').read_bytes())
    samples=report['observations']
    read={s['tick']:s['estimate'] for s in report['perception']}
    bites=[e['tick'] for e in report['events'] if e['event']=='closure']
    good=[e['tick'] for e in report['events'] if e['event']=='post_closure_reacquired']
    exposure=next((t for t in good if (folder/'nominal_hd'/f'{t:03}.npz').is_file()),None)
    if exposure is None:
        raise ValueError('no HD reacquisition exposure')
    with np.load(folder/'nominal_hd'/f'{exposure:03}.npz') as raw:
        image=Image.fromarray(raw['rgb'])
    cut=read[exposure]['cut']
    x,y=cut['front_uv']
    left,top=max(0,round(x)-185),max(0,round(y)-120)
    crop=image.crop((left,top,left+370,top+240)).resize((555,360),Image.Resampling.NEAREST)
    dot=ImageDraw.Draw(crop)
    xx,yy=(x-left)*1.5,(y-top)*1.5
    dot.ellipse((xx-6,yy-6,xx+6,yy+6),outline='white',width=2)
    out=Image.new('RGB',(970,446),'#faf8f3')
    out.paste(crop,(395,48))
    d=ImageDraw.Draw(out)
    d.text((22,12),'NOMINAL / FROM UNCUT TO VISUAL REFUSAL',fill='#172e37',font_size=22)
    x0,y0,x1,y1=43,342,364,68
    xt=lambda t:x0+(t-1)/(samples[-1]['tick']-1)*(x1-x0)
    yt=lambda mm:y0+mm/10*(y1-y0)
    for mm in (0,3,5,8,9,10):
        d.line((x0,yt(mm),x1,yt(mm)),fill='#d1d5ce')
        d.text((15,yt(mm)-6),str(mm),fill='#25383e')
    path=[]
    for obs in samples:
        t,mm=obs['tick'],obs['cut_length_m']*1000
        if path:
            path.append((xt(t),path[-1][1]))
        path.append((xt(t),yt(mm)))
        val=read[t]['cut'].get('progress_m')
        if val is not None:
            xj,yj=xt(t),yt(val*1000)
            d.ellipse((xj-2,yj-2,xj+2,yj+2),fill='#d26720')
    d.line(path,fill='#245c70',width=2)
    for i,t in enumerate(bites):
        d.line((xt(t),y0+2,xt(t),y0+9),fill='#3a4346',width=2)
        d.text((xt(t)-4,y0+12),str(i+1),fill='#25383e')
    d.text((22,379),'Blue: independent structural length (mm)',fill='#245c70',font_size=14)
    d.text((22,398),'Orange: current optical reading. Ticks below: bites.',fill='#a04d11',font_size=13)
    d.text((395,413),f'Actual ECM RGB / tick {exposure} / white: detected slit / optical {cut["progress_m"]*1000:.2f} mm',fill='#25383e',font_size=14)
    with (folder/'nominal_live.png').open('xb') as fp:
        out.save(fp,format='PNG')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder',type=Path,required=True)
    plot(ap.parse_args().folder)
