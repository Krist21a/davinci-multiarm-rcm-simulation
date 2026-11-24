"""Count independently decoded local flanks and compare real optics from frozen states."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/vision'))
from ruler_readout import classify, crossings, inks

TASK = ROOT/'validation/task'
SPEC = ROOT/'config/task/ruler_endpoint_optics_v1.json'
CAP = ROOT/'scripts/analysis/capture_ruler_optics.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out, picture):
    if out.exists() or picture.exists():
        raise FileExistsError('new outputs required')
    items, tiles = [], []
    for goal in (8,9):
        orig = TASK/f'ruler_dev_c3_g{goal}'/'oracle.json'
        ref = json.loads(orig.read_bytes())
        f = next(x for x in ref['frames'] if x['phase']=='park')
        for case in json.loads(SPEC.read_bytes())['candidates']:
            target,name = case['target'],case['name']
            p=TASK/f'ruler_optics_g{goal}_{name}'/'capture.json'
            r=json.loads(p.read_bytes())
            if (r['error'] or r['code_sha256']!=sha(CAP) or r['spec_sha256']!=sha(SPEC)
                    or r['source_sha256']!=sha(orig) or r['target']!=target
                    or not np.allclose(r['target_readback'],target,atol=1e-6,rtol=0)
                    or [x['shape'] for x in r['frames']]!=[[480,640],[1920,2560]]):
                raise ValueError(f'failed capture: {p.name}, {r["error"]}')
            for fr in r['frames']:
                sample=p.parent/fr['path']
                if sha(sample)!=fr['sha256']:
                    raise ValueError('frame hash mismatch')
                with np.load(sample) as data:
                    rgb,depth=data['rgb'],data['depth']
                # Crop well outside the printed mask for the diagnostic; preserve
                # real camera resolution and depth, do not fill absent pixels.
                label=inks(rgb)
                yy,xx=np.where(label>=0)
                y0,y1=max(0,yy.min()-16),min(len(label),yy.max()+17)
                x0,x1=max(0,xx.min()-16),min(label.shape[1],xx.max()+17)
                crop,depth=rgb[y0:y1,x0:x1],depth[y0:y1,x0:x1]
                result=classify(crop,depth)
                xs=crossings(crop,depth)
                counts=np.bincount((label+1).ravel(),minlength=27)[1:]
                row=dict(goal=goal, actual_mm=round(f['length_m']*1000), view=name,
                         shape=fr['shape'],sha256=fr['sha256'], left=counts[:13].tolist(),
                         right=counts[13:].tolist(),decision=result,crossings=xs)
                items.append(row)
                tile=Image.new('RGB',(375,210),'white')
                im=Image.fromarray(crop)
                factor=min(355/im.width,176/im.height)
                im=im.resize((int(factor*im.width),int(factor*im.height)),Image.Resampling.NEAREST)
                tile.paste(im,((tile.width-im.width)//2,26))
                ImageDraw.Draw(tile).text((4,4),f'{row["actual_mm"]} mm | {name} | {fr["shape"][1]}x{fr["shape"][0]}',fill='black')
                tiles.append(tile)
                print(goal,name,fr['shape'],result['decision'], result.get('reason'),
                      [(j,int(counts[j]),int(counts[13+j])) for j in (8,9,10)])
    sheet=Image.new('RGB',(750,840),'white')
    for k,tile in enumerate(tiles):
        sheet.paste(tile,((k%2)*375,(k//2)*210))
    sheet.save(picture)
    with out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(dict(images=items, spec_sha256=sha(SPEC), reader_sha256=sha(ROOT/'scripts/vision/ruler_readout.py'),
                       figure_sha256=sha(picture),code_sha256=sha(Path(__file__))),f,indent=2)
        f.write('\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--picture',required=True,type=Path)
    args=ap.parse_args()
    main(args.out,args.picture)
