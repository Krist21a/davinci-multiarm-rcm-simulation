import sys
sys.path.insert(0, r'D:\仿真环境\work\python_deps')
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(r'D:\仿真环境\output\davinci_multiarm')
def mesh(key):
    dt=np.dtype([('normal','<f4',3),('v','<f4',(3,3)),('attr','<u2')])
    return np.fromfile(ROOT/'meshes/cad'/f'{key}.stl',dtype=dt,offset=84)['v'].astype(float)
def read():
    components={}; cylinders={}; mates=[]; doc=None
    for line in (ROOT/'cad_export.tsv').read_text(encoding='utf-8-sig').splitlines():
        p=line.split('\t')
        if p[0]=='COMP':
            a=np.fromstring(p[3],sep=','); T=np.eye(4); T[:3,:3]=a[:9].reshape(3,3).T; T[:3,3]=a[9:12]
            components[p[1]]={'key':p[2],'T':T}
        elif p[0]=='CYL': cylinders.setdefault(p[1],[]).append(np.fromstring(p[2],sep=','))
        elif p[0]=='DOCUMENT':doc=p[1]
        elif p[0]=='MATE':mates.append({'doc':doc,'name':p[1],'type':p[2],'entities':[]})
        elif p[0]=='ENTITY':mates[-1]['entities'].append((p[1],np.fromstring(p[2],sep=',')))
    return components,cylinders,mates
def transform(T,v):return v@T[:3,:3].T+T[:3,3]
if __name__=='__main__':
    from PIL import Image,ImageDraw
    keys=sorted([p.stem for p in (ROOT/'meshes/cad').glob('*.stl')],key=lambda x:int(x.rstrip('-')))
    im=Image.new('RGB',(1500,300*6),'#e8edf2');d=ImageDraw.Draw(im)
    R=Rotation.from_euler('xyz',[70,0,-30],degrees=True).as_matrix()
    for i,k in enumerate(keys):
        v=mesh(k);q=v@R.T;z=q[:,:,2].mean(axis=1);xy=q[:,:,:2];lo=xy.reshape(-1,2).min(0);hi=xy.reshape(-1,2).max(0);sc=240/max(hi-lo);xy=(xy-(lo+hi)/2)*sc;xy[:,:,1]*=-1;xy+=np.array([i%5*300+150,i//5*300+145]);
        for j in np.argsort(z):
            normal=np.cross(q[j,1]-q[j,0],q[j,2]-q[j,0]);val=int(130+80*abs(normal[2])/(np.linalg.norm(normal)+1e-15));d.polygon([tuple(p) for p in xy[j]],fill=(val,val,min(255,val+12)))
        dim=np.ptp(v.reshape(-1,3),axis=0);d.text((i%5*300+10,i//5*300+10),f'PART {k}  '+','.join(f'{x:.3f}' for x in dim),fill='black')
    im.save(ROOT/'parts_contact_sheet.png')
