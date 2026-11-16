"""Visual skin posing with uniform cross-section weights and rigid distal limbs."""
from davinci_geometry import *
def exterior_skin(vertices, faces):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    _,labels=connected_components(coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(vertices),len(vertices))),directed=False)
    candidates=[]
    for label in np.unique(labels):
        ff=faces[labels[faces[:,0]]==label]
        if len(ff)<100:continue
        tri=vertices[ff];volume=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
        candidates.append((volume,label))
    label=max(candidates)[1];ff=faces[labels[faces[:,0]]==label];ids=np.unique(ff);mapping=np.full(len(vertices),-1);mapping[ids]=np.arange(len(ids))
    return vertices[ids],mapping[ff]
def pose_legs(local, angle_deg=20, faces=None):
    posed=local.copy()
    if faces is None:raise ValueError('Topology is required to exclude disconnected hands from lower-body posing.')
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edge=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);edge=edge[np.all(local[edge,0]>-.12,axis=1)]
    graph=coo_matrix((np.ones(len(edge)),(edge[:,0],edge[:,1])),shape=(len(local),len(local))).tocsr()
    _,labels=connected_components(graph,directed=False)
    # BodyParts3D skin contains disconnected inner/outer shells. Pose every
    # lower-limb shell, not just the component containing one extreme vertex.
    leg_component=np.isin(labels,np.unique(labels[local[:,0]>.5]))
    def smooth(x):x=np.clip(x,0,1);return x*x*(3-2*x)
    for sign in [-1,1]:
        # Hands are disconnected from the lower limb surface; never bend them with legs.
        mask=(local[:,1]*sign>0)&(local[:,0]>0)&leg_component
        p=local[mask]
        # Weight varies along the limb only. The previous lateral weight deformed
        # different sides of each calf/foot by different angles, creating a web.
        longitudinal=smooth(p[:,0]/.115)
        pelvic=smooth((abs(p[:,1])-.02)/.05)
        distal=smooth((p[:,0]-.105)/.035)
        w=longitudinal*(pelvic+(1-pelvic)*distal)
        pivot=np.array([-.035,sign*.08,.51]);d=p-pivot
        ang=sign*np.deg2rad(angle_deg)*w;co=np.cos(ang);si=np.sin(ang)
        posed[mask,0]=pivot[0]+co*d[:,0]-si*d[:,1]
        posed[mask,1]=pivot[1]+si*d[:,0]+co*d[:,1]
    # Local/global ARAP relaxation of the proximal skin, with the abdomen,
    # central genital surface, hands and complete distal legs fixed.
    from scipy.sparse import diags
    from scipy.sparse.linalg import splu
    free=leg_component&(local[:,0]>.035)&(local[:,0]<.20)
    protected=(local[:,0]<=.035)|((abs(local[:,1])>.18)&(local[:,0]<.22))
    free&=~protected
    posed[protected]=local[protected]
    free&=~((abs(local[:,1])<.025)&(local[:,0]<.115)&(local[:,2]>.585))
    fixed_center=leg_component&(abs(local[:,1])<.025)&(local[:,0]<.115)&(local[:,2]>.585)
    posed[fixed_center]=local[fixed_center]
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    edges=edges[free[edges].any(axis=1)];active=np.unique(edges);mapping=np.full(len(local),-1);mapping[active]=np.arange(len(active));ee=mapping[edges];i,j=ee.T
    p=local[active];q=posed[active].copy();d=p[i]-p[j];weight=1/np.maximum(np.linalg.norm(d,axis=1),.002)
    W=coo_matrix((np.r_[weight,weight],(np.r_[i,j],np.r_[j,i])),shape=(len(active),len(active))).tocsr();L=diags(np.asarray(W.sum(axis=1)).ravel())-W
    fi=np.flatnonzero(free[active]);ai=np.flatnonzero(~free[active]);lu=splu(L[fi][:,fi].tocsc());boundary=L[fi][:,ai]@q[ai]
    for _ in range(20):
        covariance=np.zeros((len(active),3,3));contrib=weight[:,None,None]*(q[i]-q[j])[:,:,None]*d[:,None,:]
        np.add.at(covariance,i,contrib);np.add.at(covariance,j,contrib)
        U,_,V=np.linalg.svd(covariance);rotation=U@V;negative=np.linalg.det(rotation)<0;U[negative,:,-1]*=-1;rotation=U@V
        rhs=np.zeros_like(q);term=weight[:,None]*np.einsum('nij,nj->ni',(rotation[i]+rotation[j])*.5,d);np.add.at(rhs,i,term);np.add.at(rhs,j,-term)
        q[fi]=lu.solve(rhs[fi]-boundary)
    posed[active]=q
    return posed

if __name__=='__main__':
    from plan_prostate_rcm import obj
    import json
    R=np.array([[0,0,-1],[1,0,0],[0,-1,0.]])
    v,f=exterior_skin(*obj('FJ2810.obj'));local=v@R.T+np.array([.78,0,.4652476]);posed=pose_legs(local,faces=f)
    edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    length=np.linalg.norm(local[edges[:,0]]-local[edges[:,1]],axis=1);new=np.linalg.norm(posed[edges[:,0]]-posed[edges[:,1]],axis=1);valid=length>1e-8;ratio=new[valid]/length[valid]
    distal=(local[edges].min(axis=1)[:,0]>.20)&(local[edges[:,0],1]*local[edges[:,1],1]>0)&valid
    report={'vertices':len(v),'triangles':len(f),'edge_ratio_percentiles':np.percentile(ratio,[0,1,50,99,100]).tolist(),'edges_stretched_over_2x':int(sum(ratio>2)),'distal_edge_max_relative_change':float(np.max(abs(new[distal]/length[distal]-1))),'abdomen_max_displacement_m':float(np.max(np.linalg.norm(posed[local[:,0]<0]-local[local[:,0]<0],axis=1)))}
    out=ROOT.parent/'davinci_prostate_rcm';(out/'leg_pose_validation.json').write_text(json.dumps(report,indent=2));np.savez(out/'corrected_skin.npz',vertices=(posed-np.array([.78,0,.4652476]))@R,faces=f)
    print(json.dumps(report),flush=True)
