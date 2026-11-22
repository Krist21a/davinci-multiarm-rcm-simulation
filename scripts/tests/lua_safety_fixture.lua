-- Test-only CoppeliaSim stub. Same-sign joint values / opposing jaw axes and half-total-angle limits
-- match config/endoscope/endoscope.json and scripts/build/install_instruments.py. TCP is fixed to wrist.
stub={q={},intervals={},positions={},blocked={},objects={},writes=0,mesh=0}
sim={handle_world=-1,handle_scene=-2,handle_all=-3,colorcomponent_ambient_diffuse=0,
     colorcomponent_specular=1,colorcomponent_emission=2,shapeintparam_static=0,shapeintparam_respondable=1}
package.preload.sim=function() return sim end
CONFIG={};COLLISION_PAIRS={};INSTRUMENTS={}
for i=1,3 do
    local name='a'..i
    CONFIG[i]={arm=name,joints={{name=name..'_joint',limits={-1,1}}},
        insertion_joint=name..'_insert',shaft_frame=name..'_shaft',projection=name..'_projection',
        insertion_home=0.04,rcm_distance_home=0.1,rcm_world={0,0,0.1},axis_home={0,0,1},
        transverse_x={1,0,0},transverse_y={0,1,0},grip_joints={name..'_j1',name..'_j2'},
        tcp_frame=name..'_tcp',port_radius_m=0.01,tool_shape=name..'_tool',
        limits={yaw_deg=5,pitch_deg=5,insertion_min=0,insertion_max=0.05}}
    COLLISION_PAIRS[i]={name..'_tool',name..'_obstacle'}
    INSTRUMENTS[name]={capabilities=i==2 and {'cut'} or {'grasp','grasp_tissue'}}
    for _,suffix in ipairs({'_joint','_insert','_shaft','_projection','_j1','_j2','_tcp','_tool','_obstacle','_slide'}) do
        local h=name..suffix
        stub.objects[#stub.objects+1]=h;stub.q[h]=0
        stub.intervals[h]={-0.174533,0.872663}
    end
    stub.q[name..'_insert']=0.04
end
INSTRUMENTS.a3.hinge_slide='a3_slide'
INSTRUMENTS.a3.hinge_jaw_range_rad={0,math.pi/3}
INSTRUMENTS.a3.hinge_slope_m_per_rad=-0.0021

function eq(a,b) assert(math.abs(a-b)<1e-9,tostring(a)..' ~= '..tostring(b)) end
function sim.getObjectsInTree() return stub.objects end
function sim.getObjectAlias(h) return h end
function sim.getJointPosition(h) assert(stub.q[h]~=nil,'missing joint '..tostring(h));return stub.q[h] end
function sim.getJointInterval(h) return false,stub.intervals[h] end
function sim.setJointPosition(h,q)
    stub.writes=stub.writes+1
    if stub.quantize and h:match('_j[12]$') then q=math.floor(q*100+0.5)/100 end
    -- Inject one asymmetric setter result, then permit rollback to the exact old state.
    if stub.jaw2offset and h=='a1_j2' then q=q+stub.jaw2offset;stub.jaw2offset=nil end
    stub.q[h]=q
end
function sim.checkCollision(a,b) return (stub.blocked[a] or stub.blocked[b]) and 1 or 0 end
function sim.getObjectPosition(h)
    if stub.positions[h] then return stub.positions[h] end
    if h:match('_tcp$') then return {0.01,0,0} end
    if h:match('_shaft$') then return {0,0,stub.q[h:gsub('_shaft$','_insert')]-0.04} end
    return {0,0,0}
end
function sim.setObjectPosition(h,rel,p) stub.positions[h]={p[1],p[2],p[3]} end
function sim.getObjectMatrix(h)
    local p=sim.getObjectPosition(h)
    local sign=h:match('_j2$') and -1 or 1
    return {1,0,0,p[1],0,sign,0,p[2],0,0,sign,p[3]}
end
function sim.getMatrixInverse(m)
    return {m[1],m[5],m[9],-(m[1]*m[4]+m[5]*m[8]+m[9]*m[12]),
            m[2],m[6],m[10],-(m[2]*m[4]+m[6]*m[8]+m[10]*m[12]),
            m[3],m[7],m[11],-(m[3]*m[4]+m[7]*m[8]+m[11]*m[12])}
end
function sim.multiplyVector(m,v)
    return {m[1]*v[1]+m[2]*v[2]+m[3]*v[3]+m[4],m[5]*v[1]+m[6]*v[2]+m[7]*v[3]+m[8],
            m[9]*v[1]+m[10]*v[2]+m[11]*v[3]+m[12]}
end
function sim.createMeshShape() stub.mesh=stub.mesh+1;return stub.mesh end
function sim.removeObjects() end
function sim.setObjectAlias() end
function sim.setShapeColor() end
function sim.setObjectInt32Param() end
function sim.getSimulationTime() return 1 end

-- Small actual tissue grid with constraints crossing the blade plane, or entirely coplanar with it.
function sheet(angle,r,coplanar)
    local h=0.00002
    tissueCreate{nu=2,nv=2,spacing=h,origin={r*math.cos(angle),r*math.sin(angle),0},
                 u=coplanar and {1,0,0} or {0,0,1},v={0,1,0},cut_radius=0.000001,
                 grasp_radius=0.0015,iterations=0,damping=0,tear_ratio=1000}
end