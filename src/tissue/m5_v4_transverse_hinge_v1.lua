-- Private to the separately staged flat coupon. No additions to T.springs.
local function bendCross(u,v)
    return {u[2]*v[3]-u[3]*v[2],u[3]*v[1]-u[1]*v[3],u[1]*v[2]-u[2]*v[1]}
end

local function bendTransverse()
    if not (T and T.seamMode and T.nu==24 and T.nv==14 and guideBand==0.0001) then
        error('unsupported bend fixture')
    end
    for j=1,T.nv-2 do for i=0,T.nu-2 do
        local above=T.tri[2*((j-1)*(T.nu-1)+i)+2]
        local below=T.tri[2*(j*(T.nu-1)+i)+1]
        if faceLive(above) and faceLive(below) then
            local a,b,c,d=below[1],below[2],above[1],below[3]
            local p,q,r=vsub(T.x[b],T.x[a]),vsub(T.x[c],T.x[a]),vsub(T.x[d],T.x[a])
            local gb,gc,gd=bendCross(q,r),bendCross(r,p),bendCross(p,q)
            local ids={a,b,c,d}
            local grads={vmul(vadd(vadd(gb,gc),gd),-1),gb,gc,gd}
            local denom=0
            for k,id in ipairs(ids) do
                if not T.pinned[id] then denom=denom+vdot(grads[k],grads[k]) end
            end
            if denom>1e-28 then
                local lambda=-0.08*vdot(p,gb)/denom
                for k,id in ipairs(ids) do
                    if not T.pinned[id] then T.x[id]=vadd(T.x[id],vmul(grads[k],lambda)) end
                end
            end
        end
    end end
end
