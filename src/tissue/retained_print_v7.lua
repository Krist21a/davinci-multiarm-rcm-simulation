-- Draw whole-face optical codes using the original live-face renderer. No geometry edits.
function drawRetainedPrintV7(on)
    if not on then render() return true end
    assert(T and T.nu==24 and T.nv==14 and not T.seamMode and not T.p.visualGap and not T.bridgeRefs)
    local original, ids, palette=T.colors, {}, {}
    for i,c in ipairs(original) do palette[i]=c end
    local n=#palette
    local reds={192,205,218,239,252}
    for code=0,103 do
        palette[n+code+1]={reds[code%5+1]/255,(18+14*(math.floor(code/5)%5))/255,(94+14*math.floor(code/25))/255}
    end
    for idx,tr in ipairs(T.tri) do
        ids[idx]=tr[7]
        local i,j=math.floor((idx-1)/2)%23,math.floor((idx-1)/46)
        local lane=nil
        if i==7 or i==8 then lane=0
        elseif i==9 then lane=1
        elseif i==10 then lane=2
        elseif i==11 then lane=(idx%2==1) and 4 or 3
        elseif i==12 then lane=5
        elseif i==13 then lane=6
        elseif i==14 or i==15 then lane=7 end
        if lane then tr[7]=n+8*j+lane+1 end
    end
    T.colors=palette
    local ok,err=pcall(render)
    T.colors=original
    for idx,tr in ipairs(T.tri) do tr[7]=ids[idx] end
    if not ok then error(err) end
    return true
end
