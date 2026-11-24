-- Control panel for the endoscope task scene. ENDOSCOPE is prepended by
-- scripts/build/build_endoscope_scene.py (camera path, floating-view rectangle, per-arm ranges).
sim=require('sim')
if sim.getBoolParam(sim.boolparam_headless) then return end
simUI=require('simUI')
local function sliderXml(id,callback) return '<hslider id="'..id..'" minimum="-100" maximum="100" value="0" on-change="'..callback..'"/>' end
function sysCall_init()
    controller=sim.getObject('/RCMController');selected=1;sliders={0,0,0};wristSliders={0,0,0}
    endoCamera=sim.getObject(ENDOSCOPE.camera)
    camera=sim.getObjectsInTree(sim.handle_scene,sim.sceneobject_camera,0)[1]
    overview=sim.getObjectMatrix(camera,sim.handle_world)
    local items='';local views=''
    for i,a in ipairs(ENDOSCOPE.arms) do
        items=items..'<item>'..a.label..'</item>'
        views=views..'<button text="'..a.short..'" on-click="view'..i..'"/>'
    end
    ui=simUI.create([[<ui title="Abdominal RCM | endoscope task" closeable="false" placement="relative" position="20,20" layout="vbox">
    <label text="Registered skin ports / guarded kinematic control / 30 deg endoscope"/>
    <group layout="hbox"><button text="Overview" on-click="viewOverview"/>]]..views..[[</group>
    <button text="Show endoscope view" on-click="showEndoscope"/>
    <button text="Run PSM demo (endoscope holds)" on-click="run"/>
    <button text="Hold all arms" on-click="hold"/>
    <button text="RCM home" on-click="home"/>
    <group layout="hbox"><button text="Open jaws" on-click="openJaws"/><button text="Close jaws" on-click="closeJaws"/></group>
    <combobox id="1" on-change="armChanged">]]..items..[[</combobox>
    <label id="7" text="Yaw"/>]]..sliderXml(2,'yawChanged')..[[
    <label id="8" text="Pitch"/>]]..sliderXml(3,'pitchChanged')..[[
    <label id="9" text="Insertion"/>]]..sliderXml(4,'insertChanged')..[[
    <label id="13" text="Wrist roll (separate from shaft yaw/pitch)"/>]]..sliderXml(10,'rollChanged')..[[
    <label id="14" text="Wrist pitch"/>]]..sliderXml(11,'wristPitchChanged')..[[
    <label id="15" text="Wrist yaw"/>]]..sliderXml(12,'wristYawChanged')..[[
    <label id="16" text="Wrist controls require an articulated Si scene."/>
    <label id="5" text="Start simulation to enable constrained controls."/>
    <label id="6" text="RCM residuals will appear here."/>
    <label text="Exploratory CAD joint ranges; no clinical validation."/>
    </ui>]])
    updateLabels();showEndoscope();lastRender=-1
end
function arm() return ENDOSCOPE.arms[selected] end
function values()
    local a=arm()
    local ins=sliders[3]>=0 and a.insertion_home+sliders[3]/100*(a.insertion_max-a.insertion_home) or a.insertion_home+sliders[3]/100*(a.insertion_home-a.insertion_min)
    return sliders[1]/100*a.yaw_deg,sliders[2]/100*a.pitch_deg,ins
end
function updateLabels()
    local yaw,pitch,ins=values();local a=arm()
    simUI.setLabelText(ui,7,string.format('RCM shaft yaw %.2f deg (limit %.2f)',yaw,a.yaw_deg),true)
    simUI.setLabelText(ui,8,string.format('RCM shaft pitch %.2f deg (limit %.2f)',pitch,a.pitch_deg),true)
    simUI.setLabelText(ui,9,string.format('Insertion %.1f mm (%.0f to %.0f)',ins*1000,a.insertion_min*1000,a.insertion_max*1000),true)
end
function showEndoscope()
    if floatingView and sim.adjustView(floatingView,endoCamera,256)==2 then return end
    local r=ENDOSCOPE.floating_view
    floatingView=sim.floatingViewAdd(r[1],r[2],r[3],r[4],0)
    sim.adjustView(floatingView,endoCamera,128,'Endoscope (30 deg oblique)')
end
function active() return sim.getSimulationState()~=sim.simulation_stopped end
function run()
    if not active() then sim.startSimulation() else sim.callScriptFunction('setAuto',controller,true) end
end
function hold() if active() then sim.callScriptFunction('hold',controller) end end
function home() if active() then sim.callScriptFunction('homeRCM',controller) end end
function grip(angle)
    if not active() then return end
    local ok=sim.callScriptFunction('setGrip',controller,selected,angle)
    simUI.setLabelText(ui,5,ok and string.format('%s: jaws %.2f rad',arm().label,angle) or 'Jaw command rejected (no jaws or collision).',true)
end
function openJaws() grip(0.5) end
function closeJaws() grip(0) end
function armChanged(_,_,v)
    selected=v+1;sliders={0,0,0};wristSliders={0,0,0}
    for id=2,4 do simUI.setSliderValue(ui,id,0,true) end
    for id=10,12 do simUI.setSliderValue(ui,id,0,true) end
    updateLabels()
end
function wristApply(changed)
    if not active() then return end
    local called,state=pcall(sim.callScriptFunction,'getWristState',controller,selected)
    if not called or not state or not state.available then
        simUI.setLabelText(ui,16,'No articulated wrist on this arm/scene.',true);return
    end
    local names={'roll','pitch','yaw'};local q={}
    for i,n in ipairs(names) do
        q[i]=state[n] -- preserve the other actual wrist coordinates, including a nonzero roll home
        if i==changed then
            local lim=state.limits[n];local v=wristSliders[i]/100
            q[i]=v>=0 and v*lim[2] or -v*lim[1]
        end
    end
    local ok=sim.callScriptFunction('setWrist',controller,selected,q[1],q[2],q[3])
    simUI.setLabelText(ui,16,ok and string.format('Wrist %.1f / %.1f / %.1f deg',math.deg(q[1]),math.deg(q[2]),math.deg(q[3])) or 'Wrist rejected; previous posture retained.',true)
end
function rollChanged(_,_,v) wristSliders[1]=v;wristApply(1) end
function wristPitchChanged(_,_,v) wristSliders[2]=v;wristApply(2) end
function wristYawChanged(_,_,v) wristSliders[3]=v;wristApply(3) end
function apply()
    updateLabels()
    if not active() then return end
    local yaw,pitch,ins=values()
    local ok=sim.callScriptFunction('setRCMTarget',controller,selected,yaw,pitch,ins)
    simUI.setLabelText(ui,5,ok and string.format('%s: yaw %.2f / pitch %.2f / %.1f mm',arm().label,yaw,pitch,ins*1000) or 'Target rejected; previous configuration retained.',true)
end
function yawChanged(_,_,v) sliders[1]=v;apply() end
function pitchChanged(_,_,v) sliders[2]=v;apply() end
function insertChanged(_,_,v) sliders[3]=v;apply() end
function sysCall_sensing()
    local stats=sim.callScriptFunction('getRCMStats',controller);local text=''
    for i,a in ipairs(stats) do text=text..string.format('%s: %.6f mm (max %.6f), rejected %d\n',ENDOSCOPE.arms[i].label,a.drift_m*1000,a.max_drift_m*1000,a.failures+a.collision_blocks) end
    simUI.setLabelText(ui,6,text,true)
    sim.handleVisionSensor(endoCamera)
end
-- The endoscope camera is explicitly handled: rendered each simulation step above, and at 4 Hz while stopped.
function sysCall_nonSimulation()
    local t=sim.getSystemTime()
    if t-lastRender>0.25 then lastRender=t;sim.handleVisionSensor(endoCamera) end
end
function sysCall_cleanup()
    if floatingView then sim.floatingViewRemove(floatingView) end
    if ui then simUI.destroy(ui) end
end

function viewOverview() sim.setObjectMatrix(camera,sim.handle_world,overview) end
function viewArm(index)
    local a=ENDOSCOPE.arms[index]
    local p=sim.getObjectPosition(sim.getObject(a.reference),sim.handle_world)
    local m=sim.getObjectMatrix(sim.getObject(a.shaft),sim.handle_world)
    local x={m[1],m[5],m[9]};local y={m[2],m[6],m[10]};local up={m[3],m[7],m[11]}
    local pos={};local z={};local len=0
    for k=1,3 do pos[k]=p[k]+.16*x[k]+.10*y[k]+.025*up[k];z[k]=p[k]-pos[k];len=len+z[k]*z[k] end
    len=math.sqrt(len);for k=1,3 do z[k]=z[k]/len end
    local rx={up[2]*z[3]-up[3]*z[2],up[3]*z[1]-up[1]*z[3],up[1]*z[2]-up[2]*z[1]}
    len=math.sqrt(rx[1]^2+rx[2]^2+rx[3]^2);for k=1,3 do rx[k]=rx[k]/len end
    local ry={z[2]*rx[3]-z[3]*rx[2],z[3]*rx[1]-z[1]*rx[3],z[1]*rx[2]-z[2]*rx[1]}
    sim.setObjectMatrix(camera,sim.handle_world,{rx[1],ry[1],z[1],pos[1],rx[2],ry[2],z[2],pos[2],rx[3],ry[3],z[3],pos[3]})
end
function view1() viewArm(1) end
function view2() viewArm(2) end
function view3() viewArm(3) end
function view4() viewArm(4) end
