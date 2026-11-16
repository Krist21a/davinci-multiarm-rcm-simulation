sim=require('sim')
if sim.getBoolParam(sim.boolparam_headless) then return end
simUI=require('simUI')
function sysCall_init()
    controller=sim.getObject('/RCMController');selected=1;yaw=0;pitch=0;insert=40
    camera=sim.getObjectsInTree(sim.handle_scene,sim.sceneobject_camera,0)[1]
    overview=sim.getObjectMatrix(camera,sim.handle_world)
    ui=simUI.create([[<ui title="Abdominal RCM | foot-end docking" closeable="false" placement="relative" position="20,20" layout="vbox">
    <label text="Registered skin ports / guarded kinematic control"/>
    <group layout="hbox"><button text="Overview" on-click="viewOverview"/><button text="RCM 1" on-click="view1"/><button text="RCM 2" on-click="view2"/><button text="RCM 3" on-click="view3"/></group>
    <button text="Run RCM demo: swing / insert / combined" on-click="run"/>
    <button text="Hold all arms" on-click="hold"/>
    <button text="RCM home" on-click="home"/>
    <combobox id="1" on-change="armChanged"><item>PSM 1</item><item>PSM 2</item><item>PSM 3</item></combobox>
    <label text="Yaw (-1 to +1 deg)"/><hslider id="2" minimum="-10" maximum="10" on-change="yawChanged"/>
    <label text="Pitch (-0.75 to +0.75 deg)"/><hslider id="3" minimum="-75" maximum="75" on-change="pitchChanged"/>
    <label text="Insertion (38 to 42 mm)"/><hslider id="4" minimum="38" maximum="42" value="40" on-change="insertChanged"/>
    <label id="5" text="Start simulation to enable constrained controls."/>
    <label id="6" text="RCM residuals will appear here."/>
    <label text="Exploratory CAD joint ranges; no clinical validation."/>
    </ui>]])
end
function active() return sim.getSimulationState()~=sim.simulation_stopped end
function run()
    if not active() then sim.startSimulation() else sim.callScriptFunction('setAuto',controller,true) end
end
function hold() if active() then sim.callScriptFunction('hold',controller) end end
function home() if active() then sim.callScriptFunction('homeRCM',controller) end end
function armChanged(_,_,v) selected=v+1 end
function apply()
    if not active() then return end
    local ok=sim.callScriptFunction('setRCMTarget',controller,selected,yaw,pitch,insert/1000)
    simUI.setLabelText(ui,5,ok and string.format('PSM %d: yaw %.1f / pitch %.1f / %.0f mm',selected,yaw,pitch,insert) or 'Target rejected; previous configuration retained.',true)
end
function yawChanged(_,_,v) yaw=v/10;apply() end
function pitchChanged(_,_,v) pitch=v/100;apply() end
function insertChanged(_,_,v) insert=v;apply() end
function sysCall_sensing()
    local stats=sim.callScriptFunction('getRCMStats',controller);local text=''
    for i,a in ipairs(stats) do text=text..string.format('PSM %d: %.6f mm (max %.6f), rejected %d\n',i,a.drift_m*1000,a.max_drift_m*1000,a.failures+a.collision_blocks) end
    simUI.setLabelText(ui,6,text,true)
end
function sysCall_cleanup() if ui then simUI.destroy(ui) end end

function viewOverview() sim.setObjectMatrix(camera,sim.handle_world,overview) end
function viewArm(index)
    local p=sim.getObjectPosition(sim.getObject('/psm'..index..'_RCM_reference'),sim.handle_world)
    local m=sim.getObjectMatrix(sim.getObject('/psm'..index..'_joint8'),sim.handle_world)
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
