VIEWS={["overview"]={-0.7187440012928533,-0.25107180362786663,-0.6483594759302812,2.3,-0.6952748094139029,0.2595468012186808,0.6702450278604174,-3.25,0.0,0.9325226042301605,-0.36111160684724525,2.15},["body"]={-0.3142213908464408,-0.5489139942135464,-0.7745697802593766,1.35,-0.9493497340466938,0.18168280090166672,0.25637168783232883,-1.55,0.0,0.815895083214169,-0.5781999768133375,1.7},["pelvis"]={-0.7071067811865477,-0.45749571099781383,-0.5391638660171919,0.27441561538271186,-0.7071067811865474,0.45749571099781405,0.5391638660171921,-1.3425934034922116,0.0,0.7624928516630233,-0.6469966392206304,1.1028436670533888}}
sim=require('sim')
function bonesVisible(value) for _,name in ipairs({'RightHipBone','LeftHipBone','Sacrum'}) do sim.setObjectInt32Param(sim.getObject('/Patient_'..name),sim.objintparam_visibility_layer,value) end end
function fullBody() bonesVisible(1);sim.setObjectInt32Param(skin,sim.objintparam_visibility_layer,1);sim.setShapeColor(skin,nil,sim.colorcomponent_transparency,{1});sim.setObjectMatrix(camera,-1,VIEWS.body) end
function transparentBody() sim.setObjectInt32Param(skin,sim.objintparam_visibility_layer,1);sim.setShapeColor(skin,nil,sim.colorcomponent_transparency,{0.18});sim.setObjectMatrix(camera,-1,VIEWS.body) end
function pelvicAnatomy() bonesVisible(1);sim.setObjectInt32Param(skin,sim.objintparam_visibility_layer,0);sim.setObjectMatrix(camera,-1,VIEWS.pelvis) end
function organsOnly() pelvicAnatomy();bonesVisible(0) end
function sceneOverview() sim.setObjectMatrix(camera,-1,VIEWS.overview) end
function sysCall_init()
 skin=sim.getObject('/Patient_Skin');camera=sim.getObject('/DefaultCamera')
 if sim.getBoolParam(sim.boolparam_headless) then return end
 simUI=require('simUI')
 ui=simUI.create([[<ui title="BodyParts3D | Adult male anatomy" closeable="false" placement="relative" position="20,430" layout="vbox">
 <label text="Foot-end docking / 20 deg head-down / split legs"/>
 <button text="Whole body" on-click="fullBody"/><button text="Transparent skin" on-click="transparentBody"/><button text="Pelvic organs (hide skin)" on-click="pelvicAnatomy"/><button text="Organs only (hide pelvic bones)" on-click="organsOnly"/><button text="Robot and patient overview" on-click="sceneOverview"/>
 <label text="Purple: prostate | Yellow: bladder / ureters"/>
 <label text="Abdominal RCM ports / static anatomy / no tissue response."/>
 <label text="BodyParts3D - DBCLS - CC BY 4.0"/></ui>]])
end
function sysCall_cleanup() if ui then simUI.destroy(ui) end end
