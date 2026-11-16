from davinci_geometry import *
import json,shutil,xml.etree.ElementTree as E
OUT=ROOT.parent/'davinci_prostate_rcm';plan=json.loads((OUT/'registered_plan.json').read_text());validation=json.loads((OUT/'motion_validation.json').read_text());controls=json.loads((OUT/'control_validation.json').read_text())
root=E.parse(ROOT/'davinci_multiarm.urdf');r=root.getroot();r.set('name','DaVinci_FootDocking_ExploratoryRCM')
for j in r.findall('joint'):
 if j.get('name') in plan['joint_limit_revisions']:
  lo,hi=plan['joint_limit_revisions'][j.get('name')];j.find('limit').set('lower',str(lo));j.find('limit').set('upper',str(hi))
for meshnode in r.findall('.//mesh'):meshnode.set('filename',meshnode.get('filename').replace('meshes/','robot_meshes/'))
E.indent(root);root.write(OUT/'davinci_foot_docking.urdf',encoding='utf-8',xml_declaration=True);shutil.copytree(ROOT/'meshes',OUT/'robot_meshes',dirs_exist_ok=True)
home={}
for a in plan['arms']:home.update(a['q']);home[a['arm']+'_joint7']=.04
(OUT/'home_joint_positions.json').write_text(json.dumps(home,indent=2))
scripts=OUT/'scripts';scripts.mkdir(exist_ok=True)
for name in ['davinci_geometry.py','plan_prostate_rcm.py','rcm_feasibility.py','solve_prostate_layout.py','plan_foot_docking.py','build_registered_scene.py','patient_leg_pose.py','apply_leg_fix.py','capture_registered_scene.py','install_registered_guard.py','validate_registered_motion.py','test_registered_controls.py','record_foot_docking.py','package_foot_docking.py']:
 shutil.copy2(Path(__file__).with_name(name),scripts/name)
for name in ['LICENSE_source.html','README_e.html','patient_manifest.json']:shutil.copy2(ROOT.parent/'davinci_patient'/name,OUT/('source_'+name))
max_error=max(validation['max_rcm_error_m'])*1000
report=f'''脚端对接 / 腹壁 RCM / 前列腺周围接近：验证报告

当前文件：davinci_foot_docking_rcm.ttt。此前侧方摆放的候选文件不代表本次最终结果。

摆放与模型
患者车从脚端中央对接；人体头低脚高 20 度。以分腿仰卧姿态表示，各侧腿部外展约 {plan['patient_pose']['leg_abduction_each_deg']} 度。
腿部修复：选择原始外体表壳，移除反向内壳；使用连通关系区分下肢，固定骨盆与手部，近端大腿做 ARAP 局部形状保持，小腿与脚掌整体刚性旋转。
这是静态体表姿态调整，不是肌骨关节或软组织仿真；骨盆与盆腔器官保持原始相对位置。
腿部网格与无遮挡视图见 leg_pose_validation.json、leg_fix_top.png、leg_fix_oblique.png；变形旧版位于 before_leg_fix。
原始 BodyParts3D 数据及原场景保留。工作台分为躯干台面和两块腿部支撑，中央留下脚端进入空间。
患者车纵向中心、计划摄像孔和前列腺目标位于同一纵向中心平面。PlannedCameraPort 是规划标记；ECM 尚未增加本次腹壁 RCM 控制。
三个器械穿刺点由真实体表三角网格求交生成，世界坐标见 registered_plan.json；环与参考点固定在世界坐标。

运动范围与通过情况
偏航 ±1 度，俯仰 ±0.75 度，插入关节 38–42 mm（相对 40 mm 中位为 ±2 mm）。
演示依次为摆动、插入、组合运动，共 24 秒，481 个控制步。
三臂求解失败 {validation['solver_failures']}，碰撞回退次数 {validation['collision_blocks']}。
最大 RCM 数值误差 {max_error:.9f} mm；独立轴线测量见 motion_validation.json。
患者、台面、各升降安装位和固定 RCM 参考坐标的最大变化为 {validation['fixed_frame_max_change']}。
24 个手动边界组合均通过。越界请求被拒绝；临时障碍注入测试被拒绝，关节回退误差为 0，器官位置随后已恢复。

前列腺可达范围
本次到达前列腺周围的接近区域，未宣称尖端接触前列腺表面。
末端到前列腺三角网格表面的距离（整个演示周期）：
'''
for i,interval in enumerate(validation['tip_prostate_surface_distance_range_m'],1):report+=f'PSM{i}: {interval[0]*1000:.2f}–{interval[1]*1000:.2f} mm。\n'
report+='''
碰撞范围
独立验证检查 2046 对网格，包含器械/器官、器械/机械结构、机械结构/体表、工作台/腿部支撑、跨臂和同臂非相邻结构。
49 个独立碰撞采样未发现意外表面相交；运行时控制器还在每次候选姿态提交前检查相关碰撞对。
唯一允许的体表相交是各自长杆在对应腹壁 RCM 点 16 mm 半径范围内的穿刺入口相交；记录所有交线端点，未忽略整个器械/体表碰撞对。
相邻连接面的预期接触、闭合夹爪接触及内部腕部装配接触不作为意外碰撞。
碰撞检测基于三角网格表面相交；不是连续扫掠、实体包含检测或有接触力的组织求解。
先前更大的 ±2/±1.5 度、±4 mm 范围发现第三臂与髋骨相交，已缩小演示范围，原失败记录保存在 rejected_wide_motion_validation.json。

模型适用范围
源 model_manifest.json 已说明原 CAD 限位只是保守演示范围，并非厂商限位。本次为脚端对接扩大了探索性仿真限位：
安装旋转和 mount 为 ±pi，shoulder/elbow/carrier 为 ±1.2 rad；PSM1 升降为 -0.06..0.35 m，PSM2/3 为 -0.06..0.25 m。
实际采用的安装位置见 home_joint_positions.json。升降滑座仍需真实机构行程核验；扩大旋转范围也不代表真实达芬奇的硬件可用范围。
URDF 已同步上述仿真范围，网格位于 robot_meshes。URDF 保留 CAD 零位，加载后须应用 home_joint_positions.json 才是本次对接姿态。
本结果是该 CAD 串联关节模型的运动学与网格碰撞验证。闭链传动、真实硬限位、材料力学、气腹、组织牵拉及临床入路尚未验证。

Whole body 修复
当前 CoppeliaSim 颜色接口的透明度分量按不透明度使用；Whole body 应设 1，而原先设 0 令体表完全透明。
按钮现在恢复体表可见、完全不透明及骨盆显示；透明视图使用 0.18。实际渲染检查已完成。

文件
davinci_foot_docking_rcm.ttt：自包含场景，推荐直接打开。
foot_docking_rcm_demo.mp4：24 秒双视图演示。
motion_validation.json / control_validation.json：运动、相交、手动控制与回退证据。
registered_plan.json / rcm_config.json：对接、穿刺点和控制参数。
scripts：本机复现脚本，路径与端口按 D:\\仿真环境 工作区设置；依赖 work/python_deps 与 CoppeliaSim Python Remote API。

数据署名
BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International.
官方许可来源：https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html
原始 OBJ 头部保留历史许可文字；当前官方许可页副本见 source_LICENSE_source.html。
本次修改：毫米转米、整体刚体体位、体表分腿示意、着色与场景层级；盆腔器官原始相对几何未单独调整。
'''
(OUT/'验证报告.txt').write_text(report,encoding='utf-8')
print('Packaged URDF, home joints, scripts and validation report.',flush=True)
