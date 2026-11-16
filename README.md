# 达芬奇式多臂机器人：脚端对接与腹壁 RCM 仿真

最新场景包含三条持针钳器械臂、CAD 镜臂、BodyParts3D 成年男性及分段手术台。患者采用头低脚高 20°、每侧分腿 20° 的静态示意姿态，机器人从脚端中央对接。


![最新场景](docs/gui_whole_body_verified.png)

[24 秒双视图演示](docs/foot_docking_rcm_demo.mp4) · [无遮挡腿部检查](docs/leg_fix_top.png) · [详细验证报告](docs/foot_docking_validation_zh.txt)

## 直接打开

1. 使用 CoppeliaSim Edu 4.10 打开 [scenes/davinci_foot_docking_rcm.ttt](scenes/davinci_foot_docking_rcm.ttt)。场景自包含网格与 Lua 控制器，不需要 Python、ROS 或 MATLAB。
2. 点击播放：循环演示摆动、插入、组合运动。橙色圆环为固定腹壁 RCM 点。
3. 使用 RCM 面板选择器械臂并调整 yaw、pitch、insertion；越界或碰撞请求会被拒绝并回退。
4. 人体面板可切换 Whole body、透明体表、盆腔和隐藏骨盆视图。Whole body 已修复为可见且完全不透明。

## 本次验证范围

| 项目 | 结果 |
|---|---|
| 摆动范围 | yaw ±1°、pitch ±0.75° |
| 插入关节范围 | 38–42 mm，中位 40 mm |
| 连续周期 | 24 秒，481 个控制步 |
| 求解失败 / 演示碰撞回退 | 0 / 0 |
| 最大 RCM 数值残差 | 约 0.000019 mm，不代表实机精度 |
| 独立网格相交抽检 | 49 次 × 2046 对网格，无意外相交 |
| 手动边界目标 | 24/24 通过 |
| 障碍注入测试 | 拒绝运动、精确回退，测试障碍已恢复 |
| 器械尖端距前列腺表面 | 约 20–29 mm，为周围接近区域，尚未接触组织 |

控制器在候选姿态提交前检查相关碰撞对。只允许器械长杆在各自腹壁 RCM 点 16 mm 半径内的预期体表穿越，不忽略整个人体碰撞对。相邻装配面、闭合夹爪及内部腕部配合面不属于意外碰撞。检查基于三角网格表面相交，不等同于连续扫掠、实体包含检测或接触力求解。

## 模型边界

- 这是 CAD 等效串联关节链的软件 RCM；真实闭链传动、机械 RCM 耦合与厂商限位尚未标定。
- 为脚端对接调整了探索性仿真关节范围：setup/mount 为 ±π，shoulder/elbow/carrier 为 ±1.2 rad；PSM1 升降 −0.06..0.35 m，PSM2/3 为 −0.06..0.25 m。**这些范围不是已验证的实机行程。**
- 腿部姿态是静态体表处理，不是肌骨关节或软组织仿真。腹壁与盆腔器官位置未单独调整。
- 摄像孔是规划标记；ECM 尚未加入此次腹壁 RCM 控制。
- 尚无组织变形、气腹、牵拉、缝线、切割或临床入路验证。扩大摆动范围曾出现髋骨相交，不能把本次局部通过结果推广到整个工作空间。

## 文件与复现

- `scenes/davinci_foot_docking_rcm.ttt`：当前推荐场景。
- `models/davinci_foot_docking.urdf` + `models/meshes/`：同步后的机器人模型；URDF 保留 CAD 零位，须应用 `config/home_joint_positions.json` 才是本次对接姿态。
- `models/anatomy/`：原始选定器官 OBJ、修复后的体表 NPZ、来源与许可副本。
- `config/`：最新穿刺点、初始关节位置、姿态及碰撞对。
- `src/rcm_guarded_embedded.lua`：场景使用的完整受限位与碰撞保护控制器；其余 Lua 为基础求解器和面板。
- `validation/foot_docking/`：最新运动、腿形、控制、视频及可见窗口验证记录。
- `validation/latest_asset_checksums.json`：本次资产 SHA-256 与大小。
- `scripts/authoring/`：本机作者脚本，包含原工作区路径和开发端口；不是开箱即用的跨平台构建命令。使用场景无需运行它们。

独立读取运行中的 RCM 残差：

```bash
python -m pip install -r requirements.txt
python scripts/verify_rcm.py --port 23000 --samples 100 --out runtime_validation.json
```

端口需与 CoppeliaSim ZeroMQ Remote API 一致。默认使用新版 `config/rcm_config.json`。

旧版无人体的 [davinci_three_arm_rcm.ttt](scenes/davinci_three_arm_rcm.ttt) 仍保留，验证时传 `--config config/legacy_rcm_config.json`。旧版源码在 `src/legacy/`，早期验证在 `validation/` 根目录；它们不代表当前解剖注册版本。

## 内窥镜与视觉闭环场景（开发中）

[scenes/davinci_endoscope_task.ttt](scenes/davinci_endoscope_task.ttt) 在上面的推荐场景基础上加入了 ECM 软件 RCM、30° 斜视内窥镜相机，以及一个由内窥镜引导、PSM2 完成的拾取放置任务。基础场景文件未被修改。

- **ECM**：镜杆穿过 `PlannedCameraPort`，7 个旋转关节保持镜杆位置、方向和图像滚转；手动范围 yaw/pitch ±6°，插入 70–90 mm。PSM 演示时 ECM 保持不动。
- **内窥镜相机** `EndoscopeCamera`：640×480，水平视场 70°，裁剪面 2–300 mm，图像上方对应患者腹侧。面板中的 “Show endoscope view” 打开浮动视图；“Open/Close jaws” 控制所选 PSM 的夹爪。
- **PSM2 重新摆位**：在零空间内调整手臂，器械位姿不变；摆位同时满足 ECM 和 PSM2 在各自运动范围角点上互不相交。关节变化见 `config/endoscope/ecm_plan.json`。
- **任务物体**（`config/endoscope/task.json`）：绿色珠子 `TaskBead`（直径 5 mm）、蓝色目标环 `TaskTarget`（半径 5 mm）、PSM2 腕部品红标记球 `psm2_TipMarker`（半径 6 mm）。物体放在 PSM2 回撤后能进入、且内窥镜中无遮挡的骨盆上方空腔内，两者相距 12.5 mm。它们是无重力、无接触力的静态夹具，抓取通过把珠子挂到夹爪尖端坐标系实现。
- **探索性限位，不是实机行程**：
  - ECM：setup/mount ±π，shoulder/elbow/carrier ±1.6 rad，升降 −0.20..0.25 m，插入 −0.03..0.10 m。
  - PSM2 手动范围：yaw/pitch ±5°，插入 0–45 mm（自动演示幅度仍为 ±1°/±0.75°/±2 mm）。

### 视觉闭环

`scripts/vision/visual_servo_task.py` 只使用内窥镜 RGB 图、仿真深度缓冲和相机内参：

- **感知**：按颜色分割，把深度反投影成三维点。珠子和标记球用已知半径做球拟合，目标环用平面加圆拟合。
- **控制**：基于位置的视觉伺服（PBVS）。在相机坐标系计算误差，积分到夹爪抓取点的目标位置，再按固定 RCM 几何换算成 yaw/pitch/insertion，交给场景内带碰撞保护的 RCM 控制器执行。
- **流程**：一条开环指令把器械带入视野，之后全部由视觉驱动：预抓取 → 抓取 → 夹紧并挂接 → 抬起 → 预放置 → 放置 → 松开 → 回撤。
- **真值的用途**：只用于日志记录和最终成功判定。

本次结果（`validation/endoscope/visual_task_summary.json`）：6 次运行全部成功，其中 1 次标称，5 次加入相机位姿误差 2°+3 mm 与深度噪声 σ=0.5 mm。最终珠子到目标环中心的真值误差为 0.18–0.58 mm。指令拒绝和意外相交均为 0，PSM2 RCM 残差小于 1e-7 m。

### 重建与回归（需要一个专用的 CoppeliaSim 实例）

```powershell
python -m pip install -r requirements-build.txt
# 启动专用实例，端口不要与其他实例重复
& "C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe" -h -GzmqRemoteApi.rpcPort=23170
cd scripts
python build/plan_ecm_docking.py --port 23170 --only 0.14,-60  # ECM + PSM2 对接 → config/endoscope/ecm_plan.json（约 3.5 分钟）
Remove-Item ../config/endoscope/task.json -ErrorAction SilentlyContinue
python build/build_endoscope_scene.py --port 23170             # 先建不含任务物体的场景
python build/plan_visual_task.py --port 23170                  # 选取可见且可达的拾取/放置点 → config/endoscope/task.json
python build/build_endoscope_scene.py --port 23170             # 加入任务物体后重建
python build/validate_endoscope_scene.py --port 23170          # 场景回归 → validation/endoscope/scene_validation.json
python vision/run_visual_task_suite.py --port 23170            # 视觉闭环 6 次运行 → validation/endoscope/visual_task_*.json
# 独立读取运行中四臂（含 ECM）的 RCM 残差
python verify_rcm.py --port 23170 --start --config ../config/endoscope/rcm_config.json
```

单一事实来源是 `src/endoscope/*.lua` 与 `config/endoscope/endoscope.json`。场景内嵌脚本由构建脚本生成，回归脚本会检查内嵌文本与源文件是否逐字一致。场景文件经 ASCII 临时目录中转，因为 CoppeliaSim 4.10 的 Remote API 无法处理中文路径。

### 已知限制

- 静态解剖：在对接插入深度下，PSM1/PSM3 尖端被骨盆遮挡，前列腺被膀胱和耻骨完全遮挡。任务因此放在可见空腔内，而不是前列腺上。
- 拾取放置是运动学示意：没有重力、接触力、组织形变；腕部标记球是仿真用的视觉标记。
- 被夹持的珠子由独立扫描检查是否与解剖结构相交；运行时碰撞保护只覆盖珠子与 PSM2 器械杆、其他臂的组合，不检查珠子与夹爪之间。
- 相机位姿来自 ECM 运动学；鲁棒性测试只注入了上述固定幅度的误差，没有做标定流程。

基础场景中发现的问题：PSM2 的 mount/shoulder 在 ECM 停放位已与 ECM 网格相交，原碰撞对没有覆盖这类组合；`config/collision_pairs.json` 中有 280 对重复。新场景的碰撞对已去重，并补齐了全部 ECM 与任务物体相关组合。

## 数据来源与署名

BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International.

[官方许可页](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html) · [官方数据下载](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html)

使用官方 `isa_BP3D_4.0_obj_99.zip` 中选定的模型。原始 OBJ 头部含历史许可文字；当前官方许可页副本保存在 `models/anatomy/source_LICENSE_source.html`。修改包括毫米转米、刚体体位、外体表筛选、分腿姿态、着色与仿真层级。CAD 和持针钳来自用户提供的项目资产，其许可与 BodyParts3D 分开处理。
