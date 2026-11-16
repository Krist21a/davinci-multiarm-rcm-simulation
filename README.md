# 达芬奇式多臂机器人：脚端对接与腹壁 RCM 仿真

最新场景包含三条持针钳器械臂、CAD 镜臂、BodyParts3D 成年男性及分段手术台。患者采用头低脚高 20°、每侧分腿 20° 的静态示意姿态，机器人从脚端中央对接。

**已修复腿部拉伸：**移除内体表壳造成的重叠，按网格连通关系处理下肢，固定骨盆与手部，近端大腿采用局部形状保持，小腿与脚掌整体刚性旋转。原始解剖网格仍保留。

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

## 数据来源与署名

BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International.

[官方许可页](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html) · [官方数据下载](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html)

使用官方 `isa_BP3D_4.0_obj_99.zip` 中选定的模型。原始 OBJ 头部含历史许可文字；当前官方许可页副本保存在 `models/anatomy/source_LICENSE_source.html`。修改包括毫米转米、刚体体位、外体表筛选、分腿姿态、着色与仿真层级。CAD 和持针钳来自用户提供的项目资产，其许可与 BodyParts3D 分开处理。