# Si 末端运动学与 P0 修复记录（2026-11-17）

## 当前结论

原场景不是完整 EndoWrist：joint8/9 有 roll/pitch 实体，但 yaw 壳体并入 pitch，双爪仅同角开合，缺少独立 wrist yaw。
面板原有 yaw/pitch 是 RCM 杆轴摆角，不能算作腕部自由度。

本轮提供独立新场景，不覆盖历史场景：

`D:\仿真环境\github_upload\davinci-multiarm-rcm-simulation\scenes\davinci_si_articulated_verified.ttt`

构建与验收结果：

`D:\仿真环境\github_upload\davinci-multiarm-rcm-simulation\validation\safety\articulation_verified.json`

旧 `davinci_si_instruments.ttt` 不含新增 yaw 关节；刷新 Lua 不能凭空修正其几何。
`articulated.ttt`、`articulated_v2.ttt` 是本轮中间验证场景；后续以 `articulated_verified.ttt` 为入口。

## 独立自由度与 API

| 量 | 实现 | 单位/定义 |
|---|---|---|
| roll | 原 joint8 | 轴向旋转，rad |
| wrist pitch | 原 joint9 | 腕俯仰，rad |
| wrist yaw | 新 `<arm>_wrist_yaw_joint` | 在 pitch 后，整体承载 yaw 壳、爪/滑轨、TCP，rad |
| jaw opening | 原 joint10/11 | 两轴反向，各取半开口角，合计一个独立开合 DOF |
| ProGrasp hinge slide | 从属棱柱关节 | 由总开口驱动，不是第五个独立 DOF |

`setWrist(index, roll, pitch, yaw)` 使用 rad；`getWristState(index)` 返回实际读回状态。
旧 `setGrip(index,a)` 保持兼容：a 为每爪半开口，返回 `jaw=2*a` 才是总开口。
`setRCMTarget` 的 yaw/pitch 仍为杆轴摆角（degree），不是腕部角度。

若把两片爪都表示成同一正轴下的叶片角，理想同轴情形是
`theta_plus = yaw + opening/2`，`theta_minus = yaw - opening/2`。
旧场景两关节轴相反，不能直接把这两个数原样写入它们；且偏置铰点必须随 yaw 转动，因此本轮采用显式串联 yaw，而不是仅差动旧爪角。

## 参数来源与保留边界

- 几何：`jhu-dvrk/dvrk_model` 固定提交 `90472a933995acaa88d84eb5c58aeee4cd17e9f3`，本地 SOURCE.json 记录每文件 SHA-256。
- 关节参数：`jhu-dvrk/sawIntuitiveResearchKit` 固定提交 `7435458dc847b95b9ff77723b33e4a177cfa4f43` 的对应 tool JSON。
- Maryland 420172：roll 约 ±255°、pitch ±70°、yaw ±90°；弯剪 420179 和 ProGrasp 420093：约 ±260°、±80°、±80°。是所引用研究软件参数，不代表本 CAD 全范围可达或临床认证。
- 硬件 jaw JSON 的负角用于闭合/力矩控制，不应在纯几何模型中解释成两爪互相穿过。新场景几何开口下限为 0。
- ProGrasp 几何 xacro 的滑轨关系只给出 0..60° 总开口。本轮新场景限制至该范围，不将钳口 JSON 的更宽范围误称为已校准几何。
- CAD joint8 到 pitch 的原距离 0.4301 m 保留。dVRK DH 的 D=0.467 m 来自另一套坐标约定，不能直接替换并宣称实机一致。
- `tcp` 位置使用闭合几何尖端，方向使用 dVRK tool_tip 的相对旋转；另有 `<arm>_dvrk_tool_tip` 保存官方 tool_tip 偏置。两者不要混用，开爪时参考 TCP 不是任一真实叶片顶点。
- 弯剪铰点的 2.45 mm 修正是原项目网格测量设定，不是厂商标定值。

## 已实现的保护

- 腕命令先检查有限数、逻辑/场景限位，再以不超过 0.02 rad 的关节增量检查碰撞；拒绝后恢复三腕关节，不改变开合量。
- 拆分壳体后补充外部/跨臂碰撞对和非相邻末端—近端碰撞对；相邻铰链装配面仍按模型规则排除。
- 家位动作返回显式结果，失败回退；不能把小步检查等同连续扫掠安全证明。
- 合刀前重新检查全部观测条件，包括晚发生故障、撕裂、滑脱、应变异常、遮挡。
- 删除“组织颜色直接豁免深度遮挡”；已知点中心深度与周边支持都必须通过。
- 闭合失败不建立抓持；张口释放；未达到最小节点数时清除残留固定。
- 根据旧/新实际角度分别计算两刀扫过区域，保留完整闭合周期；回归覆盖完整/分步闭合、部分闭合不提前切中心区。
- 切割扇区最大约 0.05 rad，是数值近似；不宣称任意网格/任意分段绝对相同。
- 成功需要回撤和最终验收；过切量独立报告。当前一行网格（1 mm）容差是离散 benchmark 选择，不是临床安全界限。
- 视觉实验使用 run_id 与退出码拒绝旧报告；切割 runner 的异常与清理失败写为本次失败，报告原子替换。

## 运行与验证

使用已安装 requirements-build.txt 的 Python（项目不新增第三方框架）：

```powershell
python -B -m unittest discover -s 'D:\仿真环境\github_upload\davinci-multiarm-rcm-simulation\scripts\tests' -v
python -B -m unittest discover -s 'D:\仿真环境\github_upload\davinci-multiarm-rcm-simulation\scripts\task' -p test_cut_fsm.py -v
```

Lua 回归实际执行原 Lua 源码，仿真 API 使用桩；Windows 可用安装目录的 lua.dll，其他机器通过 `LUA_LIBRARY` 指定兼容 Lua 运行库。缺运行库会明确失败，不会静默视为通过。

真实切割复核脚本（新报告，不改历史报告）：

```powershell
python -B 'D:\仿真环境\github_upload\davinci-multiarm-rcm-simulation\scripts\tests\live_cut_safety_check.py' --host 127.0.0.1 --port 23271 --mode full --scene 'D:\仿真环境\github_upload\davinci-multiarm-rcm-simulation\scenes\davinci_si_articulated_verified.ttt' --out 'D:\仿真环境\github_upload\davinci-multiarm-rcm-simulation\validation\safety\verified_full_nominal.json' --launch-simulator 'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe'
```

启动专用实例使用独立设置目录后缀，非交互 Windows 启动必须保持 stdin 管道打开（本轮已用 A/B 测试确认）；不修改正常实例设置。只保存新输出场景，不覆盖原二进制。

## 尚未完成

### 本轮实测

- 离线回归：152 项通过，0 失败、0 跳过，证据在 `validation/safety/regressions.json`。
- 最终场景：三臂单轴 FK、官方 TCP 相对方向、组合腕运动与开合、独立 RCM 残差和器械识别通过；见 `articulation_verified.json`。
- 最终场景完整三臂任务：nominal 与 persistent occlusion 均 DONE、切长 10 mm、撕裂 0；见 `verified_full_nominal.json`、`verified_full_occlusion.json`。
- 任务目标 9 mm，实际 10 mm 使用已明示的一行网格容差通过，不描述成“无过切”。这两次运行不是统计可靠性证明，也不是解剖筋膜任务。
- 默认切割入口已切换到最终新场景；构建脚本仍保留旧场景作为迁移输入，旧演示报告不覆盖。

本轮末端验证不等于整机 dVRK 闭链/传动标定；正运动学数值残差不是实机定位精度。全关节工作空间、组织接触力和参数校准仍未通过。
P1 资产审计不等于解剖任务场景建成。切割 World 仍读组织真值，P2 尚未达成；P3 还无联合控制对照；P4 无材料/断裂标定证据。详见 DEVELOPMENT_ROADMAP_zh.md。