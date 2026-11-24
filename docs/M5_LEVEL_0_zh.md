# M5：真值停刀与闭刀撤回对照（非 Level 0）

**视觉 Level 0 尚未跑通。** 这个名称只用于当前场景从未切起步、由当前 RGB-D 证据触发停刀、实际切长在 9–10 mm 且剪刀闭合回到 `ready pose` 的实跑。结构真值可用于原有守卫和离线评分，不能决定何时签发停刀。

以下是一条已完成的**真值反馈对照**。在现有 `davinci_si_articulated_verified.ttt` 的非解剖试样上，用原 `tool_pose` 和原有守卫运行完整 FSM；`control_source=truth`，读取结构切长决定是否继续。已封存的目录和文件仍保留原名 `m5_level0_v1` 以保持可追溯性，不代表取得视觉 Level 0。

| 检查点 | 本次实跑 |
| --- | --- |
| 三次接受的闭合（世界 tick 10、12、14） | 随后结构切长依次为 3、6、10 mm |
| 首次进入 9–10 mm（世界 tick 15） | 10.0 mm；同一拍触发撤回，之后没有剪刀开合命令 |
| 撤回 | 从非 ready 位出发，实际刀口闭合；回到原 ready pose 后刀口仍闭合 |
| 最终 | `CUT → DONE`，切长保持 10.0 mm，无撕裂或故障；比 9 mm 目标多 1 mm，处于原上限 |

[归档评分](../validation/task/m5_level0_v1/decision_v1_1.json) · [预登记](../validation/task/m5_level0_v1/registration.json) · [原始实跑](../validation/task/m5_level0_v1/nominal.json) · [完成画面](../validation/task/m5_level0_v1/nominal/05_DONE.png)

原始 [v1 评分](../validation/task/m5_level0_v1/decision.json) 保留不动。其唯一未通过项把 CoppeliaSim 返回的空撕裂日志 `{}` 错当成整数计数。v1.1 仅修正该日志的类型判断，在相同记录上确认零撕裂；未重跑或修改控制器、场景与门槛。

## 复现

启动一个独立的 CoppeliaSim Edu 4.10 实例，使用独立配置和 localhost RPC 端口。以下命令使用已经预留的 `24731` 端口；指定新的输出目录和 ASCII 暂存路径。第一条保留 v1 评分（空日志在 v1 会得到 `FAIL`），第二条在同一数据上给出修正版判定。

```powershell
$out = 'validation/task/m5_level0_reproduce_01'
$stage = Join-Path $env:TEMP 'davinci_m5_level0_reproduce_01'
& '.\.venv\Scripts\python.exe' scripts/task/run_m5_level0.py --host 127.0.0.1 --port 24731 --stage-dir $stage --out $out
& '.\.venv\Scripts\python.exe' scripts/analysis/correct_m5_level0_score.py --folder $out
```

结束后关闭独立仿真实例。本对照不给 [严格视觉 M5](M5_ENDPOINT.md) 验收记分，也未评估跨场景可靠性、M6 或 M7。
