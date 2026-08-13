# Nero 数据采集项目交接文档（Handoff）

> 面向：**新开的 Cursor / Chat 会话**（无历史对话依赖）。  
> 读完本文应能回答：项目是什么、当前硬件与平台、已改了什么、怎么启动录制、数据长什么样、下一步干什么。  
> 最后更新：2026-07-16

---

## 1. 一句话目标

用 **松灵 Nero 七轴机械臂 + Nero 自带 AgxGripper 夹爪 + 奥比中光 Dabai 相机**，在 **Windows** 上采集模仿学习数据，最终用于 **OpenPI π0.5 LoRA** 训练。  

训练侧倾向用 **末端位姿（TCP）+ 夹爪**，但**原始数据集要录全**：图像、关节角、法兰位姿、TCP、FK、夹爪；导出/训练时再选字段。

---

## 2. 当前硬件与现场约定

| 项目 | 现状 |
|------|------|
| 机械臂 | AgileX **Nero**（固件按脚本默认 **V112**） |
| 末端 | **AgxGripper**（不是强脑 / BrainCo Revo2） |
| CAN | USB **candleLight**（VID `1D50` / PID `606F`） |
| 平台 | **Windows**（从 Ubuntu 迁过来） |
| 相机 | 奥比中光 **Dabai DC1**（UVC）；OpenCV index 常为 **1**；默认 **640×480**；现场安装方向需 **顺时针 180°** 校正 |
| TCP 偏置 | 法兰坐标系 **+Z = 0.13 m**（工具沿法兰轴心向前伸出 13 cm） |

**已明确不用的东西：** 强脑灵巧手相关脚本/同步（Revo2、`brainco_*`、根目录 `arm_hand_bridge.py` 等）一律视为遗留，不要当主路径。

---

## 3. 仓库结构（只关心这些）

路径根：`E:\nero-data-collect`

```text
nero-data-collect/
  docs/HANDOFF.md                 ← 本文件
  docs/GUI_GUIDE.md               ← GUI 操作指引 + 分辨率/旋转说明
  teleop_mapping.py               # 夹爪开合 → grasp 映射（历史给 Revo2 用，现仍用于 0..1 grasp）
  arm_hand_bridge.py 等           # 强脑遗留，可忽略
  real_robot_data_collector/      # 另一套 episode 采集框架（Nero adapter 未完工；与当前主路径并行）
  pyAgxArm-master/                # ★ 当前主路径：SDK + 录制 GUI + 导出
    record_teleop.py              # 正式录制后端（JSONL + MP4）
    teleop_recorder_gui.py        # 桌面 GUI 启动器
    camera_idle_preview.py        # Dabai OpenCV 预览（含旋转）
    export_jsonl_to_lerobot.py    # JSONL → LeRobot 就绪导出（可选 state）
    win_can_selftest.py           # Windows CAN 自检
    safe_arm.py                   # Windows gs_usb / libusb 适配补丁
    recordings/                   # 录制输出目录（默认）
```

**关系说明：** `real_robot_data_collector` 与 `pyAgxArm-master` **目标同类、代码基本独立**；当前现场以 `pyAgxArm-master` 的 `record_teleop` / GUI 为准。

---

## 4. Windows CAN 要点（易踩坑）

- Ubuntu 用的是 `socketcan`（`can0`/`can1`）。  
- 本机 candleLight 在 Windows 上要用 **`interface=gs_usb`, `channel=0`**，**不是** 官方文档默认的 `agx_cando`。  
- 需要依赖：

```bash
pip install "python-can[gs-usb]" libusb-package
pip install -e E:\nero-data-collect\pyAgxArm-master
```

- `safe_arm._install_gs_usb_adapter()` 会注入 libusb，并用 `index=0` 打开设备；**不要**再读 USB `product` 字符串（Windows 上易 Access denied）。  
- 自检（只读关节，不会使能运动）：

```bash
cd E:\nero-data-collect\pyAgxArm-master
python win_can_selftest.py
```

---

## 5. 怎么启动录制（现在就能做）

### 5.1 推荐：GUI

详细操作说明见 **[docs/GUI_GUIDE.md](GUI_GUIDE.md)**。

```bash
cd E:\nero-data-collect\pyAgxArm-master
python teleop_recorder_gui.py
```

启动前确认：
1. Nero **上电**，CAN 线接到 candleLight，适配器已插 PC。  
2. Dabai 已插入。  
3. GUI 里 **CAN 通道** Windows 默认应为 **`0`**。  
4. **「Revo2 同步」保持关闭**（无强脑手）。  
5. 空格：开始/停止；Backspace：删本段；Esc：退出。  
6. 预览应已是 **正方向**（默认 `camera-rotate=180`）。

> Windows 说明：启停子进程已改为 `CREATE_NEW_PROCESS_GROUP` + `CTRL_BREAK`（不再使用 Linux 专用的 `os.killpg`）。打开目录用资源管理器。

输出默认在：

```text
pyAgxArm-master/recordings/
  nero_teleop_YYYYMMDD_HHMMSS_xxxxxx.jsonl
  nero_teleop_YYYYMMDD_HHMMSS_xxxxxx.rgb.mp4
```

### 5.2 备选：控制台交互（无 GUI）

```bash
cd E:\nero-data-collect\pyAgxArm-master
python record_teleop_console.py
```

- `Enter`：开始/停止一段  
- `d`：删除刚保存的一段  
- `q`：退出  

### 5.3 命令行单次录制

```bash
cd E:\nero-data-collect\pyAgxArm-master
python record_teleop.py --no-execute-hand --camera --camera-backend opencv --camera-index 1 --camera-rotate 180
```

Windows 会默认 `--interface gs_usb --channel 0`。Ctrl+C 停止。

### 5.4 相机分辨率与旋转

| 项 | 当前默认 | 说明 |
|----|----------|------|
| 分辨率 | **640×480** | 对 OpenPI π0.5（训练常缩到 ~224）通常够用；比 720p/1080p 更稳 |
| 旋转 | **`--camera-rotate 180`**（顺时针） | 现场 Dabai 装反；预览与 MP4 一并校正。可选 `0/90/180/270` |
| GUI 常量 | `DEFAULT_CAMERA_WIDTH/HEIGHT/ROTATE` | 在 `teleop_recorder_gui.py` 顶部改 |

细节写在 `docs/GUI_GUIDE.md` 开头。

### 5.5 环境提醒

- 本机可能 **没有 ffmpeg**；录视频会 fallback 到 OpenCV `mp4v`。**建议安装 ffmpeg**，长录更稳。  
- 图像以 **MP4** 存，不是逐帧 jpg；JSONL 用 `camera_rgb.video_frame_index` 对齐。  
- 预览用旁路 `.tick` 通知帧号，避免狂打 stdout 堵管道。

---

## 6. 当前录了什么？（schema v8）

原始格式：**自定义 `nero_teleop_jsonl` + RGB MP4**（**还不是** LeRobot v3）。

每帧（`kind: sample`）核心字段：

| 字段 | 含义 |
|------|------|
| `camera_rgb` | RGB 对齐信息（含 `video_frame_index`；meta 含 `rotate_deg_cw`） |
| `leader` / `follower` | 主/从臂关节角（rad） |
| `gripper_ctrl` / `gripper_feedback` | Agx 夹爪 CAN 原始量 |
| `gripper.state_grasp` / `action_grasp` | 归一化夹爪 0..1 |
| `poses.flange_pose` | `get_flange_pose()` |
| `poses.tcp_pose` | `get_tcp_pose()`（已设 TCP offset 0.13m） |
| `poses.fk_pose` | `fk(follower joints)` |
| `training.*` | 便于导出的向量：`vector`（关节+夹爪）、`vector_tcp_gripper`（TCP+夹爪）等 |
| `alignment` | CAN skew / 相机年龄等；GUI 用 `--keep-unaligned` 仍写入未齐行 |

连接后自动：

```python
robot.set_tcp_offset([0.0, 0.0, 0.13, 0.0, 0.0, 0.0])  # 法兰 +Z
```

---

## 7. 和 LeRobot / OpenPI 的关系

```text
录制：JSONL + MP4（全量原始）
   ↓
导出：export_jsonl_to_lerobot.py  （选择 observation.state / action）
   ↓
训练：OpenPI π0.5 LoRA
```

导出示例（训练用 TCP+夹爪）：

```bash
cd E:\nero-data-collect\pyAgxArm-master
python export_jsonl_to_lerobot.py ^
  --input-dir recordings ^
  --output-dir exports\lerobot_tcp ^
  --state-mode tcp+gripper ^
  --action-mode tcp+gripper
```

可选 mode：`joints+gripper` / `tcp+gripper` / `flange+gripper` / `fk+gripper` / `all`。  
若已安装 `lerobot`，可加 `--prefer-lerobot-api` 走官方 Dataset API；否则写 staging（npz + mp4 + meta）。

用户意图：**原始全量保存；LoRA 前自己选 TCP（不是关节）训练。**

---

## 8. 本会话 / 近期已完成的改动清单

1. Windows candleLight：`gs_usb` + `libusb-package`；`win_can_selftest.py`。  
2. 相机默认：Dabai OpenCV **index=1**；预览不再依赖 RealSense D435i。  
3. Revo2 默认同步：**关闭**。  
4. TCP 偏置 **13 cm（法兰 +Z）**。  
5. schema **v8**：法兰 / TCP / FK + 夹爪 grasp。  
6. `export_jsonl_to_lerobot.py`；无 ffmpeg 时 OpenCV `mp4v` 回退。  
7. GUI Windows 进程启停修复；`docs/GUI_GUIDE.md`。  
8. **gs_usb 相对 CAN 时间戳**：对齐改 skew；相对时钟不再与墙钟比年龄；`--keep-unaligned`。  
9. **相机长录卡死**：采集/预览/编码拆线程；约 2s 无帧强制 reopen；预览 `.tick` 旁路。  
10. **相机默认旋转 180°**（装反校正）；分辨率默认保持 640×480。

---

## 9. 已知坑 / 注意

- `get_flange_pose` 与 `fk(joints)` 可能不完全一致；**三类位姿都保留**。  
- 无强脑时不要开 GUI「末端同步」。  
- **Windows gs_usb 对齐坑（已修）**：相对 CAN 时钟不能与 `time.time_ns()` 相减。  
- **相机卡死 →「未对齐 · 相机过旧」**：旧版 `mp4v` 阻塞取流；现有看门狗重开。日志出现 `camera stalled` / `reopened` 属恢复中。装 ffmpeg 更稳。  
- 「未对齐」不等于没在录（GUI 仍写盘）；看灯旁原因即可。  
- 旋转仅影响**新录制**；旧 MP4 不会自动转正。

---

## 10. 建议的下一步（给新会话执行）

按优先级：

1. **实机长录验证**：确认 180° 方向正确，预览不卡死，JSONL/MP4 正常。  
2. **跑导出**：`--state-mode tcp+gripper`，检查 `observation.state` 维数 = 7。  
3. **对接 OpenPI π0.5 LoRA**：确认图像键（倾向 `observation.images.head`）与 config 一致。  
4. （可选）需要细节再升 **1280×720**；否则保持 640×480。  
5. （可选）清洗 GUI 残留 Revo2 文案；装 ffmpeg。  
6. （可选）LeRobot v3.0 shards / Dabai 深度。

---

## 11. 快速命令备忘

```bash
# 依赖（首次）
pip install "python-can[gs-usb]" libusb-package opencv-python
pip install -e E:\nero-data-collect\pyAgxArm-master

# CAN 自检
cd E:\nero-data-collect\pyAgxArm-master
python win_can_selftest.py

# 开始录制
python teleop_recorder_gui.py

# 导出给训练（示例：TCP + 夹爪）
python export_jsonl_to_lerobot.py --input-dir recordings --output-dir exports\lerobot_tcp --state-mode tcp+gripper --action-mode tcp+gripper
```

---

## 12. 给后续 Agent 的行为约束（摘要）

- 默认假设：**Nero + AgxGripper + Dabai + Windows candleLight/gs_usb**。  
- **不要**把强脑 Revo2 恢复成默认路径。  
- 录制主路径：`pyAgxArm-master/record_teleop.py`。  
- 相机默认：**index=1、640×480、rotate=180**；改旋转只改 `DEFAULT_CAMERA_ROTATE` / `--camera-rotate`。  
- 原始格式保持 **JSONL+MP4 全量**；训练字段选择放在导出阶段。  
- 改 TCP 长度：只改 `NERO_GRIPPER_TCP_OFFSET_M`（法兰系 +Z）。  
- 未经用户要求：**不要 commit / 不要 push**。
