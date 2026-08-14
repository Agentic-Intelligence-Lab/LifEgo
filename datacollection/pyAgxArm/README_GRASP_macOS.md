# NERO 机械臂 + 强脑 Revo2 灵巧手 抓取控制（macOS 实战指南）

> 本文档是在官方 `README.md` 基础上的补充，记录了一次**在 macOS 上从零打通 AgileX NERO 机械臂 + 强脑 Revo2 灵巧手抓取控制**的完整过程，包含：自研安全封装、抓取任务脚本、以及 macOS 下 candleLight（gs_usb）连接的全部排查与解决方案。
>
> 适用硬件：NERO 七轴机械臂 + 强脑 Revo2/RENV2 灵巧手 + candleLight USB-CAN 适配器（`0x1d50:0x606f`）。

---

## 目录

- [1. 新增文件清单](#1-新增文件清单)
- [2. 硬件与通信背景](#2-硬件与通信背景)
- [3. 环境安装](#3-环境安装)
- [4. macOS 连接排查历程（重要）](#4-macos-连接排查历程重要)
- [5. 使用步骤](#5-使用步骤)
- [6. 抓取任务流程](#6-抓取任务流程)
- [7. 防撞桌面的安全机制](#7-防撞桌面的安全机制)
- [8. 故障排查速查表](#8-故障排查速查表)
- [9. 长期建议：迁移到 Linux](#9-长期建议迁移到-linux)

---

## 1. 新增文件清单

本指南配套以下自研脚本（均位于仓库根目录）：

| 文件 | 作用 |
| --- | --- |
| `safe_arm.py` | **核心安全封装**。`SafeNeroArm` 类 + `SafetyConfig` 配置，提供带"软地板防撞"的运动接口、灵巧手控制、急停，以及 macOS 的 gs_usb 适配层。 |
| `grasp_task.py` | **抓取任务脚本**。按"回 home → 张手 → 预抓取 → 慢速靠近 → 闭合 → 抬起 → 放置 → 松开 → 撤回"的 10 步完整 pick-and-place。 |
| `selftest_connect.py` | **只读连接自检**。不使能、不运动，仅验证能否读到机械臂关节角。 |
| `can_sniff.py` | **底层 CAN 嗅探**。打印总线原始帧，判断是否有流量（支持指定波特率）。 |
| `gs_diag.py` | **candleLight 深度诊断**。读设备时钟/固件，并用 listen-only 模式判定波特率是否匹配、总线是否有流量。 |

---

## 2. 硬件与通信背景

- **机械臂**：AgileX NERO（7 轴），通过 CAN 总线通信，默认波特率 **1 Mbps**。
- **灵巧手**：强脑 Revo2/RENV2，作为末端执行器（`effector`）挂在同一套 SDK 下，手指位置范围 `0~100`（0=张开，100=闭合）。
- **USB-CAN 适配器**：**candleLight**（bytewerk，`VID=0x1d50 PID=0x606f`）。

> ⚠️ **关键点**：candleLight 用的是 **gs_usb** 协议（原生 USB 设备），**不是** 官方 README 默认的 macOS `slcan`。它**不会**出现在 `/dev/tty.*` 里，所以在 macOS 上找不到 `usbmodem`/串口是正常现象。本指南的 `safe_arm.py` 已内置 gs_usb 适配层自动处理。

---

## 3. 环境安装

```bash
# 1) 基础依赖
python3 -m pip install "python-can>=3.3.4"

# 2) 安装本仓库的 pyAgxArm
cd /Users/bochen/Downloads/pyAgxArm-master
python3 -m pip install .

# 3) macOS 使用 candleLight 必须的 gs_usb 相关依赖
python3 -m pip install gs_usb pyusb
brew install libusb     # 若未安装
```

验证：

```bash
python3 -c "import can, pyAgxArm, gs_usb, usb.core; print('依赖 OK')"
```

> **注意 Python 解释器一致性**：本项目的包安装在
> `/Library/Frameworks/Python.framework/Versions/3.9/bin/python3`。
> macOS 上需要 `sudo` 运行（见下文），务必用**同一个完整路径的 python3**，
> 不要用 conda 的 `(base)` python，否则会找不到已装的包。

---

## 4. macOS 连接排查历程（重要）

这次从"完全连不上"到"成功读到数据"，依次踩过并解决了 **5 个坑**，按出现顺序记录如下：

### 坑 1：用错了 Linux 的 CAN 激活命令
- **现象**：`sudo: ip: command not found`。
- **原因**：`ip link set can0 up` 是 **Linux** 激活 socketcan 的方式，**macOS 没有**。
- **解决**：macOS 走 gs_usb，无需 `ip link`。

### 坑 2：依赖未安装 + 目录不对
- **现象**：`ModuleNotFoundError: No module named 'can'`；`can't open grasp_task.py`。
- **原因**：未装 `python-can`；且当时不在脚本所在目录。
- **解决**：见[第 3 节](#3-环境安装)安装依赖，并 `cd` 到 `pyAgxArm-master`。

### 坑 3：CAN 模块是 candleLight（gs_usb），不是 slcan
- **现象**：`/dev/tty.*` 里只有 FTDI 的 `usbserial-FTAN8U0H*`（其实是 Rokoko 设备），找不到机械臂串口。
- **原因**：candleLight 是原生 USB（gs_usb），不枚举为串口。
- **解决**：`safe_arm.py` 增加 gs_usb 适配层 `_install_gs_usb_adapter()`，自动把 `channel` 改成 USB product id、补 `index`、剔除 gs_usb 不支持的 `local_loopback` 参数。macOS 默认 `interface="gs_usb"`。

### 坑 4：libusb 访问 USB 需要 sudo
- **现象**：`USBError(13, 'Access denied (insufficient permissions)')`。
- **原因**：macOS 上 libusb 访问 USB 设备需要 root 权限。
- **解决**：所有连接/控制脚本都用 `sudo` 运行（见[第 5 节](#5-使用步骤)）。

### 坑 5：gs_usb 库的 `start()` 在 macOS 崩溃
- **现象**：`USBError [Errno 2] Entity not found`，发生在 `is_kernel_driver_active`。
- **原因**：gs_usb 库的 `start()` 为 Linux 写死了内核驱动 detach 逻辑，macOS 的 libusb 不支持该调用。
- **解决**：`safe_arm.py` 的 `_patch_gs_usb_start_for_macos()` 用安全版 `start()` 覆盖它（把内核驱动操作包进 `try/except`）。该补丁会被 `_install_gs_usb_adapter()` 自动调用。

### 还会遇到的两个"非错误"现象
- **退出时 `segmentation fault`**：gs_usb 后端关闭时的已知崩溃，**不影响功能**。脚本已用 `os._exit(0)` 规避。
- **listen-only 模式只看到一个 ID 疯狂刷屏**（如 `0x2A5`，5 秒数万帧）：因为监听模式**不发 ACK**，机械臂那一帧得不到应答便按 CAN 协议**无限重传**。这恰恰证明总线有流量、波特率正确；正常读取要用 **NORMAL 模式**（由 SDK 自动 ACK）。

---

## 5. 使用步骤

> macOS 上全部命令需 `sudo` + 完整路径 python3。

### 第 1 步：底层诊断（确认总线有数据）

```bash
cd /Users/bochen/Downloads/pyAgxArm-master
sudo /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 gs_diag.py
# 可指定波特率：gs_diag.py 500000
```

期望：`CAN 时钟 fclk_can = 48 MHz`、`实际 1000000 bps`、`LISTEN-ONLY 收到 N 帧`（N>0）。

### 第 2 步：只读连接自检（NORMAL 模式读关节角）

```bash
sudo /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 selftest_connect.py
```

期望：`✅ 收到关节角: [...]`。看到即代表读取链路完全打通。

### 第 3 步：运行抓取任务（会真正使能并运动！）

```bash
sudo /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 grasp_task.py
```

> 🛑 **运行前务必**：先空载、低速、手放急停旁，并按实物修改 `grasp_task.py` 里的 `SafetyConfig` 和 `GraspPoses`（见下文）。

---

## 6. 抓取任务流程

`grasp_task.py` 中 `run_grasp()` 严格对应以下 10 步：

| 步骤 | 动作 | 实现 |
| --- | --- | --- |
| 1 | 机械臂回安全初始位 | `arm.move_j(home_joints)` |
| 2 | 灵巧手张开 | `arm.open_hand()` |
| 3 | 移动到物体上方/前方 | `arm.move_tcp_p(pre_grasp)` |
| 4 | 慢速靠近物体 | `arm.move_tcp_l(grasp, speed_percent=slow)` |
| 5 | 灵巧手闭合（接触自适应） | `arm.close_hand_until_contact(...)` |
| 6 | 等待抓稳 | `time.sleep(...)` |
| 7 | 抬起物体 | `arm.move_tcp_l(lift, ...)` |
| 8 | 移动到放置位置 | `arm.move_tcp_p(place, ...)` |
| 9 | 张开灵巧手释放 | `arm.open_hand()` |
| 10 | 机械臂撤回 | `arm.move_j(home_joints)` |

**上机前必须按实物修改的参数**（都在 `grasp_task.py` 的 `main()` 里）：

```python
SafetyConfig(
    hand_length=0.16,   # 灵巧手指尖沿法兰 +Z 的伸出长度 (m)
    table_z=0.0,        # 桌面在基座坐标系的高度 (m)
    safe_margin=0.05,   # 指尖距桌面安全余量 (m)
)
GraspPoses(...)         # home_joints + pre_grasp/grasp/lift/place 五个位姿
```

> 位姿均为**指尖（TCP）**在基座系下的 `[x, y, z, roll, pitch, yaw]`；`home` 用关节角。当前都是占位示例值，**必须**改成你的工位实测值。

---

## 7. 防撞桌面的安全机制

`SafeNeroArm` 内置三道闸，避免灵巧手撞桌面：

1. **TCP 偏置**：`set_tcp_offset([0,0,hand_length,...])` 把控制点移到指尖，所有高度判断都基于指尖而非法兰。
2. **Z 软地板**：任何运动目标下发前自动把指尖 Z 夹到 `table_z + safe_margin` 之上；`move_j` 还会用 `fk()` **离线预判**，过低直接拒绝。
3. **实时监控 + 底层兜底**：运动中持续读 `get_tcp_pose`，指尖逼近真实桌面立即急停；外加低速、关节软限位、末端限速、碰撞防护。

其它安全能力：`emergency_stop()` 急停（关节带阻尼缓降）、`reset()` 复位、`clear_errors()`、单步临时调速、灵巧手电流接触判定 `close_hand_until_contact()`、`with` 退出/`Ctrl+C` 自动急停+失能。

---

## 8. 故障排查速查表

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `command not found: #` | 把注释行也粘进了终端 | 忽略，不要整段连注释粘贴 |
| `sudo: ip: command not found` | 用了 Linux 命令 | macOS 走 gs_usb，无需 `ip link` |
| `No module named 'can'/'gs_usb'` | 依赖没装 | 见[第 3 节](#3-环境安装) |
| 找不到 `/dev/tty.usbmodem*` | candleLight 是 gs_usb 非串口 | 正常现象，用 gs_usb |
| `Access denied (permissions)` | libusb 需要 root | 加 `sudo` |
| `Entity not found` (start) | gs_usb 库 macOS 兼容问题 | 已由 `_patch_gs_usb_start_for_macos()` 修复 |
| 退出时 `segmentation fault` | gs_usb 关闭已知崩溃 | 无害，已用 `os._exit` 规避 |
| 收到 0 帧 | 物理/波特率问题 | `gs_diag.py` 诊断；万用表量 CANH-CANL≈120Ω；换波特率 |
| listen-only 只刷一个 ID | 监听模式不发 ACK 致重传 | 正常；用 NORMAL 模式（selftest）即可 |
| `selftest` 未收到关节角 | 多为上层 ID/固件配置 | 检查 `NeroFW` 版本是否匹配固件 |

---

## 9. 长期建议：迁移到 Linux

macOS + candleLight 存在两个固有不便：① 每次都要 `sudo`；② gs_usb 用户态驱动偶有稳定性问题。

**若要长期稳定运行，强烈建议迁移到 Linux**：candleLight 在 Linux 有原生内核驱动，激活后用 `socketcan` 即可，免 sudo、最稳定（也是 AgileX 官方主推路径）：

```bash
sudo ip link set can0 up type can bitrate 1000000
python3 grasp_task.py
```

本项目的 `safe_arm.py` / `grasp_task.py` 已写好跨平台分支（`_auto_channel()` 自动按系统选择 `socketcan`/`gs_usb`/`agx_cando`），**迁移到 Linux 无需改动代码**。

---

> 备注：本指南配套脚本为社区自研封装，与 AgileX 官方 SDK 解耦。机械臂为高功率设备，任何运动测试请务必空载、低速、随时可急停。
