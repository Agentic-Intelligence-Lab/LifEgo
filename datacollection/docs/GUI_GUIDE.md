# Nero 数据采集 GUI 使用指引

路径：`E:\nero-data-collect\pyAgxArm-master\teleop_recorder_gui.py`

---

## 分辨率建议

当前默认已改成 **1280×720**（配合 `patches/camera_insrinsics.json` 里 1280x720 下的实测内参做实验）；
640×480 对 OpenPI π0.5（训练时常缩到 ~224）也够用，带宽和 Windows 稳定性更好，仍可按需切回。

| 分辨率 | 建议 |
|--------|------|
| 640×480 | 日常遥操作模仿学习、带宽/稳定性优先时用 |
| 1280×720（默认） | 当前实验分辨率，画面细节更好 |
| 更高（1080p+） | 通常不必；更易卡、文件更大 |

改分辨率：GUI 里改 `DEFAULT_CAMERA_WIDTH/HEIGHT`，或命令行 `--camera-width/--camera-height`。

## 相机安装方向

现场 Dabai **装反**时默认做 **顺时针 180°**（`--camera-rotate 180`）。预览与 MP4 都会转到正方向。若装对可设 `0`；若方向反了改成 `90` / `270`。

---

## 1. 启动前检查

1. USB **candleLight CAN** 已插上（Windows 用 `gs_usb`，通道一般为 **0**）。
2. Dabai 相机已接 USB；本机预览默认用 OpenCV **index=1**（0 常常是笔记本自带摄像头）。
3. 不要开「末端同步」（那是强脑 Revo2；当前用 Nero 自带 AgxGripper）。

可选自检：

```bash
cd E:\nero-data-collect\pyAgxArm-master
python win_can_selftest.py
```

启动 GUI：

```bash
python teleop_recorder_gui.py
```

---

## 2. 界面怎么看

| 区域 | 作用 |
|------|------|
| 顶部参数 | CAN 通道、采样频率、任务名、保存目录 |
| 当前采集段 | 已对齐/已写入的样本数与时长 |
| 关节实时值 | 主臂目标 / 从臂实测 / 跟随误差（度） |
| Dabai RGB 预览 | 空闲时由预览进程刷新；**开始采集后由录制进程接管** |
| 数据质量 | 四路红绿灯：机械臂、时间对齐、夹爪/TCP、RGB |
| 采集控制 | 开始 / 停止并保存；空格键可切换开始/停止 |

空闲时应先看到 Dabai 实时画面；若预览失败，先换相机 USB 口或确认 index。

---

## 3. 标准录制流程

1. 确认预览有画面，机械臂上电、示教正常。
2. 填好任务名（会进文件名前缀），确认保存目录。
3. 采样频率建议 **20 Hz**；时长填 **0** = 手动停。
4. 点 **开始采集**（或空格）：
   - 会先关掉空闲预览（释放 DirectShow），再开录制进程；
   - 顶部应变为「正在采集」；
   - 样本数应开始增加，关节/TCP 开始跳动。
5. 做完本段动作后点 **停止并保存**（或空格）。
6. 输出在保存目录，例如：
   - `nero_teleop_YYYYMMDD_….jsonl`（状态 + 对齐元数据）
   - `nero_teleop_YYYYMMDD_….rgb.mp4`（RGB）
7. 需要下一段：再点开始即可（新文件，不会覆盖上一段）。

「删除本段」会删掉刚录的这一对 jsonl/mp4，慎用。「打开目录」直接打开保存文件夹。

---

## 4. 数据质量灯含义

- **机械臂**：主/从关节有数据；「源偏差」= 各 CAN 消息时间戳互相差多久（ms）。
- **时间对齐（本帧有效 / 未对齐）**：这一拍里，CAN、（可选）末端、相机是否都够「新鲜」：
  - 未对齐 **不等于没在录**（GUI 已 `--keep-unaligned`，仍会写入）；
  - 常见原因：相机帧过旧（尤其本机没有 ffmpeg、走 OpenCV `mp4v` 写盘时曾会卡编码）；
  - 灯旁会尽量写出原因，例如「未对齐 · 相机过旧 xxxms」。
- **夹爪 / TCP**：grasp 与 TCP 位姿是否在刷新。
- **RGB 相机**：最新帧编号与相对采样时刻的年龄（ms）；年龄长期几千 ms 说明相机线程卡住。

训练导出阶段再筛字段即可；原始录制尽量录全。

---

## 5. 常见现象

| 现象 | 怎么处理 |
|------|----------|
| 一直「等待 / 正在连接」 | 看底部日志；确认 CAN=0、线已接；重启 GUI |
| 空闲预览好、一点开始预览停 | 正常交接瞬间会黑一下；若持续不动，看日志是否有 `CAMERA_FRAME` / OpenCV 打开失败 |
| 「未对齐」偶发变红 | 看是否写「相机过旧」；优先装 ffmpeg 后再录；当前版本已把采集与编码拆线程 |
| 示教口无 0x159 | 夹爪仍可从反馈录 grasp；不一定挡采集 |
| 预览指到错误摄像头 | 改默认 `DEFAULT_CAMERA_INDEX`，或用 `camera_idle_preview.py` 试 index |
| 录了一段后画面卡住、对齐变红 | Windows OpenCV/`mp4v` 常见；当前版本会自动重开相机。日志若出现 `camera stalled` / `reopened` 属恢复中。建议安装 **ffmpeg** 后再录，更稳 |

---

## 6. 录完导出（训练用）

```bash
cd E:\nero-data-collect\pyAgxArm-master
python export_jsonl_to_lerobot.py --input-dir recordings --output-dir exports\lerobot_tcp --state-mode tcp+gripper --action-mode tcp+gripper
```

更完整的工程说明见 `docs/HANDOFF.md`。
