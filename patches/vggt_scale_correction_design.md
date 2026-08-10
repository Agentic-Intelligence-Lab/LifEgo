# VGGT-Omega 在 LifEgo 中的用途、可行性分析与手部深度校正设计

状态：**分析/设计文档，尚未实现**。记录现状代码怎么用 VGGT，以及"能不能用 VGGT 的尺度信息提升
WiLoR 手部位姿在 robot_base 系下精度"这个问题的分析结论和推荐方案，供后续决定要不要落地、怎么落地。

---

## 1. ymq 现有脚本里 VGGT 的设计意图（现状盘点，全部基于代码逐条核实）

### 1.1 `patches/run_vggt_omega_infer.py` —— VGGT 推理本体

- 定位：docstring 明确写 "Smoke-test VGGT-Ω inference... **no EEF correction yet**"——目前是独立的
  推理/验证脚本，**还没有接入** WiLoR → EEF → IK 这条主流程。
- 输入：一批 RGB 帧（`--images` 指向 HumanEgo 的 `preprocess/all_data`，或 `--video`）。
- 模型：`vggt_omega.models.VGGTOmega`，权重 `thirdparty/vggt-omega/weights/VGGT-Omega/vggt_omega_1b_512.pt`，
  一次前向对整批帧联合推理（多视角），大序列用 `--chunk-size` 分块。
- 输出 `predictions.npz`：
  - `depth` / `depth_conf`：每帧一张深度图 + 置信度
  - `pose_enc` → 解码出 `extrinsic` / `intrinsic`：**VGGT 自己推断的**每帧相机位姿/内参（不是我们标定的
    真实相机参数，是网络内部自洽的一套相对位姿，`encoding_to_camera()` 解码得到）
  - 可选 `camera_and_register_tokens`（`--save-tokens`，体积大，一般不需要）
- 脚本自己在 `predictions_summary.json` 写明的关键限制：

  > `"note": "Chunked multi-view inference for long sequences; each chunk is jointly processed.
  > Absolute scale still uncalibrated."`

  即 VGGT 重建的深度/相机位姿**内部自洽，但整体尺度是任意的**——纯视觉多视角重建的通病，图像本身
  不含度量信息。

- `patches/README.md` 里对它的定位：「VGGT-Ω 本地推理 smoke test（深度/相机姿态；**后续做物体尺寸
  尺度校正**）」——设计意图是拿它重建场景/物体几何，但要变成"米"，需要额外的尺度校准。

### 1.2 `patches/estimate_scale_apriltag.py` —— 用 AprilTag 给 VGGT 定尺度

这是目前唯一把 VGGT 输出接上"米制"的脚本：

- `scale_from_vggt_depth()`：在检测到的 tag 四个角点像素处采样 VGGT 深度图，反投影出这四个角点的
  3D 坐标，量出四条边长的中位数 `L_pred`，与已知真实边长 `tag_size_m` 做比：

  ```python
  s = tag_size_m / L_pred
  ```

- 对多帧、多次检测的 `s_i` 取中位数：`s_vggt = median(scales_vggt)`（`run()` 函数里），做法上和整个
  项目"多次采样取中位数降噪"的风格一致（例如 `average_SE3()`、AprilTag 标定脚本里对多张照片的像素
  角点取平均）。
- `apply_to_eef()`：**已经写好**了一条"用尺度 + 新外参改写 EEF 轨迹"的路径——
  `T_c_scaled[:3, 3] *= s`（相机系平移向量整体缩放），再 `T_base = T_cam_in_base @ T_c_scaled`。
  由 `--apply-scale` / `--eef-scale` 触发，默认关闭。
- **这条路径目前改的是 WiLoR 导出的 EEF 轨迹本身**（`T_ee_in_cam` 整体乘 `s`），但正如下面第 2 节分析
  的，这里如果不做修改就直接把 `s_vggt` 套用到 WiLoR 位置上，语义是有问题的——`apply_to_eef()` 这个
  开关目前更适合用在"WiLoR 位置真的存在跟 VGGT 同源的尺度模糊性"的场景，而不是我们这个用例。

### 1.3 与相机标定 / assets.py 的关系

- `patches/assets.py`（`DEFAULT_ASSETS`）里已经有精确标定的固定桌面相机 `T_cam_in_base`（两 tag
  联合最小二乘，重投影 RMSE 0.477px）和内参（`fx=606.381, fy=605.975, cx=331.115, cy=238.649`）。
  这是目前所有需要相机内外参的脚本的唯一真源（上一轮已经把 `calibrate_realsense_extrinsic_from_apriltags.py`
  / `process_examples_wilor.py` / `estimate_scale_apriltag.py` / `export_robot_eef_from_wilor.py` /
  `build_nero_mujoco_scene.py` 的默认取值都改成读它）。
- 这意味着：**`T_cam_in_base` 这个外参不需要、也不应该再通过 `estimate_scale_apriltag.py` 的
  `--tag-in-base-json` 路径重新求**——那条路径是单帧 PnP 反解，精度不如现成的联合标定结果。VGGT 相关
  的这套东西，唯一还没有定论的是"尺度"，不是外参。

---

## 2. 可行性分析：能不能用 VGGT 的尺度信息提升 WiLoR 手部位姿精度？

**结论：思路本身可行，但不能直接复用 `s_vggt` 这个数——它和 WiLoR 的误差不是同一个未知量。需要单独
构造一个针对 WiLoR 的校正标量。**

### 2.1 WiLoR 深度怎么算的（`preprocess/WiLoRHands.py:_recover_absolute_3d`, L365-416）

```python
physical_dist = ‖middle_mcp_3d - wrist_3d‖   # WiLoR 网络自己预测的相对3D手型（MANO先验）算出的手长
pixel_dist    = ‖middle_mcp_2d - wrist_2d‖   # 同一段在图像里的像素长度
focal         = (fx + fy) / 2
z_wrist       = focal * physical_dist / pixel_dist          # 深度
x_wrist       = (wrist_2d.x - cx) * z_wrist / fx             # 反投影
y_wrist       = (wrist_2d.y - cy) * z_wrist / fy
```

这个式子里**唯一的相机相关未知量是 `focal`**（我们上一轮已经用 assets.py 的真实值修好了）。剩下的
`physical_dist` 是 WiLoR 网络自己对"这只手多大"的预测——一个从训练数据里学出来的、跟真实这个人的
手可能存在系统性偏差的先验，**跟相机、跟 VGGT 完全独立**。

一个直接可验证的推论（代数化简）：把 `z_wrist` 代入 `x_wrist` 公式，`focal` 会精确抵消：

```
x_wrist = (u-cx) · [focal·physical_dist/pixel_dist] / focal = (u-cx)·physical_dist/pixel_dist
```

即 **X、Y 完全不依赖相机焦距是否正确，只有 Z（深度）依赖**。这也解释了上一轮那个 K bug 为什么是
"纯深度方向的系统误差"，而不是位置随机乱掉。

### 2.2 VGGT 的 `s_vggt` 纠正的是什么

`s_vggt = tag_size_m / L_pred_from_vggt_depth`——纠正的是 **VGGT 自己的多视角深度重建**相对真实
世界的整体缩放。这个缩放来自 VGGT 网络内部的多视角三角化过程，跟 WiLoR 的 `physical_dist` 手型先验
没有任何数学联系。两者是两个独立模型、两套独立的"未知尺度"，把 `s_vggt` 直接乘到 WiLoR 的输出上，
物理意义上是错的（除非巧合数值接近）。

### 2.3 但 VGGT 确实能帮上忙——纠正的对象要换

K 修好之后，WiLoR 深度估计剩下的误差来源只有一个：`physical_dist` 这个手型假设是否匹配这个具体人的
真实手。这个偏差有两个特点：

1. **和相机、和 VGGT 的尺度模糊性无关**，是 WiLoR 自己的模型偏差。
2. **对同一个人、同一只手，在整个 session 里大致是恒定的一个比例**（不会逐帧随机跳变）。

VGGT 重建的场景深度（一旦用 `s_vggt` 换算成米），是一个**跟 WiLoR 完全独立、基于多视角几何的手部
位置估计**——可以拿来标定这个"个人手型偏差"，但要单独算一个新的标量，不是 `s_vggt` 本身：

```
k_hand = median_over_frames( z_vggt_metric(手腕像素处) / z_wilor(该帧WiLoR估计的深度) )
```

`z_vggt_metric = VGGT depth[frame, wrist_pixel] × s_vggt`。

因为 `X = (u-cx)·z/fx, Y = (v-cy)·z/fy, Z = z` 三者都正比于同一个 `z_wrist`，给 `z_wrist` 乘
`k_hand` 等价于把整条相机系位置向量沿视线方向整体缩放 `k_hand` 倍——跟 `estimate_scale_apriltag.py::apply_to_eef`
里现成的 `T_c_scaled[:3, 3] *= s` 是同一种写法，只是缩放因子换成 `k_hand` 而不是 `s_vggt`，且只作用于
手部相关的位置向量。

### 2.4 可行性上的风险点（要老实说清楚）

1. **VGGT 在手部区域的深度质量未知，可能明显弱于静态背景**：VGGT 的多视角一致性假设是"同一个 3D
   点在不同视角里能被观察到"，手是运动的前景物体，在各帧间位置/姿态都在变，跟静止桌面/背景不是一回事。
   这个假设是否成立需要实测验证（跑一次 VGGT，人工看几帧 `depth_frame_*.png` 里手部区域是否合理），
   不能想当然认为 VGGT 手部深度和背景深度一样可靠。
2. **`depth_conf` 目前没被用上**：`estimate_scale_apriltag.py::scale_from_vggt_depth()` 现在直接
   采样深度值，不看置信度；给手部深度取样时应该过滤低置信度像素/帧，否则中位数会被污染。
3. **需要覆盖两类帧**：既要有清晰可见 tag 的帧（求 `s_vggt`），又要有手在场景里、WiLoR 也给出有效
   检测的帧（求 `k_hand`）——理想情况下同一个 session 里两者都要有，且都要用同一套（已修好的）
   相机 K，帧数不够会让两个中位数都不稳。
4. **`k_hand` 是"整个 session 一个常数"的假设**：如果同一个 session 里换了人手、或者中途重新调整了
   镜头，这个假设会失效，需要按需要重新分段计算。
5. 这一整套目前代码里**完全没有实现**（`run_vggt_omega_infer.py` 不输出手腕深度对比，
   `estimate_scale_apriltag.py` 不读 WiLoR 的 `wilor_hands.json`），是全新的代码，不是改几行现成逻辑。

---

## 3. 推荐方案（设计，未落地）

### 3.1 总体流程

```
1. run_vggt_omega_infer.py            → predictions.npz（VGGT 相对尺度深度）
2. estimate_scale_apriltag.py         → s_vggt（tag 定的 VGGT→米 换算因子），来自看得到 tag 的那些帧
3. 【新增，未实现】k_hand 计算：
   对每一帧同时有 WiLoR wrist 2D 检测 + VGGT depth 的情况：
     z_vggt_metric = VGGT.depth[frame, wrist_px] × s_vggt      （用 depth_conf 过滤低置信度）
     ratio_i = z_vggt_metric / z_wilor_i
   k_hand = median(ratio_i)   （多帧聚合降噪，和 s_vggt 的聚合方式一致）
4. 【新增，未实现】应用：
   T_hand_in_cam.translation *= k_hand   （整条相机系位置向量沿视线整体缩放，复用 apply_to_eef 的写法）
   之后照常 T_ee_in_base = T_cam_in_base @ T_ee_in_cam   （T_cam_in_base 来自 assets.py，不变）
```

### 3.2 和现有代码的关系

- 第 1、2 步已有现成脚本，不用改。
- 第 3、4 步需要新代码，读取：
  - `outputs/<session>/preprocess/all_data/*/wilor_hands.json`（WiLoR 的 2D 腕部关键点 + 已估计深度，
    需要确认 `wilor_hands.json` 里是否直接存了 `wrist_2d` 和 `z_wrist`，或者需要重新从 `kpts_2d`/
    `pred_keypoints_3d` 反推——这点在真正写代码前需要读一下 `wilor_hands.json` 的实际字段核实）
  - `outputs/<session>/vggt_omega/predictions.npz`（`depth`, `depth_conf`, 以及帧序号对应关系——
    VGGT 是按 `--frame-stride` 抽样跑的，帧号和 WiLoR 全量帧号要对齐，这点 `estimate_scale_apriltag.py`
    里已经有类似的帧号映射逻辑可以参考，`run()` 函数里 `d_i = depth_all[frame_idx]` 那段）
  - `outputs/<session>/apriltag_calib/apriltag_calibration.json` 里的 `s_vggt`
- 落地位置待定（上一轮讨论过两个方向，用户选择先不定）：
  - 方案 A：扩展 `estimate_scale_apriltag.py`，在现有 `--vggt-npz`/`--eef-in` 基础上加一个
    `k_hand` 计算 + 应用模式，复用它已有的 K 加载、npz 读取、中位数聚合逻辑。
  - 方案 B：单独一个 `patches/correct_wilor_hand_depth.py`，职责更单一，但要重复一部分 session/npz/K
    读取逻辑。

### 3.3 验证方式（建议，无论最终选哪个方案都需要）

- 拿到 `k_hand` 后，看它离 1.0 有多远、多帧的 `ratio_i` 方差多大——如果方差很大（VGGT 手部深度本身
  不稳），这个校正可能弊大于利，需要先确认第 2.4 节的风险点 1 是否成立。
- 校正前后，用 `build_nero_mujoco_scene.py` 生成的场景里青色 HumanEgo EEF 轨迹和真机橙色轨迹做视觉
  对比（`compare_realbot_humanego_eef.py` 的 `|tcp-HE|` 误差数值），看校正后是不是真的更接近真机轨迹，
  而不是纸面上"看起来合理"就直接采用。

---

## 4. 待决定事项

1. 落地方式：扩展 `estimate_scale_apriltag.py` vs. 新脚本（上面 3.2 两个方案）。
2. `wilor_hands.json` 里手腕的 2D 像素坐标 / 已用深度具体字段名——写代码前需要先读一下真实文件核实，
   目前只是从 `WiLoRHands.py` 的算法反推，没有对照过实际 JSON。
3. 用哪个 / 哪些 session 跑第一次验证（需要该 session 同时有清晰 tag 帧和有效手部检测帧）。
4. `depth_conf` 的过滤阈值怎么定（没有先验，需要先看一次实际分布再定，别拍脑袋写死）。
