#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""candleLight 深度诊断：读设备能力 + 用 listen-only 模式测试收帧。

判断"收不到数据"到底是波特率不匹配、还是总线真的没流量。
macOS 需 sudo：
    sudo /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 gs_diag.py
可选指定波特率：gs_diag.py 500000
"""

import os
import sys
import time

import can
from gs_usb.gs_usb import GsUsb
from gs_usb.gs_usb_frame import GsUsbFrame
from gs_usb.constants import (
    GS_CAN_MODE_LISTEN_ONLY,
    GS_CAN_MODE_HW_TIMESTAMP,
)

# 修复 macOS 上 gs_usb start() 的 is_kernel_driver_active 报错
from safe_arm import _patch_gs_usb_start_for_macos

_patch_gs_usb_start_for_macos()

bitrate = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000

devs = GsUsb.scan()
print(f"扫描到 gs_usb 设备: {len(devs)} 个")
if not devs:
    print("❌ 没扫描到 candleLight。")
    os._exit(1)

dev = devs[0]

# ---- 设备信息 ----
try:
    info = dev.device_info
    print(f"固件版本 fw={info.fw_version/10.0}  硬件版本 hw={info.hw_version/10.0}  通道数 icount={info.icount}")
except Exception as e:
    print("读取 device_info 失败:", repr(e))

cap = dev.device_capability
print(f"CAN 时钟 fclk_can = {cap.fclk_can} Hz  ({cap.fclk_can/1e6:.0f} MHz)")
print(f"feature   = 0x{cap.feature:08x}")
print(f"tseg1 {cap.tseg1_min}-{cap.tseg1_max}  tseg2 {cap.tseg2_min}-{cap.tseg2_max}  "
      f"sjw_max {cap.sjw_max}  brp {cap.brp_min}-{cap.brp_max}")

# ---- 计算并设置 timing ----
try:
    bt = can.BitTiming.from_sample_point(
        f_clock=cap.fclk_can, bitrate=bitrate, sample_point=87.5
    )
    print(f"\n目标波特率 {bitrate} → 计算 timing: brp={bt.brp} tseg1={bt.tseg1} "
          f"tseg2={bt.tseg2} sjw={bt.sjw}（实际 {bt.bitrate} bps, 采样点 {bt.sample_point}%）")
    dev.set_timing(
        prop_seg=1,
        phase_seg1=bt.tseg1 - 1,
        phase_seg2=bt.tseg2,
        sjw=bt.sjw,
        brp=bt.brp,
    )
except Exception as e:
    print("计算/设置 timing 失败:", repr(e))
    os._exit(1)

# ---- listen-only 模式监听（纯接收，不发 ACK，不干扰总线）----
print("\n以 LISTEN-ONLY 模式监听 5 秒...")
dev.start(GS_CAN_MODE_LISTEN_ONLY | GS_CAN_MODE_HW_TIMESTAMP)

frame = GsUsbFrame()
n = 0
ids = set()
t0 = time.time()
while time.time() - t0 < 5.0:
    if dev.read(frame=frame, timeout_ms=100):
        n += 1
        ids.add(frame.can_id)
        if n <= 20:
            print(f"  id=0x{frame.can_id:03X} dlc={frame.can_dlc}")

print(f"\nLISTEN-ONLY 共收到 {n} 帧, IDs={sorted(hex(i) for i in ids)}")
if n == 0:
    print(
        "仍然 0 帧。结论：\n"
        "  - 若 fclk_can 是 48M 或 80M 且 timing 正常 → 波特率参数没问题，\n"
        "    那就是【物理层确实没流量】：CAN 线没真正接到机械臂数据线 / 接触不良 / 没共地。\n"
        "  - 可换波特率重试：gs_diag.py 500000（万一机械臂不是 1Mbps）。"
    )
else:
    print("✅ 收到数据！波特率正确，总线有流量。问题之前出在 NORMAL 模式或上层。")

dev.stop()
os._exit(0)
