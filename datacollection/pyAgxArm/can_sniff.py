#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""底层 CAN 嗅探：直接打印 gs_usb 总线上的原始帧，绕过 pyAgxArm 解析。

用来判断"收不到机械臂数据"是物理层没流量，还是上层解析问题。
macOS 需要 sudo：
    sudo /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 can_sniff.py
"""

import os
import sys
import time

from safe_arm import _install_gs_usb_adapter

_install_gs_usb_adapter()
import can

DURATION = 5.0
# 可在命令行指定波特率，例如：can_sniff.py 500000
bitrate = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000

print(f"打开 gs_usb 总线（{bitrate} bps）...")
bus = can.interface.Bus(interface="gs_usb", bitrate=bitrate)
print(f"开始嗅探 {DURATION:.0f} 秒，请确认机械臂已上电...\n")

n = 0
counts: dict = {}
t0 = time.time()
while time.time() - t0 < DURATION:
    m = bus.recv(0.2)
    if m is not None:
        n += 1
        counts[m.arbitration_id] = counts.get(m.arbitration_id, 0) + 1
        if n <= 30:
            print(f"  id=0x{m.arbitration_id:03X}  dlc={m.dlc}  data={bytes(m.data).hex()}")

print(f"\n共收到 {n} 帧。各 CAN ID 计数（NORMAL 模式应出现多个 ID）：")
for cid in sorted(counts):
    print(f"  0x{cid:03X}: {counts[cid]} 帧")
ids = set(counts)
if n == 0:
    print(
        "❌ 总线上没有任何帧 → 物理层问题：\n"
        "   1) 机械臂/灵巧手是否真的上电？\n"
        "   2) candleLight 是否插在机械臂的 CAN 口（不是 Rokoko/其它设备）？\n"
        "   3) CANH/CANL 是否接反、终端电阻是否就位？\n"
        "   4) 机械臂 CAN 波特率是否为 1 Mbps？"
    )
else:
    print("✅ 总线有数据！说明物理通路 OK，问题在上层（型号/固件配置）。")

# gs_usb 关闭时会 segfault，强制退出规避（不影响功能）
os._exit(0)
