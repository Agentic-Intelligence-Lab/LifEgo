#!/bin/bash
# 激活 gs_usb USB-CAN 接口（Jetson 上通常为 can1）
set -e
IFACE="${1:-}"
BITRATE="${2:-1000000}"

if [ -z "$IFACE" ]; then
  while read -r name _; do
    if ethtool -i "$name" 2>/dev/null | grep -q "driver: gs_usb"; then
      IFACE="$name"
      break
    fi
  done < <(ip -br link show type can 2>/dev/null || true)
fi

if [ -z "$IFACE" ]; then
  IFACE="can1"
fi

if ! ip link show "$IFACE" &>/dev/null; then
  echo "错误: 接口 $IFACE 不存在，请插入 USB-CAN 模块"
  exit 1
fi

ip link set "$IFACE" down 2>/dev/null || true
ip link set "$IFACE" type can bitrate "$BITRATE"
ip link set "$IFACE" up
echo "OK: $IFACE UP @ ${BITRATE} bps"
