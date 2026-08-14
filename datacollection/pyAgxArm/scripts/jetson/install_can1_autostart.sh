#!/bin/bash
# 一次性安装：开机自动 up can1 + 插 USB 自动 up + 免密手动激活
# 用法（必须在系统终端，非 Cursor 终端）:
#   cd ~/Downloads/pyAgxArm-master/scripts/jetson
#   sudo bash install_can1_autostart.sh
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME="${SUDO_USER:-$USER}"

if [ "$(id -u)" -ne 0 ]; then
  echo "请用 sudo 运行: sudo bash install_can1_autostart.sh"
  exit 1
fi

install -m 755 "$DIR/can1_up.sh" /usr/local/sbin/agx-can-up.sh
install -m 644 "$DIR/99-agx-gs-usb-can.rules" /etc/udev/rules.d/
install -m 644 "$DIR/can1-up.service" /etc/systemd/system/

SUDOERS="/etc/sudoers.d/agx-can1-${USER_NAME}"
cat > "$SUDOERS" <<EOF
# Agilex USB-CAN 免密激活（${USER_NAME}）
${USER_NAME} ALL=(ALL) NOPASSWD: /usr/local/sbin/agx-can-up.sh
${USER_NAME} ALL=(ALL) NOPASSWD: /sbin/ip link set can1 *
EOF
chmod 440 "$SUDOERS"
visudo -cf "$SUDOERS"

systemctl daemon-reload
systemctl enable can1-up.service
udevadm control --reload-rules
udevadm trigger -c add -s net 2>/dev/null || true

echo ""
echo "安装完成。正在激活 can1 ..."
/usr/local/sbin/agx-can-up.sh || true
ip -br link show type can
echo ""
echo "之后可用（无需密码）:"
echo "  sudo -n /usr/local/sbin/agx-can-up.sh"
echo "  或: sudo -n /usr/local/sbin/agx-can-up.sh can1"
