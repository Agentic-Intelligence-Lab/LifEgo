#!/usr/bin/env python3
"""Read-only SocketCAN probe for Nero / pyAgxArm debugging.

This script does not enable motors and does not send motion commands.
It only opens a SocketCAN interface, prints link status, listens for
raw CAN frames, and highlights CAN IDs used by Nero feedback.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter


NERO_FOLLOWER_JOINT_IDS = {0x2A5, 0x2A6, 0x2A7, 0x2A9}
NERO_LEADER_V112_IDS = {0x155, 0x156, 0x157, 0x170}
NERO_LEADER_LEGACY_IDS = set(range(0x501, 0x508))
NERO_STATUS_IDS = {0x2A1}
NERO_HIGH_SPEED_IDS = set(range(0x251, 0x258))
NERO_LOW_SPEED_IDS = set(range(0x261, 0x268))
NERO_FIRMWARE_IDS = {0x4AF}


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except FileNotFoundError:
        return f"{cmd[0]}: command not found"
    except subprocess.CalledProcessError as e:
        return e.output.strip() or f"{' '.join(cmd)} failed with exit {e.returncode}"


def print_link_info(channel: str) -> None:
    print("== CAN interfaces ==")
    print(run_text(["ip", "-br", "link", "show", "type", "can"]) or "(none)")
    print()

    print(f"== {channel} link detail ==")
    print(run_text(["ip", "-details", "-statistics", "link", "show", channel]))
    print()

    print(f"== {channel} driver ==")
    print(run_text(["ethtool", "-i", channel]))
    print()


def sniff(channel: str, duration: float, max_print: int) -> Counter[int]:
    try:
        import can
    except ImportError:
        print("python-can is not installed. Try: timeout 5s candump " + channel)
        sys.exit(2)

    counts: Counter[int] = Counter()
    print(f"== Listening on {channel} for {duration:.1f}s ==")
    bus = can.interface.Bus(interface="socketcan", channel=channel)
    start = time.monotonic()
    printed = 0
    try:
        while time.monotonic() - start < duration:
            msg = bus.recv(0.2)
            if msg is None:
                continue
            counts[msg.arbitration_id] += 1
            if printed < max_print:
                printed += 1
                data = bytes(msg.data).hex(" ")
                print(f"id=0x{msg.arbitration_id:03X} dlc={msg.dlc} data={data}")
    finally:
        bus.shutdown()
    print()
    return counts


def describe_ids(ids: set[int]) -> None:
    def yes(name: str, expected: set[int]) -> None:
        found = ids & expected
        missing = expected - ids
        print(
            f"{name}: found {fmt_ids(found) if found else '-'}; "
            f"missing {fmt_ids(missing) if missing else '-'}"
        )

    yes("follower joint feedback", NERO_FOLLOWER_JOINT_IDS)
    yes("leader feedback v112", NERO_LEADER_V112_IDS)
    yes("leader feedback default/v111", NERO_LEADER_LEGACY_IDS)
    yes("arm status", NERO_STATUS_IDS)
    yes("motor high-speed feedback", NERO_HIGH_SPEED_IDS)
    yes("driver low-speed feedback", NERO_LOW_SPEED_IDS)
    yes("firmware response", NERO_FIRMWARE_IDS)


def fmt_ids(ids: set[int]) -> str:
    return " ".join(f"0x{i:03X}" for i in sorted(ids))


def print_summary(counts: Counter[int]) -> int:
    total = sum(counts.values())
    print(f"== Summary: {total} frame(s) ==")
    if counts:
        for can_id, count in sorted(counts.items()):
            print(f"0x{can_id:03X}: {count}")
        print()
        describe_ids(set(counts))
        print()

    if total == 0:
        print("No frames were received.")
        print("Most likely: wrong CAN channel, arm not powered, CANH/CANL wiring,")
        print("missing common ground/termination, or bitrate is not 1 Mbps.")
        return 1

    ids = set(counts)
    if ids & (NERO_FOLLOWER_JOINT_IDS | NERO_LEADER_V112_IDS | NERO_LEADER_LEGACY_IDS):
        print("Nero joint-related frames are present on this bus.")
        print("If read_leader_joints.py still prints No data, check firmware mode (-f).")
        return 0

    print("Frames are present, but Nero joint feedback IDs were not seen.")
    print("This can be a wrong bus, wrong firmware/CAN-push mode, or another device.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only SocketCAN/Nero probe")
    parser.add_argument("-c", "--channel", default="can1")
    parser.add_argument("-t", "--duration", type=float, default=5.0)
    parser.add_argument("--max-print", type=int, default=30)
    args = parser.parse_args()

    print_link_info(args.channel)
    counts = sniff(args.channel, args.duration, args.max_print)
    return print_summary(counts)


if __name__ == "__main__":
    raise SystemExit(main())
