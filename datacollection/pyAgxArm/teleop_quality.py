"""Shared quality helpers for Nero teleop collection (GUI + recorder)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Optional, Sequence


SequenceLike = Sequence[Any]

DEFAULT_TASK = "pick up the red block and place it on the yellow plate"
DEFAULT_SESSION_PREFIX = "nero_pick_place"

# Reject these as episode-level language instructions.
PLACEHOLDER_TASKS = frozenset(
    {
        "",
        "nero_teleop",
        "teleop",
        "test",
        "todo",
        "task",
        "none",
        "n/a",
        "na",
    }
)

# Per-frame TCP jump warn thresholds (adjacent samples @ ~20 Hz).
TCP_JUMP_WARN_MM = 50.0
TCP_JUMP_WARN_DEG = 10.0

OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"
OUTCOME_ABORT = "abort"
OUTCOME_UNREVIEWED = "unreviewed"

# Converter keeps success + legacy (missing outcome); drops failure/abort.
TRAINABLE_OUTCOMES = frozenset({OUTCOME_SUCCESS, OUTCOME_UNREVIEWED, None})


def sanitize_session_name(name: str, *, fallback: str = DEFAULT_SESSION_PREFIX) -> str:
    text = (name or "").strip()
    if not text:
        return fallback
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in text)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or fallback


def validate_task(task: str) -> str:
    text = " ".join((task or "").strip().split())
    if not text:
        raise ValueError("任务描述不能为空：请填写英文指令，例如 pick up the red block…")
    if text.lower() in PLACEHOLDER_TASKS:
        raise ValueError(
            f"任务描述不能使用占位符 “{text}”。请填写真实语言指令。"
        )
    if len(text) < 8:
        raise ValueError("任务描述过短：请写完整英文指令（至少 8 个字符）。")
    return text


def _rpy_zyx_to_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
            for j in range(3)
        ]
        for i in range(3)
    ]


def _transpose3(r: list[list[float]]) -> list[list[float]]:
    return [[r[0][i], r[1][i], r[2][i]] for i in range(3)]


def _rotation_angle_deg(r_rel: list[list[float]]) -> float:
    trace = r_rel[0][0] + r_rel[1][1] + r_rel[2][2]
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(cos_theta))


def tcp_frame_delta(
    prev_tcp: Optional[SequenceLike],
    curr_tcp: Optional[SequenceLike],
) -> tuple[Optional[float], Optional[float]]:
    """Return (translation_mm, rotation_deg) between adjacent TCP poses, or (None, None)."""
    if prev_tcp is None or curr_tcp is None:
        return None, None
    if len(prev_tcp) < 6 or len(curr_tcp) < 6:
        return None, None
    dx = float(curr_tcp[0]) - float(prev_tcp[0])
    dy = float(curr_tcp[1]) - float(prev_tcp[1])
    dz = float(curr_tcp[2]) - float(prev_tcp[2])
    translation_mm = math.sqrt(dx * dx + dy * dy + dz * dz) * 1000.0
    r_prev = _rpy_zyx_to_matrix(float(prev_tcp[3]), float(prev_tcp[4]), float(prev_tcp[5]))
    r_curr = _rpy_zyx_to_matrix(float(curr_tcp[3]), float(curr_tcp[4]), float(curr_tcp[5]))
    r_rel = _matmul3(_transpose3(r_prev), r_curr)
    return translation_mm, _rotation_angle_deg(r_rel)


def build_tcp_quality(
    prev_tcp: Optional[SequenceLike],
    curr_tcp: Optional[SequenceLike],
    *,
    warn_mm: float = TCP_JUMP_WARN_MM,
    warn_deg: float = TCP_JUMP_WARN_DEG,
) -> dict[str, Any]:
    jump_mm, jump_deg = tcp_frame_delta(prev_tcp, curr_tcp)
    warn = False
    if jump_mm is not None and jump_mm > warn_mm:
        warn = True
    if jump_deg is not None and jump_deg > warn_deg:
        warn = True
    return {
        "tcp_jump_mm": jump_mm,
        "tcp_jump_deg": jump_deg,
        "tcp_jump_warn": warn,
        "tcp_jump_warn_mm": warn_mm,
        "tcp_jump_warn_deg": warn_deg,
    }


def patch_jsonl_metadata(path: Path, updates: dict[str, Any]) -> None:
    """Atomically merge ``updates`` into the first metadata JSONL row."""
    path = path.expanduser().resolve()
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"{path} is empty")
    lines = raw.splitlines(keepends=True)
    first = lines[0]
    newline = "\n" if first.endswith("\n") else ""
    meta = json.loads(first)
    if meta.get("kind") != "metadata":
        raise ValueError(f"{path} first row is not metadata")
    for key, value in updates.items():
        if (
            key == "collection"
            and isinstance(value, dict)
            and isinstance(meta.get("collection"), dict)
        ):
            merged = dict(meta["collection"])
            merged.update(value)
            meta["collection"] = merged
        else:
            meta[key] = value
    lines[0] = json.dumps(meta, ensure_ascii=False) + newline
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    tmp.replace(path)


def is_trainable_outcome(outcome: Optional[str]) -> bool:
    """Legacy files without outcome are kept; failure/abort are dropped."""
    return outcome in TRAINABLE_OUTCOMES
