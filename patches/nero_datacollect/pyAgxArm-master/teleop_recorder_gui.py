#!/usr/bin/env python3
"""Desktop GUI for Nero + AgxGripper teleoperation data collection (Windows/Linux)."""

from __future__ import annotations

import json
import math
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from platform import system as platform_system
from tkinter import filedialog, messagebox
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Any, Optional

from teleop_quality import (
    DEFAULT_SESSION_PREFIX,
    DEFAULT_TASK,
    TCP_JUMP_WARN_DEG,
    TCP_JUMP_WARN_MM,
    sanitize_session_name,
    validate_task,
)


HERE = Path(__file__).resolve().parent
RECORDER = HERE / "record_teleop.py"
IDLE_CAMERA_PREVIEW = HERE / "camera_idle_preview.py"
# macOS dev machine: the unofficial pyrealsense2-macosx wheel segfaults inside
# libusb on pipeline.start() (EXC_BAD_ACCESS in librealsense2.dylib), so this
# machine reads the D435i's color sensor as a plain UVC camera via OpenCV
# instead. On the real (Windows) deployment target, official pyrealsense2
# wheels are stable -- switch back to "realsense" there.
DEFAULT_CAMERA_BACKEND = "opencv"  # "opencv" (D435i as plain UVC, or Dabai) or "realsense"
DEFAULT_CAMERA_INDEX = 0  # D435i color sensor on this machine (verified via cv2 enumeration)
DEFAULT_CAMERA_SERIAL = "auto"  # only used when DEFAULT_CAMERA_BACKEND == "realsense"
DEFAULT_CAMERA_ROTATE = 0
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
CAMERA_LABEL = "RealSense"
IS_WINDOWS = platform_system() == "Windows"


def _default_can_channel() -> str:
    if IS_WINDOWS:
        return "0"
    if platform_system() == "Darwin":
        return "candlelight"
    return "can1"


def _default_can_interface() -> str:
    if IS_WINDOWS or platform_system() == "Darwin":
        return "gs_usb"
    return "socketcan"


def _subprocess_group_kwargs() -> dict[str, Any]:
    """Create an interruptible child process group on Windows and Unix."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _interrupt_process(process: subprocess.Popen[str]) -> None:
    """Ask child to stop gracefully (maps to SIGINT/SIGBREAK)."""
    if process.poll() is not None:
        return
    if IS_WINDOWS:
        try:
            # Requires CREATE_NEW_PROCESS_GROUP at spawn time.
            process.send_signal(signal.CTRL_BREAK_EVENT)
            return
        except (OSError, ValueError, AttributeError):
            pass
        process.terminate()
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except (ProcessLookupError, PermissionError, AttributeError):
        try:
            process.send_signal(signal.SIGINT)
        except Exception:
            process.terminate()


def _force_kill_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        process.kill()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def _open_path_in_file_manager(path: Path) -> None:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if IS_WINDOWS:
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif platform_system() == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _preferred_tk_scaling(root: tk.Tk) -> float:
    """Pick Tk scaling for the current logical screen (DPI already applied by OS).

    Old hardcoded 1.35 was tuned for 4K desks; on 2.5K laptops it overflows.
    """
    screen_h = root.winfo_screenheight()
    if screen_h >= 1400:
        return 1.25
    if screen_h >= 1100:
        return 1.1
    return 1.0


class TeleopRecorderGUI:
    POLL_MS = 100
    BG = "#f3efe7"
    CARD = "#fffdf9"
    INK = "#18344f"
    MUTED = "#6a7b8f"
    BORDER = "#d8d0c2"
    GREEN = "#2d8f68"
    RED = "#b54737"
    AMBER = "#b7791f"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Nero 数据采集工作台")
        self.root.configure(bg=self.BG)

        self.process: Optional[subprocess.Popen[str]] = None
        self.stdout_queue: queue.Queue[str] = queue.Queue()
        self.output_path: Optional[Path] = None
        self.video_path: Optional[Path] = None
        self.camera_preview_path: Optional[Path] = Path(tempfile.gettempdir()) / (
            f"nero_teleop_gui_{os.getpid()}_camera.png"
        )
        self.camera_preview_photo: Optional[tk.PhotoImage] = None
        self.camera_preview_mtime_ns = 0
        self.camera_preview_frame_hint = -1
        self.idle_camera_process: Optional[subprocess.Popen[str]] = None
        self.idle_camera_queue: queue.Queue[str] = queue.Queue()
        self.idle_camera_ready = False
        self.idle_camera_error: Optional[str] = None
        self.idle_camera_error_reported: Optional[str] = None
        self.tail_file: Optional[Any] = None
        self.tail_buffer = ""
        self.samples = 0
        self.latest_sample: Optional[dict[str, Any]] = None
        self.stop_requested = False
        self.close_after_stop = False
        self.delete_after_stop = False
        self.sync_control_path: Optional[Path] = None
        self.saved_segments = 0
        self.success_segments = 0
        self.trace_history: list[tuple[int, list[float]]] = []
        self.last_trace_seq: Optional[int] = None
        self.jump_warn_count = 0
        self.max_jump_mm = 0.0
        self.max_jump_deg = 0.0
        self._jump_log_seq: Optional[int] = None

        self.channel_var = tk.StringVar(value=_default_can_channel())
        self.hz_var = tk.StringVar(value="20")
        self.duration_var = tk.StringVar(value="0")
        self.directory_var = tk.StringVar(value=str(HERE / "recordings"))
        self.session_var = tk.StringVar(value=DEFAULT_SESSION_PREFIX)
        self.task_var = tk.StringVar(value=DEFAULT_TASK)
        self.hand_sync_var = tk.BooleanVar(value=False)
        self.state_var = tk.StringVar(value="未开始")
        self.file_var = tk.StringVar(value="—")
        self.count_var = tk.StringVar(value="0")
        self.elapsed_var = tk.StringVar(value="0.0 s")
        self.rate_var = tk.StringVar(value="—")
        self.segment_var = tk.StringVar(value="第 1 段")
        self.can_quality_var = tk.StringVar(value="● 等待 CAN")
        self.align_quality_var = tk.StringVar(value="● 等待数据")
        self.hand_quality_var = tk.StringVar(value="● 等待夹爪")
        self.camera_quality_var = tk.StringVar(value=f"● 等待 {CAMERA_LABEL}")
        self.tcp_jump_var = tk.StringVar(value="● 等待 TCP")
        self.target_hand_var = tk.StringVar(value="—")
        self.hand_sync_status_var = tk.StringVar(value="已关闭")
        self.pose_tcp_var = tk.StringVar(value="—")
        self.pose_flange_var = tk.StringVar(value="—")
        self.footer_var = tk.StringVar(
            value="空格：开始/停止并保存    有问题点「删除本段」    Backspace：删除本段    Esc：退出"
        )

        self.leader_vars = [tk.StringVar(value="—") for _ in range(7)]
        self.follower_vars = [tk.StringVar(value="—") for _ in range(7)]
        self.error_vars = [tk.StringVar(value="—") for _ in range(7)]
        self.teach_value_var = tk.StringVar(value="—")
        self.teach_raw_var = tk.StringVar(value="—")
        self.teach_force_var = tk.StringVar(value="—")
        self.feedback_value_var = tk.StringVar(value="—")
        self.feedback_force_var = tk.StringVar(value="—")
        self.gripper_status_var = tk.StringVar(value="—")

        # Logical pixels after Windows DPI; 2.5K laptops are often ~900–1100 tall.
        screen_h = self.root.winfo_screenheight()
        screen_w = self.root.winfo_screenwidth()
        self.compact_ui = screen_h < 1100 or screen_w < 1400
        self.ui_pad = 12 if self.compact_ui else 22
        self.ui_gap = 8 if self.compact_ui else 14
        self.camera_min_h = 180 if self.compact_ui else 280
        self.title_font_size = 20 if self.compact_ui else 28

        self._build_style()
        self._build_ui()
        self._fit_window_to_screen()
        self.root.bind("<space>", self._on_space)
        self.root.bind("<BackSpace>", self._on_backspace)
        self.root.bind("<Escape>", lambda _event: self.on_close())
        self.root.bind("<Configure>", self._on_root_configure, add="+")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._start_idle_camera_preview)
        self.root.after(self.POLL_MS, self.poll)

    def _fit_window_to_screen(self) -> None:
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        # Leave room for taskbar / window chrome; avoid forcing a 4K-era minsize.
        margin_x = 24 if self.compact_ui else 48
        margin_y = 48 if self.compact_ui else 72
        max_w = max(screen_w - margin_x, 640)
        max_h = max(screen_h - margin_y, 480)
        req_w = self.root.winfo_reqwidth() + 16
        req_h = self.root.winfo_reqheight() + 16
        prefer_w = 1100 if self.compact_ui else 1180
        prefer_h = 700 if self.compact_ui else 760
        width = min(max(req_w, prefer_w), max_w)
        height = min(max(req_h, prefer_h), max_h)
        x = max((screen_w - width) // 2, 0)
        y = max((screen_h - height) // 2, 0)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(min(900, width), min(560, height))
        if self.compact_ui:
            try:
                self.root.state("zoomed")
            except tk.TclError:
                self.root.attributes("-zoomed", True)

    def _on_root_configure(self, event: tk.Event) -> None:
        if event.widget is not self.root:
            return
        wrap = max(int(event.width * 0.28), 220)
        if hasattr(self, "file_label"):
            self.file_label.configure(wraplength=wrap)
        if hasattr(self, "footer_label"):
            self.footer_label.configure(wraplength=wrap)

    def _build_style(self) -> None:
        available = set(tkfont.families(self.root))
        self.display_font = next(
            (name for name in ("Noto Sans CJK SC", "Noto Sans", "DejaVu Sans") if name in available),
            "TkDefaultFont",
        )
        self.mono_font = next(
            (name for name in ("JetBrains Mono", "Noto Sans Mono", "DejaVu Sans Mono") if name in available),
            "TkFixedFont",
        )
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(
            "Nero.Horizontal.TProgressbar",
            troughcolor="#e4ddd1",
            background=self.GREEN,
            bordercolor="#e4ddd1",
            lightcolor=self.GREEN,
            darkcolor=self.GREEN,
            thickness=12 if self.compact_ui else 18,
        )
        style.configure(
            "Nero.TEntry",
            fieldbackground="#f8f6f1",
            bordercolor=self.BORDER,
            padding=5 if self.compact_ui else 7,
        )

    def _build_ui(self) -> None:
        pad = self.ui_pad
        gap = self.ui_gap
        outer = tk.Frame(self.root, bg=self.BG, padx=pad, pady=pad)
        outer.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(outer, bg=self.BG)
        header.pack(fill=tk.X, pady=(0, gap))
        title_box = tk.Frame(header, bg=self.BG)
        title_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            title_box,
            text="Nero 数据采集工作台",
            bg=self.BG,
            fg=self.INK,
            font=(self.display_font, self.title_font_size, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            title_box,
            text="正式采集 · 英文任务必填 · 空格启停连续采集 · TCP 跳变告警",
            bg=self.BG,
            fg=self.MUTED,
            font=(self.display_font, 10 if self.compact_ui else 11),
        ).pack(anchor=tk.W, pady=(3, 0))
        self.state_label = tk.Label(
            header,
            textvariable=self.state_var,
            bg="#e9eef2",
            fg=self.INK,
            font=(self.display_font, 12 if self.compact_ui else 13, "bold"),
            padx=14 if self.compact_ui else 18,
            pady=6 if self.compact_ui else 9,
        )
        self.state_label.pack(side=tk.RIGHT)

        config = self._card(outer)
        config.pack(fill=tk.X, pady=(0, gap))
        config_inner = tk.Frame(
            config, bg=self.CARD, padx=12 if self.compact_ui else 16, pady=8 if self.compact_ui else 12
        )
        config_inner.pack(fill=tk.X)
        fields = [
            ("CAN 通道", self.channel_var, 8 if self.compact_ui else 10, 0),
            ("采样频率 Hz", self.hz_var, 9 if self.compact_ui else 11, 1),
            ("时长（0=不限）", self.duration_var, 11 if self.compact_ui else 13, 2),
            ("文件名前缀", self.session_var, 14 if self.compact_ui else 18, 3),
            ("保存目录", self.directory_var, 28 if self.compact_ui else 42, 4),
        ]
        for label, variable, width, column in fields:
            box = tk.Frame(config_inner, bg=self.CARD)
            box.grid(row=0, column=column, sticky=tk.EW, padx=(0, 12))
            tk.Label(
                box, text=label, bg=self.CARD, fg=self.MUTED,
                font=(self.display_font, 9, "bold"),
            ).pack(anchor=tk.W, pady=(0, 4))
            ttk.Entry(
                box, textvariable=variable, width=width, style="Nero.TEntry"
            ).pack(fill=tk.X)
        config_inner.columnconfigure(4, weight=1)
        self.choose_directory_button = tk.Button(
            config_inner,
            text="选择目录",
            command=self.choose_directory,
            **self._button_style("secondary", compact=True),
        )
        self.choose_directory_button.grid(row=0, column=5, sticky=tk.S, pady=(0, 1))

        task_box = tk.Frame(config_inner, bg=self.CARD)
        task_box.grid(row=1, column=0, columnspan=6, sticky=tk.EW, pady=(12, 0))
        tk.Label(
            task_box,
            text="任务指令（英文，写入 metadata.task，训练时用作 prompt）",
            bg=self.CARD,
            fg=self.MUTED,
            font=(self.display_font, 9, "bold"),
        ).pack(anchor=tk.W, pady=(0, 4))
        ttk.Entry(
            task_box, textvariable=self.task_var, style="Nero.TEntry"
        ).pack(fill=tk.X)

        content = tk.Frame(outer, bg=self.BG)
        content.pack(fill=tk.BOTH, expand=True)
        content.grid_columnconfigure(0, weight=3, uniform="main")
        content.grid_columnconfigure(1, weight=2, uniform="main")
        content.grid_rowconfigure(0, weight=1)

        left = tk.Frame(content, bg=self.BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = tk.Frame(content, bg=self.BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        status_card = self._card(left)
        status_card.pack(fill=tk.X, pady=(0, gap))
        status_top = tk.Frame(
            status_card,
            bg=self.CARD,
            padx=12 if self.compact_ui else 18,
            pady=8 if self.compact_ui else 14,
        )
        status_top.pack(fill=tk.X)
        tk.Label(
            status_top, text="当前采集段", bg=self.CARD, fg=self.MUTED,
            font=(self.display_font, 10, "bold"),
        ).pack(anchor=tk.W)
        stats = tk.Frame(status_top, bg=self.CARD)
        stats.pack(fill=tk.X, pady=(4, 10))
        self._stat(stats, 0, self.segment_var, "本次运行")
        self._stat(stats, 1, self.count_var, "已对齐样本")
        self._stat(stats, 2, self.elapsed_var, "采集时长")
        self._stat(stats, 3, self.rate_var, "源数据频率")
        self.progress = ttk.Progressbar(
            status_card,
            mode="determinate",
            maximum=100,
            style="Nero.Horizontal.TProgressbar",
        )
        self.progress.pack(
            fill=tk.X,
            padx=12 if self.compact_ui else 18,
            pady=(0, 10 if self.compact_ui else 16),
        )

        joint_card = self._card(left)
        joint_card.pack(fill=tk.X, pady=(0, gap))
        self._card_title(joint_card, "关节实时值", "单位：度；误差 = 从臂实测 − 主臂目标")
        joint_grid = tk.Frame(
            joint_card, bg=self.CARD, padx=10 if self.compact_ui else 14, pady=6 if self.compact_ui else 8
        )
        joint_grid.pack(fill=tk.X)
        tk.Label(joint_grid, text="", bg=self.CARD).grid(row=0, column=0, padx=5)
        for index in range(7):
            tk.Label(
                joint_grid, text=f"J{index + 1}", bg=self.CARD, fg=self.MUTED,
                font=(self.display_font, 9, "bold"),
            ).grid(row=0, column=index + 1, sticky=tk.EW, padx=5, pady=(0, 5))
            joint_grid.columnconfigure(index + 1, weight=1)
        self._joint_row(joint_grid, 1, "主臂目标", self.leader_vars)
        self._joint_row(joint_grid, 2, "从臂实测", self.follower_vars)
        self._joint_row(joint_grid, 3, "跟随误差", self.error_vars, error=True)

        camera_card = self._card(left)
        camera_card.pack(fill=tk.BOTH, expand=True)
        self._card_title(
            camera_card,
            f"{CAMERA_LABEL} RGB 预览",
            f"{DEFAULT_CAMERA_WIDTH}×{DEFAULT_CAMERA_HEIGHT} · 旋转{DEFAULT_CAMERA_ROTATE}° · "
            + (
                "RealSense D435i (pyrealsense2)"
                if DEFAULT_CAMERA_BACKEND == "realsense"
                else "RealSense D435i (UVC/OpenCV)"
            ),
        )
        camera_holder = tk.Frame(camera_card, bg="#1f2933", height=self.camera_min_h)
        camera_holder.pack(
            fill=tk.BOTH, expand=True, padx=10 if self.compact_ui else 14, pady=(0, 8 if self.compact_ui else 12)
        )
        # Allow shrinking below request height on short screens.
        camera_holder.pack_propagate(not self.compact_ui)
        self.camera_preview_label = tk.Label(
            camera_holder,
            text="开始采集后显示相机画面",
            bg="#1f2933",
            fg="#dbe5ec",
            font=(self.display_font, 11 if self.compact_ui else 12),
            bd=0,
        )
        self.camera_preview_label.pack(fill=tk.BOTH, expand=True)

        quality_card = self._card(right)
        quality_card.pack(fill=tk.X, pady=(0, gap))
        self._card_title(quality_card, "数据质量", "每一行均按采样时刻做最近值对齐")
        quality_inner = tk.Frame(
            quality_card, bg=self.CARD, padx=12 if self.compact_ui else 16, pady=6 if self.compact_ui else 8
        )
        quality_inner.pack(fill=tk.X)
        self.can_quality_label = self._quality_row(
            quality_inner, "机械臂", self.can_quality_var, 0
        )
        self.align_quality_label = self._quality_row(
            quality_inner, "时间对齐", self.align_quality_var, 1
        )
        self.hand_quality_label = self._quality_row(
            quality_inner, "夹爪 / TCP", self.hand_quality_var, 2
        )
        self.camera_quality_label = self._quality_row(
            quality_inner, "RGB 相机", self.camera_quality_var, 3
        )
        self.tcp_jump_label = self._quality_row(
            quality_inner, "TCP 连续性", self.tcp_jump_var, 4
        )

        hand_card = self._card(right)
        hand_card.pack(fill=tk.X, pady=(0, gap))
        self._card_title(hand_card, "末端 / AgxGripper", "夹爪 grasp 0..1 · TCP 含 13cm 偏置")
        hand_inner = tk.Frame(
            hand_card, bg=self.CARD, padx=12 if self.compact_ui else 16, pady=6 if self.compact_ui else 8
        )
        hand_inner.pack(fill=tk.X)
        self._value_row(hand_inner, 0, "示教器输入", self.teach_value_var)
        self._value_row(hand_inner, 1, "夹爪 grasp", self.target_hand_var)
        self._value_row(hand_inner, 2, "夹爪反馈", self.feedback_value_var)
        self._value_row(hand_inner, 3, "TCP xyz", self.pose_tcp_var)
        self._value_row(hand_inner, 4, "法兰 xyz", self.pose_flange_var)
        self._value_row(hand_inner, 5, "状态", self.gripper_status_var)
        self.hand_sync_check = tk.Checkbutton(
            hand_inner,
            text="（遗留）示教器 → Revo2 同步，默认关闭",
            variable=self.hand_sync_var,
            command=self.on_hand_sync_changed,
            bg=self.CARD,
            fg=self.MUTED,
            activebackground=self.CARD,
            activeforeground=self.MUTED,
            selectcolor="#e7f3ed",
            font=(self.display_font, 9 if self.compact_ui else 10),
            cursor="hand2",
            pady=4 if self.compact_ui else 6,
        )
        self.hand_sync_check.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))

        action_card = self._card(right)
        action_card.pack(fill=tk.BOTH, expand=True)
        self._card_title(action_card, "采集控制", "每段生成对齐的 JSONL + RGB MP4")
        actions = tk.Frame(
            action_card, bg=self.CARD, padx=12 if self.compact_ui else 16, pady=6 if self.compact_ui else 8
        )
        actions.pack(fill=tk.X)
        self.start_button = tk.Button(
            actions, text="▶  开始采集", command=self.start_recording,
            **self._button_style("primary"),
        )
        self.start_button.grid(row=0, column=0, sticky=tk.EW, padx=(0, 6))
        self.stop_button = tk.Button(
            actions, text="■  停止并保存", command=self.stop_recording,
            state=tk.DISABLED, **self._button_style("secondary"),
        )
        self.stop_button.grid(row=0, column=1, sticky=tk.EW, padx=(6, 0))
        self.delete_button = tk.Button(
            actions, text="删除本段", command=self.delete_current_segment,
            state=tk.DISABLED, **self._button_style("danger", compact=True),
        )
        self.delete_button.grid(row=1, column=0, sticky=tk.EW, padx=(0, 6), pady=(10, 0))
        self.open_directory_button = tk.Button(
            actions, text="打开保存目录", command=self.open_directory,
            **self._button_style("secondary", compact=True),
        )
        self.open_directory_button.grid(row=1, column=1, sticky=tk.EW, padx=(6, 0), pady=(10, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        side_pad = 12 if self.compact_ui else 16
        tk.Label(
            action_card, text="当前文件", bg=self.CARD, fg=self.MUTED,
            font=(self.display_font, 9, "bold"), padx=side_pad,
        ).pack(anchor=tk.W, pady=(2, 2))
        self.file_label = tk.Label(
            action_card, textvariable=self.file_var, bg=self.CARD, fg=self.INK,
            font=(self.mono_font, 9), anchor=tk.W, justify=tk.LEFT,
            wraplength=320 if self.compact_ui else 500, padx=side_pad,
        )
        self.file_label.pack(fill=tk.X, pady=(0, 8 if self.compact_ui else 10))
        self.footer_label = tk.Label(
            action_card, textvariable=self.footer_var, bg=self.CARD, fg=self.MUTED,
            font=(self.display_font, 9), anchor=tk.W, justify=tk.LEFT,
            wraplength=340 if self.compact_ui else 520, padx=side_pad,
        )
        self.footer_label.pack(fill=tk.X, pady=(0, 10 if self.compact_ui else 14))

        log_card = self._card(outer)
        log_card.pack(fill=tk.X, pady=(gap, 0))
        log_header = tk.Frame(log_card, bg=self.CARD, padx=12 if self.compact_ui else 14, pady=5 if self.compact_ui else 7)
        log_header.pack(fill=tk.X)
        tk.Label(
            log_header, text="运行信息", bg=self.CARD, fg=self.INK,
            font=(self.display_font, 10, "bold"),
        ).pack(side=tk.LEFT)
        self.log_text = tk.Text(
            log_card, height=3 if self.compact_ui else 4, wrap=tk.WORD, state=tk.DISABLED,
            bg="#f7f8fa", fg="#41566c", relief=tk.FLAT,
            font=(self.mono_font, 8), padx=10, pady=6 if self.compact_ui else 7,
        )
        self.log_text.pack(fill=tk.X, padx=12 if self.compact_ui else 14, pady=(0, 8 if self.compact_ui else 12))
        self.append_log(
            "准备就绪。确认英文任务指令后按空格开始；再按空格停止并保存，可连续下一段。"
        )

    def _card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=self.CARD,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )

    def _card_title(self, parent: tk.Widget, title: str, subtitle: str) -> None:
        title_row = tk.Frame(
            parent,
            bg=self.CARD,
            padx=12 if self.compact_ui else 16,
            pady=7 if self.compact_ui else 11,
        )
        title_row.pack(fill=tk.X)
        tk.Label(
            title_row, text=title, bg=self.CARD, fg=self.INK,
            font=(self.display_font, 11 if self.compact_ui else 13, "bold"),
        ).pack(side=tk.LEFT)
        if not self.compact_ui:
            tk.Label(
                title_row, text=subtitle, bg=self.CARD, fg=self.MUTED,
                font=(self.display_font, 9),
            ).pack(side=tk.RIGHT)

    def _button_style(self, kind: str, compact: bool = False) -> dict[str, Any]:
        palette = {
            "primary": (self.GREEN, "#ffffff", "#24795a"),
            "secondary": ("#eef4f8", self.INK, "#dce9f0"),
            "danger": ("#fff0ea", "#8a2f21", "#f7dfd4"),
        }
        bg, fg, active = palette[kind]
        if self.compact_ui:
            padx = 10 if compact else 14
            pady = 6 if compact else 8
            font_size = 9 if compact else 11
        else:
            padx = 14 if compact else 18
            pady = 8 if compact else 12
            font_size = 10 if compact else 12
        return {
            "bg": bg,
            "fg": fg,
            "activebackground": active,
            "activeforeground": fg,
            "disabledforeground": "#9aa6b2",
            "relief": tk.FLAT,
            "borderwidth": 0,
            "padx": padx,
            "pady": pady,
            "font": (self.display_font, font_size, "bold"),
            "cursor": "hand2",
        }

    def _stat(self, parent: tk.Widget, column: int, value: tk.StringVar, label: str) -> None:
        box = tk.Frame(parent, bg=self.CARD)
        box.grid(row=0, column=column, sticky=tk.EW, padx=(0, 12))
        parent.columnconfigure(column, weight=1)
        tk.Label(
            box, textvariable=value, bg=self.CARD, fg=self.INK,
            font=(self.display_font, 13 if self.compact_ui else 15, "bold"), anchor=tk.W,
        ).pack(fill=tk.X)
        tk.Label(
            box, text=label, bg=self.CARD, fg=self.MUTED,
            font=(self.display_font, 8), anchor=tk.W,
        ).pack(fill=tk.X)

    def _joint_row(
        self,
        parent: tk.Widget,
        row: int,
        name: str,
        variables: list[tk.StringVar],
        error: bool = False,
    ) -> None:
        row_pad = 2 if self.compact_ui else 5
        tk.Label(
            parent, text=name, bg=self.CARD, fg=self.MUTED,
            font=(self.display_font, 9, "bold"),
        ).grid(row=row, column=0, sticky=tk.W, padx=5, pady=row_pad)
        color = self.RED if error else self.INK
        for index, variable in enumerate(variables):
            tk.Label(
                parent, textvariable=variable, bg=self.CARD, fg=color,
                font=(self.mono_font, 9 if self.compact_ui else 10), anchor=tk.E,
            ).grid(row=row, column=index + 1, sticky=tk.EW, padx=5, pady=row_pad)

    def _value_row(
        self, parent: tk.Widget, row: int, name: str, variable: tk.StringVar
    ) -> None:
        row_pad = 2 if self.compact_ui else 5
        tk.Label(
            parent, text=name, bg=self.CARD, fg=self.MUTED,
            font=(self.display_font, 9, "bold"),
        ).grid(row=row, column=0, sticky=tk.W, padx=(0, 14), pady=row_pad)
        tk.Label(
            parent, textvariable=variable, bg=self.CARD, fg=self.INK,
            font=(self.mono_font, 9), anchor=tk.W, justify=tk.LEFT,
        ).grid(row=row, column=1, sticky=tk.EW, pady=row_pad)
        parent.columnconfigure(1, weight=1)

    def _quality_row(
        self, parent: tk.Widget, name: str, variable: tk.StringVar, row: int
    ) -> tk.Label:
        row_pad = 2 if self.compact_ui else 5
        tk.Label(
            parent, text=name, bg=self.CARD, fg=self.MUTED,
            font=(self.display_font, 9, "bold"),
        ).grid(row=row, column=0, sticky=tk.W, pady=row_pad)
        label = tk.Label(
            parent, textvariable=variable, bg=self.CARD, fg=self.MUTED,
            font=(self.display_font, 9 if self.compact_ui else 10, "bold"), anchor=tk.E,
        )
        label.grid(row=row, column=1, sticky=tk.EW, pady=row_pad)
        parent.columnconfigure(1, weight=1)
        return label

    def append_log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{stamp}] {text.rstrip()}\n")
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 300:
            self.log_text.delete("1.0", "50.0")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def choose_directory(self) -> None:
        selected = filedialog.askdirectory(
            title="选择录制保存目录", initialdir=self.directory_var.get() or str(HERE)
        )
        if selected:
            self.directory_var.set(selected)

    def open_directory(self) -> None:
        directory = Path(self.directory_var.get()).expanduser()
        try:
            _open_path_in_file_manager(directory)
        except Exception as exc:
            messagebox.showerror("无法打开目录", str(exc))

    def _write_sync_control(self) -> None:
        if self.sync_control_path is None:
            return
        self.sync_control_path.write_text(
            "1\n" if self.hand_sync_var.get() else "0\n",
            encoding="utf-8",
        )

    def _cleanup_sync_control(self) -> None:
        if self.sync_control_path is not None:
            try:
                self.sync_control_path.unlink(missing_ok=True)
            except OSError:
                pass
        self.sync_control_path = None

    def _cleanup_camera_preview(self) -> None:
        if self.camera_preview_path is not None:
            try:
                self.camera_preview_path.unlink(missing_ok=True)
            except OSError:
                pass
        self.camera_preview_path = None
        self.camera_preview_photo = None
        self.camera_preview_mtime_ns = 0
        if hasattr(self, "camera_preview_label"):
            self.camera_preview_label.configure(
                image="", text="相机预览已停止"
            )

    def _start_idle_camera_preview(self) -> None:
        if self.process is not None:
            return
        if (
            self.idle_camera_process is not None
            and self.idle_camera_process.poll() is None
        ):
            return
        if self.camera_preview_path is None:
            self.camera_preview_path = Path(tempfile.gettempdir()) / (
                f"nero_teleop_gui_{os.getpid()}_camera.png"
            )
        self.idle_camera_ready = False
        self.idle_camera_error = None
        self.idle_camera_error_reported = None
        while True:
            try:
                self.idle_camera_queue.get_nowait()
            except queue.Empty:
                break
        self.camera_preview_label.configure(image="", text=f"正在连接 {CAMERA_LABEL}…")
        preview_args = [
            sys.executable,
            "-u",
            str(IDLE_CAMERA_PREVIEW),
            str(self.camera_preview_path),
            "--backend",
            DEFAULT_CAMERA_BACKEND,
            "--rotate",
            str(DEFAULT_CAMERA_ROTATE),
        ]
        if DEFAULT_CAMERA_BACKEND == "opencv":
            preview_args += ["--camera-index", str(DEFAULT_CAMERA_INDEX)]
        else:
            preview_args += ["--camera-serial", DEFAULT_CAMERA_SERIAL]
        try:
            self.idle_camera_process = subprocess.Popen(
                preview_args,
                cwd=str(HERE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **_subprocess_group_kwargs(),
            )
        except Exception as exc:
            self.idle_camera_process = None
            self.idle_camera_error = str(exc)
            return
        threading.Thread(
            target=self._read_idle_camera_output,
            daemon=True,
        ).start()

    def _read_idle_camera_output(self) -> None:
        process = self.idle_camera_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.idle_camera_queue.put(line)

    def _stop_idle_camera_preview(self, *, wait_release: bool = False) -> None:
        process = self.idle_camera_process
        if process is not None and process.poll() is None:
            try:
                # Prefer immediate release on Windows so recorder can reopen DSHOW.
                if IS_WINDOWS:
                    _force_kill_process(process)
                else:
                    _interrupt_process(process)
                    process.wait(timeout=3.0)
            except (ProcessLookupError, PermissionError):
                pass
            except subprocess.TimeoutExpired:
                _force_kill_process(process)
        if process is not None and process.stdout is not None:
            try:
                process.stdout.close()
            except Exception:
                pass
        self.idle_camera_process = None
        self.idle_camera_ready = False
        if wait_release and IS_WINDOWS:
            # DirectShow often keeps the device locked briefly after the process dies.
            time.sleep(1.2)

    def _consume_idle_camera_output(self) -> None:
        while True:
            try:
                line = self.idle_camera_queue.get_nowait().strip()
            except queue.Empty:
                break
            if line.startswith("CAMERA_READY"):
                self.idle_camera_ready = True
                self.idle_camera_error = None
            elif line.startswith("CAMERA_FRAME"):
                self.idle_camera_ready = True
                self.idle_camera_error = None
                try:
                    self.camera_preview_frame_hint = int(line.split()[1])
                except (IndexError, ValueError):
                    self.camera_preview_frame_hint += 1
            elif line.startswith("CAMERA_ERROR"):
                self.idle_camera_error = line.removeprefix("CAMERA_ERROR ").strip()
            elif line:
                # Non-fatal chatter
                pass
        process = self.idle_camera_process
        if (
            process is not None
            and process.poll() is not None
            and not self.idle_camera_error
            and not self.idle_camera_ready
        ):
            self.idle_camera_error = f"预览进程退出，返回码 {process.returncode}"

    def _update_camera_preview(self) -> None:
        path = self.camera_preview_path
        if path is None or not path.exists():
            return
        tick_path = path.with_suffix(".tick")
        if tick_path.exists():
            try:
                self.camera_preview_frame_hint = int(
                    tick_path.read_text(encoding="utf-8").strip() or "0"
                )
            except (OSError, ValueError):
                pass
        try:
            mtime_ns = path.stat().st_mtime_ns
            needs_reload = (
                mtime_ns != self.camera_preview_mtime_ns
                or self.camera_preview_frame_hint != getattr(
                    self, "_last_loaded_frame_hint", -2
                )
            )
            if not needs_reload:
                return
            import cv2  # type: ignore

            image = cv2.imread(str(path))
            if image is None:
                return
            img_h, img_w = image.shape[:2]
            # Fit the whole frame inside the preview panel (letterboxed),
            # preserving aspect ratio -- fixes cropped/oversized preview on
            # small screens. Never upscale past 1x so small frames stay sharp.
            box_w = max(self.camera_preview_label.winfo_width(), 320)
            box_h = max(self.camera_preview_label.winfo_height(), self.camera_min_h)
            scale = min(box_w / img_w, box_h / img_h, 1.0)
            new_w = max(int(img_w * scale), 1)
            new_h = max(int(img_h * scale), 1)
            if (new_w, new_h) != (img_w, img_h):
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                image = cv2.resize(image, (new_w, new_h), interpolation=interp)
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                return
            # Passing raw bytes (not a file path) sidesteps Tk's PhotoImage
            # filename-based caching, so every frame is a genuinely new image.
            photo = tk.PhotoImage(data=encoded.tobytes())
        except (OSError, tk.TclError):
            return
        self.camera_preview_mtime_ns = mtime_ns
        self._last_loaded_frame_hint = self.camera_preview_frame_hint
        self.camera_preview_photo = photo
        self.camera_preview_label.configure(image=photo, text="")

    def on_hand_sync_changed(self) -> None:
        enabled = self.hand_sync_var.get()
        if self.sync_control_path is not None:
            try:
                self._write_sync_control()
            except OSError as exc:
                self.hand_sync_var.set(not enabled)
                messagebox.showerror("末端同步切换失败", str(exc))
                return
        self.hand_sync_status_var.set("已开启" if enabled else "已关闭")
        self.append_log(f"末端同步：{'开启' if enabled else '关闭'}")

    def _on_space(self, _event: Optional[tk.Event] = None) -> str:
        focus = self.root.focus_get()
        if isinstance(focus, (tk.Entry, ttk.Entry, tk.Text)):
            return "break"
        if self.process is None:
            self.start_recording()
        else:
            self.stop_recording()
        return "break"

    def _on_backspace(self, _event: Optional[tk.Event] = None) -> str:
        focus = self.root.focus_get()
        if isinstance(focus, (tk.Entry, ttk.Entry, tk.Text)):
            return "break"
        if str(self.delete_button.cget("state")) != str(tk.DISABLED):
            self.delete_current_segment()
        return "break"

    def delete_current_segment(self) -> None:
        path = self.output_path
        if path is None:
            return
        if self.process is not None and self.process.poll() is None:
            self.delete_after_stop = True
            self.delete_button.configure(state=tk.DISABLED)
            self.append_log("已请求删除本段，先安全停止并落盘…")
            self.stop_recording()
            return
        errors = self._delete_episode_files()
        if errors:
            messagebox.showerror("删除失败", "\n".join(errors))
            return
        self.saved_segments = max(0, self.saved_segments - 1)
        self.output_path = None
        self.video_path = None
        self.file_var.set("—")
        self.segment_var.set(f"第 {self.saved_segments + 1} 段")
        self.delete_button.configure(state=tk.DISABLED)
        self.set_state("本段已删除", "normal")
        self.footer_var.set("本段已删除。按空格开始下一段采集")
        self.append_log(f"已删除本段：{path}")

    def _delete_episode_files(self) -> list[str]:
        errors = []
        for path in (self.output_path, self.video_path):
            if path is None:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"{path.name}: {exc}")
        return errors

    def _validated_settings(self) -> tuple[str, float, float, Path, str, str]:
        channel = self.channel_var.get().strip()
        if not channel:
            raise ValueError("CAN 通道不能为空")
        try:
            hz = float(self.hz_var.get())
        except ValueError as exc:
            raise ValueError("采样频率必须是数字") from exc
        if not 0 < hz <= 500:
            raise ValueError("采样频率应在 0～500 Hz 之间")
        try:
            duration = float(self.duration_var.get())
        except ValueError as exc:
            raise ValueError("录制时长必须是数字") from exc
        if duration < 0:
            raise ValueError("录制时长不能为负数")
        directory = Path(self.directory_var.get()).expanduser().resolve()
        task = validate_task(self.task_var.get())
        self.task_var.set(task)
        session_name = sanitize_session_name(self.session_var.get())
        self.session_var.set(session_name)
        return channel, hz, duration, directory, session_name, task

    def start_recording(self) -> None:
        if self.process is not None:
            return
        try:
            channel, hz, duration, directory, session_name, task = self._validated_settings()
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("设置错误", str(exc))
            return
        if not RECORDER.exists():
            messagebox.showerror("缺少后端", f"找不到 {RECORDER}")
            return

        self._stop_idle_camera_preview(wait_release=True)
        self.append_log("已释放预览相机，正在启动录制进程…")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output_path = directory / f"{session_name}_{stamp}.jsonl"
        self.video_path = self.output_path.with_suffix(".rgb.mp4")
        assert self.camera_preview_path is not None
        self._cleanup_sync_control()
        self.sync_control_path = None
        if self.hand_sync_var.get():
            self.sync_control_path = Path(tempfile.gettempdir()) / (
                f"nero_teleop_gui_{os.getpid()}_{stamp}.sync"
            )
            try:
                self._write_sync_control()
            except OSError as exc:
                self.sync_control_path = None
                self._start_idle_camera_preview()
                messagebox.showerror("无法创建末端同步开关", str(exc))
                return
        command = [
            sys.executable,
            "-u",
            str(RECORDER),
            "--interface",
            _default_can_interface(),
            "--channel",
            channel,
            "--hz",
            str(hz),
            "--hand-hz",
            str(hz),
            "--output",
            str(self.output_path),
            "--task",
            task,
            "--camera",
            "--camera-backend",
            DEFAULT_CAMERA_BACKEND,
        ]
        if DEFAULT_CAMERA_BACKEND == "opencv":
            command += ["--camera-index", str(DEFAULT_CAMERA_INDEX)]
        else:
            command += ["--camera-serial", DEFAULT_CAMERA_SERIAL]
        command += [
            "--camera-width",
            str(DEFAULT_CAMERA_WIDTH),
            "--camera-height",
            str(DEFAULT_CAMERA_HEIGHT),
            "--camera-rotate",
            str(DEFAULT_CAMERA_ROTATE),
            "--camera-video",
            str(self.video_path),
            "--camera-preview-path",
            str(self.camera_preview_path),
            "--no-execute-hand",
            "--keep-unaligned",
        ]
        if self.sync_control_path is not None:
            # Drop conflicting --no-execute-hand by rebuilding with execute-hand.
            command = [c for c in command if c != "--no-execute-hand"]
            command.extend(
                [
                    "--execute-hand",
                    "--sync-control-file",
                    str(self.sync_control_path),
                ]
            )
        if duration > 0:
            command.extend(["--duration", str(duration)])

        self._reset_live_values()
        self.stop_requested = False
        self.close_after_stop = False
        self.delete_after_stop = False
        self.file_var.set(f"{self.output_path.name}\n{self.video_path.name}")
        self.segment_var.set(f"第 {self.saved_segments + 1} 段")
        self.progress["value"] = 0
        self.footer_var.set("正在连接 CAN / 相机，请稍候…")
        self.set_state("正在连接…", "running")
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(HERE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **_subprocess_group_kwargs(),
            )
        except Exception as exc:
            self.process = None
            self._cleanup_sync_control()
            self._start_idle_camera_preview()
            self.set_state("启动失败", "error")
            messagebox.showerror("启动失败", str(exc))
            return

        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.delete_button.configure(state=tk.NORMAL)
        self._set_settings_enabled(False)
        self.footer_var.set("采集中：空格停止并保存")
        self.append_log(f"开始采集：{self.output_path}")
        self.append_log(f"task: {task}")
        camera_desc = (
            f"camera index={DEFAULT_CAMERA_INDEX}"
            if DEFAULT_CAMERA_BACKEND == "opencv"
            else f"camera serial={DEFAULT_CAMERA_SERIAL}"
        )
        self.append_log(
            f"CAN {_default_can_interface()}:{channel} · {camera_desc}"
        )
        thread = threading.Thread(target=self._read_process_output, daemon=True)
        thread.start()

    def _read_process_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.stdout_queue.put(line)

    def stop_recording(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        if self.stop_requested:
            return
        self.stop_requested = True
        self.stop_button.configure(state=tk.DISABLED)
        self.set_state("正在安全停止…", "running")
        self.footer_var.set("正在补齐末帧、刷新缓冲区并强制落盘…")
        self.append_log("正在停止采集并强制落盘…")
        try:
            _interrupt_process(process)
        except Exception as exc:
            self.append_log(f"停止信号发送失败，改用强制结束：{exc}")
            _force_kill_process(process)
        # If the child ignores the interrupt, escalate after a few seconds.
        self.root.after(8000, self._force_stop_if_still_running)

    def _force_stop_if_still_running(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        self.append_log("录制进程未退出，正在强制结束…")
        _force_kill_process(process)

    def _set_settings_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for child in self.root.winfo_children():
            self._set_entries_recursive(child, state)
        self.choose_directory_button.configure(state=state)

    def _set_entries_recursive(self, widget: tk.Widget, state: str) -> None:
        for child in widget.winfo_children():
            if isinstance(child, (ttk.Entry, tk.Entry)):
                child.configure(state=state)
            else:
                self._set_entries_recursive(child, state)

    def _open_tail_if_ready(self) -> None:
        if self.tail_file is None and self.output_path is not None and self.output_path.exists():
            self.tail_file = self.output_path.open("r", encoding="utf-8")

    def _read_new_samples(self) -> None:
        self._open_tail_if_ready()
        if self.tail_file is None:
            return
        chunk = self.tail_file.read()
        if not chunk:
            return
        self.tail_buffer += chunk
        lines = self.tail_buffer.split("\n")
        self.tail_buffer = lines.pop()
        for line in lines:
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                self.append_log("忽略一条不完整的 JSON 记录")
                continue
            if record.get("kind") == "sample":
                self.samples += 1
                self.latest_sample = record

    def _update_live_display(self) -> None:
        sample = self.latest_sample
        self.count_var.set(str(self.samples))
        if sample is None:
            return
        elapsed = float(sample.get("elapsed_s", 0.0))
        self.elapsed_var.set(f"{elapsed:.1f} s")
        try:
            duration = float(self.duration_var.get())
        except ValueError:
            duration = 0.0
        self.progress["value"] = min(elapsed / duration * 100.0, 100.0) if duration > 0 else 0

        leader = sample.get("leader")
        follower = sample.get("follower")
        leader_values = leader.get("position_rad") if leader else None
        follower_values = follower.get("position_rad") if follower else None
        self._set_joint_values(self.leader_vars, leader_values)
        self._set_joint_values(self.follower_vars, follower_values)
        if leader_values is not None and follower_values is not None:
            errors = [follower_values[i] - leader_values[i] for i in range(7)]
            self._set_joint_values(self.error_vars, errors, signed=True)
            sequence = int(sample.get("seq", self.samples))
            if sequence != self.last_trace_seq:
                self.last_trace_seq = sequence
                self.trace_history.append(
                    (sequence, [math.degrees(value) for value in errors])
                )
                del self.trace_history[:-300]
                self._draw_trace()
        else:
            self._set_joint_values(self.error_vars, None)

        rates = []
        if leader:
            rates.append(f"主 {leader.get('source_hz', 0):.0f}Hz")
        if follower:
            rates.append(f"从 {follower.get('source_hz', 0):.0f}Hz")
        self.rate_var.set(" / ".join(rates) if rates else "无数据")

        teach = sample.get("gripper_ctrl")
        nero_feedback = sample.get("gripper_feedback")
        gripper = sample.get("gripper") or {}
        poses = sample.get("poses") or {}
        training_state = ((sample.get("training") or {}).get("state") or {})
        camera_rgb = sample.get("camera_rgb")
        alignment = sample.get("alignment") or {}
        sync_enabled = bool(sample.get("revo2_sync_enabled", False))

        if teach:
            self.teach_value_var.set(
                f"{teach['value']:.5g} {teach['value_unit']}  ·  raw {teach['raw_value']}"
            )
            self.teach_raw_var.set(str(teach["raw_value"]))
            self.teach_force_var.set(f"{teach['force_n']:.3f} N")
        else:
            self.teach_value_var.set("无 0x159 数据")
            self.teach_raw_var.set("—")
            self.teach_force_var.set("—")

        action_grasp = gripper.get("action_grasp")
        state_grasp = gripper.get("state_grasp")
        if action_grasp is None:
            action_grasp = training_state.get("gripper_grasp") or training_state.get(
                "hand_grasp"
            )
        if state_grasp is None:
            state_grasp = training_state.get("gripper_grasp") or training_state.get(
                "hand_grasp"
            )
        if action_grasp is not None:
            self.target_hand_var.set(f"{float(action_grasp):.3f}")
        else:
            self.target_hand_var.set("—")
        if state_grasp is not None:
            self.feedback_value_var.set(f"{float(state_grasp):.3f}")
        elif nero_feedback is not None:
            self.feedback_value_var.set(
                f"{nero_feedback['value']:.5g} {nero_feedback.get('value_unit', '')}"
            )
        else:
            self.feedback_value_var.set("等待夹爪反馈")

        tcp = poses.get("tcp_pose") or training_state.get("tcp_pose")
        flange = poses.get("flange_pose") or training_state.get("flange_pose")
        if tcp is not None and len(tcp) >= 3:
            self.pose_tcp_var.set(
                f"[{float(tcp[0]):.3f}, {float(tcp[1]):.3f}, {float(tcp[2]):.3f}] m"
            )
        else:
            self.pose_tcp_var.set("—")
        if flange is not None and len(flange) >= 3:
            self.pose_flange_var.set(
                f"[{float(flange[0]):.3f}, {float(flange[1]):.3f}, {float(flange[2]):.3f}] m"
            )
        else:
            self.pose_flange_var.set("—")

        status_parts = []
        if sync_enabled:
            status_parts.append("Revo2同步开")
        status_parts.append("夹爪录制中")
        if nero_feedback:
            status_parts.append(f"CAN 0x2A8=0x{nero_feedback['status_code']:02X}")
        self.gripper_status_var.set("  ·  ".join(status_parts))
        self.hand_sync_status_var.set("已开启" if sync_enabled else "已关闭")

        can_skew = float(alignment.get("can_source_skew_ms", 0.0))
        can_ok = leader is not None and follower is not None
        can_valid = bool(alignment.get("can_valid_10ms", can_ok))
        self._set_quality(
            self.can_quality_label,
            self.can_quality_var,
            f"● 正常 · 源偏差 {can_skew:.1f} ms" if can_ok else "● 数据缺失",
            "good" if can_ok and can_valid else "bad",
        )
        aligned = bool(alignment.get("valid", False))
        cam_age = None
        if camera_rgb is not None:
            cam_age = camera_rgb.get("age_at_sample_ms")
        cam_valid = bool(alignment.get("camera_valid_100ms", True))
        reasons = []
        if not can_valid:
            reasons.append("CAN")
        if not alignment.get("hand_valid_100ms", True):
            reasons.append("末端")
        if camera_rgb is not None and not cam_valid:
            if cam_age is not None:
                reasons.append(f"相机过旧 {float(cam_age):.0f}ms")
            else:
                reasons.append("相机")
        align_text = (
            "● 本帧有效"
            if aligned
            else ("● 未对齐 · " + "/".join(reasons) if reasons else "● 本帧未对齐")
        )
        self._set_quality(
            self.align_quality_label,
            self.align_quality_var,
            align_text,
            "good" if aligned else "bad",
        )
        gripper_ok = state_grasp is not None or nero_feedback is not None
        tcp_ok = tcp is not None
        self._set_quality(
            self.hand_quality_label,
            self.hand_quality_var,
            (
                f"● grasp={float(state_grasp):.2f}" + (" · TCP√" if tcp_ok else " · TCP×")
                if gripper_ok
                else ("● 仅 TCP√" if tcp_ok else "● 无夹爪/TCP")
            ),
            "good" if gripper_ok or tcp_ok else "bad",
        )
        camera_fresh = cam_age is not None and 0.0 <= float(cam_age) <= 500.0
        self._set_quality(
            self.camera_quality_label,
            self.camera_quality_var,
            (
                f"● 帧 #{camera_rgb['video_frame_index']} · "
                f"{float(cam_age):.1f} ms"
                if camera_rgb and cam_age is not None
                else ("● 有相机帧" if camera_rgb else "● 无相机帧")
            ),
            "good" if camera_rgb and camera_fresh else "bad",
        )

        quality = sample.get("quality") or {}
        jump_mm = quality.get("tcp_jump_mm")
        jump_deg = quality.get("tcp_jump_deg")
        jump_warn = bool(quality.get("tcp_jump_warn"))
        if jump_mm is not None:
            self.max_jump_mm = max(self.max_jump_mm, float(jump_mm))
        if jump_deg is not None:
            self.max_jump_deg = max(self.max_jump_deg, float(jump_deg))
        if jump_warn:
            seq = sample.get("seq")
            if seq != self._jump_log_seq:
                self._jump_log_seq = seq
                self.jump_warn_count += 1
                if self.jump_warn_count <= 5 or self.jump_warn_count % 20 == 0:
                    self.append_log(
                        f"TCP 跳变告警 #{self.jump_warn_count}: "
                        f"{float(jump_mm or 0.0):.1f} mm / "
                        f"{float(jump_deg or 0.0):.1f}°"
                    )
            self._set_quality(
                self.tcp_jump_label,
                self.tcp_jump_var,
                (
                    f"● 跳变 {float(jump_mm or 0.0):.0f}mm / "
                    f"{float(jump_deg or 0.0):.1f}° · 累计 {self.jump_warn_count}"
                ),
                "bad",
            )
        elif jump_mm is not None:
            self._set_quality(
                self.tcp_jump_label,
                self.tcp_jump_var,
                f"● 正常 · {float(jump_mm):.0f}mm / {float(jump_deg or 0.0):.1f}°",
                "good",
            )
        else:
            self._set_quality(
                self.tcp_jump_label,
                self.tcp_jump_var,
                "● 等待 TCP",
                "normal",
            )

        if jump_warn:
            self.set_state("TCP 跳变告警", "error")
        elif leader and follower:
            self.set_state("正在采集", "running")
        elif leader or follower:
            self.set_state("部分数据缺失", "error")
        else:
            self.set_state("等待 CAN 数据…", "running")

    def _set_quality(
        self,
        label: tk.Label,
        variable: tk.StringVar,
        text: str,
        kind: str,
    ) -> None:
        variable.set(text)
        colors = {
            "good": self.GREEN,
            "warn": self.AMBER,
            "bad": self.RED,
            "normal": self.MUTED,
        }
        label.configure(fg=colors.get(kind, self.MUTED))

    def _draw_trace(self) -> None:
        canvas = getattr(self, "trace_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        width = max(canvas.winfo_width(), 10)
        height = max(canvas.winfo_height(), 10)
        left, right, top, bottom = 42, 10, 12, 22
        plot_w = max(width - left - right, 1)
        plot_h = max(height - top - bottom, 1)
        if not self.trace_history:
            canvas.create_text(
                width / 2,
                height / 2,
                text="开始采集后显示 J1–J7 主从误差",
                fill=self.MUTED,
                font=(self.display_font, 10),
            )
            return
        max_abs = max(
            1.0,
            max(abs(value) for _, values in self.trace_history for value in values),
        )
        max_abs = min(max_abs * 1.1, 180.0)
        center_y = top + plot_h / 2
        canvas.create_line(left, center_y, width - right, center_y, fill="#cbd3db")
        canvas.create_line(left, top, left, height - bottom, fill="#cbd3db")
        canvas.create_text(
            left - 5, top, text=f"+{max_abs:.1f}°", anchor=tk.E,
            fill=self.MUTED, font=(self.mono_font, 7),
        )
        canvas.create_text(
            left - 5, height - bottom, text=f"−{max_abs:.1f}°", anchor=tk.E,
            fill=self.MUTED, font=(self.mono_font, 7),
        )
        colors = ["#2d8f68", "#3274a1", "#ad5d9e", "#d17a22", "#6f62a8", "#87952a", "#b54737"]
        count = len(self.trace_history)
        for joint_index, color in enumerate(colors):
            coordinates: list[float] = []
            for index, (_, values) in enumerate(self.trace_history):
                x = left + (index / max(count - 1, 1)) * plot_w
                clipped = max(-max_abs, min(max_abs, values[joint_index]))
                y = center_y - clipped / max_abs * plot_h / 2
                coordinates.extend((x, y))
            if len(coordinates) >= 4:
                canvas.create_line(*coordinates, fill=color, width=1.5, smooth=True)
            legend_x = left + joint_index * 48
            canvas.create_text(
                legend_x, height - 8, text=f"J{joint_index + 1}", anchor=tk.W,
                fill=color, font=(self.display_font, 7, "bold"),
            )

    @staticmethod
    def _set_joint_values(
        variables: list[tk.StringVar], values: Optional[list[float]], signed: bool = False
    ) -> None:
        if values is None or len(values) != 7:
            for variable in variables:
                variable.set("—")
            return
        pattern = "+.2f" if signed else ".2f"
        for variable, radians in zip(variables, values):
            variable.set(format(math.degrees(float(radians)), pattern))

    def _reset_live_values(self) -> None:
        if self.tail_file is not None:
            self.tail_file.close()
        self.tail_file = None
        self.tail_buffer = ""
        self.samples = 0
        self.latest_sample = None
        self.jump_warn_count = 0
        self.max_jump_mm = 0.0
        self.max_jump_deg = 0.0
        self._jump_log_seq = None
        self.count_var.set("0")
        self.elapsed_var.set("0.0 s")
        self.rate_var.set("—")
        self.progress["value"] = 0
        self.trace_history.clear()
        self.last_trace_seq = None
        self._draw_trace()
        for group in (self.leader_vars, self.follower_vars, self.error_vars):
            self._set_joint_values(group, None)
        for variable in (
            self.teach_value_var,
            self.teach_raw_var,
            self.teach_force_var,
            self.target_hand_var,
            self.feedback_value_var,
            self.feedback_force_var,
            self.pose_tcp_var,
            self.pose_flange_var,
            self.gripper_status_var,
        ):
            variable.set("—")
        self._set_quality(
            self.can_quality_label, self.can_quality_var, "● 等待 CAN", "bad"
        )
        self._set_quality(
            self.align_quality_label, self.align_quality_var, "● 等待数据", "bad"
        )
        self._set_quality(
            self.hand_quality_label, self.hand_quality_var, "● 等待夹爪", "bad"
        )
        self._set_quality(
            self.camera_quality_label,
            self.camera_quality_var,
            f"● 等待 {CAMERA_LABEL}",
            "bad",
        )
        self._set_quality(
            self.tcp_jump_label,
            self.tcp_jump_var,
            f"● 阈值 {TCP_JUMP_WARN_MM:.0f}mm / {TCP_JUMP_WARN_DEG:.0f}°",
            "normal",
        )

    def _finish_process(self, return_code: int) -> None:
        process = self.process
        self.process = None
        self._read_new_samples()
        self._update_live_display()
        if self.tail_file is not None:
            self.tail_file.close()
            self.tail_file = None
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self._set_settings_enabled(True)
        deleted_path: Optional[Path] = None
        delete_error: Optional[str] = None
        if self.delete_after_stop and self.output_path is not None:
            deleted_path = self.output_path
            errors = self._delete_episode_files()
            if errors:
                delete_error = "\n".join(errors)
            else:
                self.output_path = None
                self.video_path = None
                self.file_var.set("—")
        self.delete_after_stop = False
        self._cleanup_sync_control()

        if deleted_path is not None and delete_error is None:
            self.set_state("本段已删除", "normal")
            self.footer_var.set("本段已删除。按空格开始下一段采集")
            self.append_log(f"已删除本段：{deleted_path}")
        elif delete_error is not None:
            self.set_state("删除失败", "error")
            self.footer_var.set("文件删除失败，请查看运行信息")
            self.append_log(f"删除失败：{delete_error}")
            messagebox.showerror("删除失败", delete_error)
        elif return_code == 0 and self.samples == 0:
            deleted_path = self.output_path
            errors = self._delete_episode_files()
            self.output_path = None
            self.video_path = None
            self.file_var.set("—")
            self.set_state("空段已丢弃", "normal")
            self.footer_var.set("没有有效样本，已自动删除空文件")
            self.append_log(f"空段已丢弃：{deleted_path}")
            if errors:
                messagebox.showerror("删除空段失败", "\n".join(errors))
        elif return_code == 0:
            # Keep continuous space start/stop flow: save immediately, no modal.
            # Bad takes are removed via「删除本段」/ Backspace.
            self.saved_segments += 1
            self.success_segments += 1
            self.set_state("已保存", "normal")
            self.footer_var.set(
                f"已保存 {self.saved_segments} 段。空格下一段，有问题点「删除本段」"
            )
            jump_note = (
                f" · tcp_jumps={self.jump_warn_count}"
                if self.jump_warn_count > 0
                else ""
            )
            self.append_log(f"采集完成：{self.samples} 样本{jump_note}")
        else:
            self.set_state(f"异常结束 ({return_code})", "error")
            self.footer_var.set("采集器异常退出，请查看运行信息后重试")
            self.append_log(f"录制器异常退出，返回码 {return_code}")
        self.delete_button.configure(
            state=(
                tk.NORMAL
                if self.output_path is not None and self.output_path.exists()
                else tk.DISABLED
            )
        )
        if process is not None and process.stdout is not None:
            process.stdout.close()
        if self.close_after_stop:
            self._cleanup_camera_preview()
            self.root.destroy()
        else:
            self._start_idle_camera_preview()

    def set_state(self, text: str, kind: str) -> None:
        self.state_var.set(text)
        colors = {
            "running": ("#e7f3ed", self.GREEN),
            "error": ("#fff0ea", self.RED),
            "normal": ("#eaf1f6", self.INK),
        }
        bg, fg = colors.get(kind, ("#e9eef2", self.INK))
        self.state_label.configure(bg=bg, fg=fg)

    def poll(self) -> None:
        self._consume_idle_camera_output()
        self._update_camera_preview()
        if self.process is None:
            if self.idle_camera_ready:
                self._set_quality(
                    self.camera_quality_label,
                    self.camera_quality_var,
                    "● 实时预览中 · 30 Hz",
                    "good",
                )
            elif self.idle_camera_error:
                self._set_quality(
                    self.camera_quality_label,
                    self.camera_quality_var,
                    "● 相机连接失败",
                    "bad",
                )
                self.camera_preview_label.configure(
                    image="", text=f"{CAMERA_LABEL} 预览失败\n{self.idle_camera_error}"
                )
                if self.idle_camera_error != self.idle_camera_error_reported:
                    self.append_log(f"{CAMERA_LABEL} 预览失败：{self.idle_camera_error}")
                    self.idle_camera_error_reported = self.idle_camera_error
        while True:
            try:
                line = self.stdout_queue.get_nowait()
            except queue.Empty:
                break
            clean = line.replace("\r", "").strip()
            if clean:
                if clean.startswith("CAMERA_FRAME"):
                    try:
                        self.camera_preview_frame_hint = int(clean.split()[1])
                    except (IndexError, ValueError):
                        self.camera_preview_frame_hint += 1
                else:
                    self.append_log(clean)
                if "Recording to" in clean and self.process is not None:
                    self.set_state("正在采集", "running")
                    self.footer_var.set("采集中：空格停止并保存")
                if "Connect failed" in clean or "camera start failed" in clean:
                    self.set_state("连接告警", "error")

        if self.process is not None:
            self._read_new_samples()
            self._update_live_display()
            return_code = self.process.poll()
            if return_code is not None:
                self._finish_process(return_code)
        self.root.after(self.POLL_MS, self.poll)

    def on_close(self) -> None:
        if self.process is None:
            self._stop_idle_camera_preview()
            self._cleanup_camera_preview()
            self._cleanup_sync_control()
            self.root.destroy()
            return
        if messagebox.askyesno("正在录制", "停止录制、保存文件并退出吗？"):
            self.close_after_stop = True
            self.stop_recording()


def main() -> None:
    root = tk.Tk()
    try:
        # Adaptive: 4K desks keep a bump; 2.5K laptops stay near OS DPI (was fixed 1.35).
        root.tk.call("tk", "scaling", _preferred_tk_scaling(root))
    except tk.TclError:
        pass
    TeleopRecorderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
