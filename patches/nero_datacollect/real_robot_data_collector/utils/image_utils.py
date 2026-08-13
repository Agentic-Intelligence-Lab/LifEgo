from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


def resize_frame(frame: np.ndarray, width: int | None = None, height: int | None = None) -> np.ndarray:
    if width is None and height is None:
        return frame
    import cv2

    current_height, current_width = frame.shape[:2]
    if width is None:
        scale = height / float(current_height)
        width = int(round(current_width * scale))
    if height is None:
        scale = width / float(current_width)
        height = int(round(current_height * scale))
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def save_image(path: str | Path, frame: np.ndarray) -> None:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def read_image_rgb(path: str | Path, image_size: Tuple[int, int] | None = None) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Failed to read image: {path}")
    if image_size is not None:
        width, height = image_size
        bgr = cv2.resize(bgr, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_overlay(frame: np.ndarray, lines: Iterable[str]) -> np.ndarray:
    import cv2

    output = frame.copy()
    y = 28
    for line in lines:
        cv2.putText(output, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return output


def maybe_resize_for_display(
    frame: np.ndarray,
    display_width: Optional[int] = None,
    display_height: Optional[int] = None,
) -> np.ndarray:
    return resize_frame(frame, width=display_width, height=display_height)
