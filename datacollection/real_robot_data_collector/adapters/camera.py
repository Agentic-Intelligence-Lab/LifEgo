from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CameraReadResult:
    ok: bool
    frame: np.ndarray | None
    timestamp_unix: float | None
    capture_msec: float | None


class CameraStream:
    def __init__(self, camera_id: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.camera_id = int(camera_id)
        self.width = width
        self.height = height
        self._cap = None

    def open(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self.camera_id)
        if self.width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"Failed to open camera id {self.camera_id}. Check camera permissions and device index.")

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> CameraReadResult:
        import cv2
        import time

        if self._cap is None:
            raise RuntimeError("CameraStream.read() called before open()")
        ok, frame = self._cap.read()
        timestamp_unix = time.time()
        capture_msec = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        if not ok or frame is None:
            return CameraReadResult(False, None, timestamp_unix, capture_msec)
        return CameraReadResult(True, frame, timestamp_unix, capture_msec)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
