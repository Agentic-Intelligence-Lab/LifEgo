from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class FrameTimestamp:
    timestamp_unix: float
    timestamp_monotonic: float
    camera_timestamp_unix: float | None = None
    camera_timestamp_msec: float | None = None

    @classmethod
    def now(
        cls,
        camera_timestamp_unix: float | None = None,
        camera_timestamp_msec: float | None = None,
    ) -> "FrameTimestamp":
        return cls(
            timestamp_unix=time.time(),
            timestamp_monotonic=time.monotonic(),
            camera_timestamp_unix=camera_timestamp_unix,
            camera_timestamp_msec=camera_timestamp_msec,
        )
