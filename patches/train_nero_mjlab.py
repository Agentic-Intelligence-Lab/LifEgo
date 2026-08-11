#!/usr/bin/env python3
"""mjlab train entrypoint with the LifEgo Nero task registered."""

from __future__ import annotations

import sys
from pathlib import Path


PATCHES_DIR = Path(__file__).resolve().parent
if str(PATCHES_DIR) not in sys.path:
  sys.path.insert(0, str(PATCHES_DIR))

import nero_mjlab_task  # noqa: F401
from mjlab.scripts.train import main


if __name__ == "__main__":
  main()
