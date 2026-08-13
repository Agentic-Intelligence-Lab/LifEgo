#!/usr/bin/env python3
"""mjlab train entrypoint with the LifEgo Nero task registered."""

from __future__ import annotations

import sys
from pathlib import Path


NEW_DIR = Path(__file__).resolve().parent
if str(NEW_DIR) not in sys.path:
  sys.path.insert(0, str(NEW_DIR))

import rl_env_mjlab  # noqa: F401
from mjlab.scripts.train import main


if __name__ == "__main__":
  main()
