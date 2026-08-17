"""OpenPI data transforms for Nero EEF action-space training."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


ACTION_DIM = 8


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(255.0 * image, 0.0, 255.0).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class NeroEefInputs(transforms.DataTransformFn):
    """Map LeRobot Nero EEF samples into OpenPI observations."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != ACTION_DIM:
            raise ValueError(f"Nero EEF state must be {ACTION_DIM}D, got {state.shape}")

        if self.model_type == _model.ModelType.PI0_FAST:
            images = {
                "base_0_rgb": base_image,
                "base_1_rgb": np.zeros_like(base_image),
                "wrist_0_rgb": np.zeros_like(base_image),
            }
            image_mask = {
                "base_0_rgb": np.True_,
                "base_1_rgb": np.False_,
                "wrist_0_rgb": np.False_,
            }
        else:
            images = {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            }
            image_mask = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            }

        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"Nero EEF actions must be {ACTION_DIM}D, got {actions.shape}")
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class NeroEefOutputs(transforms.DataTransformFn):
    """Return only the Nero EEF action dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM], dtype=np.float32)}

