# Nero EEF OpenPI Training

This directory keeps LifEgo-specific OpenPI training code outside
`thirdparty/openpi`.

## Dataset

The first training target is ego RGB to robot-base EEF action:

```text
state/action = [x, y, z, qx, qy, qz, qw, gripper]
```

The current dataset is:

```text
outputs/lerobot/local/nero_ego_stack_object_horizontal_eef
```

It contains 30 episodes and 5950 frames from `ymq`, `xule`, and `hyj`.

Only ego RGB is used:

```text
base_0_rgb = ego RGB, mask true
other image inputs = zeros, mask false
```

Actions are absolute next-frame EEF targets. No delta transform is applied,
because quaternion deltas should not be represented by plain subtraction.

The task prompt is:

```text
Place the black pillar in the plate.
```

## Check Data

Run from the OpenPI uv environment:

```bash
cd thirdparty/openpi

uv run python ../../training/check_nero_eef_dataloader.py \
  --skip-norm-stats \
  --batch-size 2 \
  --num-workers 0
```

## Norm Stats

```bash
cd thirdparty/openpi

uv run python ../../training/compute_norm_stats.py \
  --batch-size 32 \
  --num-workers 0
```

This writes:

```text
outputs/openpi_assets/nero_eef/local/nero_ego_stack_object_horizontal_eef/norm_stats.json
```

## Train Pi0.5 With PyTorch

Use this for the `model.safetensors` Pi0.5 checkpoint:

```bash
cd thirdparty/openpi

rm -r ../../outputs/openpi_checkpoints/nero_eef/nero_eef_pi05_pytorch_v1 2>/dev/null || true

uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  ../../training/train_nero_eef_pytorch.py \
  --model pi05 \
  --pytorch-weight-path /mnt/data/szeluresearch/models/pi05_base \
  --exp-name nero_eef_pi05_pytorch_v1 \
  --batch-size 8 \
  --num-workers 2 \
  --num-train-steps 1486 \
  --save-interval 1000 \
  --loss-action-dim 8
```

For the current dataset, two epochs with PyTorch global batch size 8 are:

```text
2 * floor(5950 frames / 8) = 1486 train steps
```

If `--num-train-steps` is omitted, the local training entry point computes this
two-epoch value from the selected global batch size.

This expects:

```text
/mnt/data/szeluresearch/models/pi05_base/model.safetensors
```

OpenPI's PyTorch trainer currently runs full fine-tuning. The Pi0.5 model keeps
the base 32D action head; Nero EEF data occupies the first 8 dimensions and
OpenPI pads the rest. The local PyTorch entry point keeps the 32D shape for
checkpoint compatibility, but its default `--loss-action-dim 8` setting crops
the returned loss tensor so only the real EEF dimensions are supervised.

Checkpoints are written under:

```text
outputs/openpi_checkpoints/nero_eef/<exp-name>
```
