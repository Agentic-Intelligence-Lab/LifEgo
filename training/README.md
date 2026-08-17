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
outputs/lerobot/local/nero_ego_ymq_eef
```

It contains 10 episodes and 2017 frames.

Only ego RGB is used:

```text
base_0_rgb = ego RGB, mask true
other image inputs = zeros, mask false
```

Actions are absolute next-frame EEF targets. No delta transform is applied,
because quaternion deltas should not be represented by plain subtraction.

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
outputs/openpi_assets/nero_eef/local/nero_ego_ymq_eef/norm_stats.json
```

## Train

Low-memory LoRA fine-tuning is the default:

```bash
cd thirdparty/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python ../../training/train_nero_eef.py \
  --exp-name nero_eef_lora_v1 \
  --batch-size 16 \
  --num-workers 2 \
  --num-train-steps 30000 \
  --save-interval 1000 \
  --overwrite
```

The default model is `pi0_fast` with:

```text
action_dim = 8
action_horizon = 10
max_token_len = 180
```

Checkpoints are written under:

```text
outputs/openpi_checkpoints/nero_eef/<exp-name>
```

