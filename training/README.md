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
outputs/openpi_assets/nero_eef/local/nero_ego_ymq_eef/norm_stats.json
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
  --num-train-steps 30000 \
  --save-interval 1000 \
  --loss-action-dim 8
```

This expects:

```text
/mnt/data/szeluresearch/models/pi05_base/model.safetensors
```

OpenPI's PyTorch trainer currently runs full fine-tuning. LoRA is used only by
the JAX entry point below. The Pi0.5 model keeps the base 32D action head; Nero
EEF data occupies the first 8 dimensions and OpenPI pads the rest. The local
PyTorch entry point keeps the 32D shape for checkpoint compatibility, but its
default `--loss-action-dim 8` setting crops the returned loss tensor so only the
real EEF dimensions are supervised.

## Train Pi0.5 With JAX

Use this only if you have the JAX `params/` checkpoint:

```bash
cd thirdparty/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python ../../training/train_nero_eef.py \
  --model pi05 \
  --exp-name nero_eef_pi05_lora_v1 \
  --batch-size 16 \
  --num-workers 2 \
  --num-train-steps 30000 \
  --save-interval 1000 \
  --overwrite
```

The JAX checkpoint path is:

```text
$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params
```

## Train Pi0-FAST

Pi0-FAST is still available with:

```bash
cd thirdparty/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python ../../training/train_nero_eef.py \
  --model pi0_fast \
  --exp-name nero_eef_pi0_fast_lora_v1 \
  --batch-size 16 \
  --num-workers 2 \
  --num-train-steps 30000 \
  --save-interval 1000 \
  --overwrite
```

The Pi0-FAST config uses:

```text
action_dim = 8
action_horizon = 10
max_token_len = 180
```

Checkpoints are written under:

```text
outputs/openpi_checkpoints/nero_eef/<exp-name>
```
