# Nero EEF Inference

This folder contains deployment-time code only. Model inference runs in a
dedicated server process. The robot client captures ego RGB, reads the current
Nero TCP state, requests an EEF action chunk from the server, and sends `move_p`
commands to the real arm.

## Server

Run this on the machine that has the OpenPI checkpoint and GPU:

```bash
cd thirdparty/openpi

uv run python ../../inference/nero_eef_policy_server.py \
  --exp-name nero_eef_pi05_pytorch_v1 \
  --step 10000 \
  --host 0.0.0.0 \
  --port 8000
```

Dry-load the checkpoint and run one dummy inference:

```bash
uv run python ../../inference/nero_eef_policy_server.py \
  --exp-name nero_eef_pi05_pytorch_v1 \
  --step 10000 \
  --validate-only
```

The server receives:

- `image`: RGB `uint8`, shape `224x224x3`
- `state`: 8D EEF state `[x, y, z, qx, qy, qz, qw, gripper]`

It returns unnormalized EEF actions with shape `10x8`.

## Client

Run this on the machine connected to the ego camera and Nero arm:

```bash
cd thirdparty/openpi

uv run python ../../inference/nero_eef_remote_client.py \
  --host <server_ip> \
  --port 8000 \
  --camera-index 0
```

The command above is dry-run: it requests one action chunk and prints the target
EEF poses. To command the real robot:

```bash
uv run python ../../inference/nero_eef_remote_client.py \
  --host <server_ip> \
  --port 8000 \
  --camera-index 0 \
  --channel can1 \
  --steps 60 \
  --execute
```

The client is safe by default. `--execute` still asks for `DEPLOY` before moving
the robot unless `--yes` is also passed. Before policy control starts, the arm
moves to a standby flange pose read from:

```text
DATA/20260816_nero_stack_object_horizontal/stack_object_20260816_163253_653146.jsonl
```

Camera capture currently uses OpenCV `VideoCapture`, not the RealSense SDK. The
default capture request is `1280x720@30`, then each frame is resized to
`224x224` RGB to match the training input.

## Test Client

Use this to verify the server and camera pipeline without connecting to the real
robot:

```bash
cd thirdparty/openpi

uv run python ../../inference/nero_eef_test_client.py \
  --host <server_ip> \
  --port 8000 \
  --camera-index 0 \
  --steps 20
```

Optionally save request records and RGB inputs:

```bash
uv run python ../../inference/nero_eef_test_client.py \
  --host <server_ip> \
  --port 8000 \
  --camera-index 0 \
  --steps 20 \
  --output-dir ../../outputs/openpi_inference_test \
  --save-images
```

The test client uses the standby EEF state from the realbot JSONL and only
checks websocket inference round trips. It never imports or commands the robot
runtime.
