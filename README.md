# Dexterous In-Hand Manipulation — Project B

> MuJoCo Playground (MJX + JAX) · PPO · Domain Randomization · LEAP Hand

## Overview

Train a dexterous manipulation policy for the LEAP Hand to reorient a cube in-hand to match target orientations. Built with MuJoCo Playground (MJX + JAX), demonstrating contact-rich RL on a different simulation stack than Project A (Isaac Lab).

**Hardware**: AutoDL RTX 5090 32GB | **Framework**: Brax PPO + MuJoCo MJX

---

## Method

### 1. Baseline (LeapCubeReorient)

Standard PPO training on the Playground built-in `LeapCubeReorient` task.

- 24-DOF LEAP Hand, cube reorientation to random target orientations
- Success threshold: orientation error < 0.1 rad
- num_envs=8192, num_timesteps=200M
- Network: policy (512, 256, 128), value (512, 256, 128)

| Metric | Value |
|---|---|
| Training time | TBD |
| Converged reward | TBD |
| Success rate | TBD |

### 2. Domain Randomization (Built-in)

Playground provides a built-in `domain_randomize` function for LeapCubeReorient:

| DR Item | Description |
|---|---|
| Contact friction | Randomized across episodes |
| Object properties | Mass/size perturbation |
| Perturbation forces | Configurable velocity perturbations on the cube |

### 3. Reward Structure

See [notes/leap_hand_reward.md](notes/leap_hand_reward.md) for detailed breakdown.

Key reward terms:
- `orientation` (5.0): primary task objective
- `success_reward` (100.0): bonus on reaching target
- `termination` (-100.0): penalty for dropping cube
- `hand_pose` (-0.5): encourages efficient hand configurations

---

## Results

### Training Curves

TBD — will be updated after training completes.

### Demo Videos

TBD — will be added after recording.

---

## Key Engineering Decisions

1. **MuJoCo over Isaac Sim**: Contact-rich manipulation tasks benefit from MuJoCo's soft contact model; avoids the PhysX GPU kernel bugs encountered in Project A
2. **JAX 0.10 + brax compatibility**: Patched `jax.device_put_replicated` removal in JAX 0.10 via site-packages monkey-patch (see [docs/B1_install_debug.md](docs/B1_install_debug.md))
3. **Stack diversity for resume**: Demonstrating RL competence across both Isaac Lab (NVIDIA) and MuJoCo Playground (Google DeepMind) stacks

---

## Reproduce

```bash
# Clone and install
git clone https://github.com/google-deepmind/mujoco_playground.git
cd mujoco_playground
python -m venv .venv && source .venv/bin/activate
pip install "jax[cuda12]" --index-url https://pypi.org/simple
pip install -e .

# Train baseline
train-jax-ppo \
    --env_name LeapCubeReorient \
    --num_envs 8192 \
    --num_timesteps 200000000 \
    --domain_randomization \
    --logdir logs
```

---

## File Structure

```
mjx_hand/
├── README.md                    # This file
├── scripts/
│   ├── b0_setup.sh              # Environment setup
│   ├── b1_train_baseline.sh     # Training script
│   └── fetch_videos_b.sh        # Pull videos from remote
├── docs/
│   └── B1_install_debug.md      # Installation troubleshooting
├── notes/
│   └── leap_hand_reward.md      # Reward function analysis
└── assets/
    ├── videos/hand/             # Demo videos
    └── runs/                    # Training logs + checkpoints
```
