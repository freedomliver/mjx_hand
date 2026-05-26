# RUN_INFO: LeapCubeReorient Baseline

**Task**: LeapCubeReorient (Playground built-in)
**GPU**: RTX 5090 32GB (CUDA 13.0)
**Date**: 2026-05-26

## Training Parameters

- num_envs: 4096
- num_timesteps: 200,000,000
- impl: jax (MJX, not Warp)
- Domain Randomization: OFF
- Policy network: (512, 256, 128)
- Value network: (512, 256, 128)
- Discounting: 0.99
- Learning rate: 3e-4
- Entropy cost: 1e-2
- Unroll length: 40
- Num minibatches: 32
- Num updates per batch: 4

## Results

- Initial reward: -8.982
- Final reward: 431.444 (at 194M steps)
- JIT compile time: 67.2s
- Total training time: 1712.6s (~28.5 min)
- GPU memory: ~3.7 GB (peak during training)

## Reward Progression

| Timesteps | Reward |
|-----------|--------|
| 0 | -8.982 |
| 10.8M | 134.965 |
| 21.6M | 153.747 |
| 32.4M | 158.695 |
| 43.3M | 157.433 |
| 54.1M | 167.367 |
| 64.9M | 171.829 |
| 75.7M | 152.602 |
| 86.5M | 171.268 |
| 97.3M | 203.037 |
| 108.1M | 203.059 |
| 118.9M | 224.717 |
| 129.8M | 251.151 |
| 140.6M | 280.847 |
| 151.4M | 305.684 |
| 162.2M | 308.394 |
| 173.0M | 351.090 |
| 183.8M | 397.388 |
| 194.6M | 431.444 |
| 205.5M | 408.285 |

## Checkpoint Location (remote)

`/root/autodl-tmp/logs/LeapCubeReorient-20260526-063532/checkpoints/`

## Video Location

`assets/videos/hand/leap_reorient_baseline_*.mp4`
