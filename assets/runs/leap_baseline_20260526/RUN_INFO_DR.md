# RUN_INFO: LeapCubeReorient with Domain Randomization

**Task**: LeapCubeReorient (Playground built-in + DR)
**GPU**: RTX 5090 32GB (CUDA 13.0)
**Date**: 2026-05-26

## Training Parameters

Same as baseline, plus:
- Domain Randomization: ON (built-in `domain_randomize` function)
- DR includes: contact friction, object mass/size, perturbation forces

## Results

- Initial reward: -9.580
- Final reward: 245.954 (at 194M steps)
- JIT compile time: 79.0s
- Total training time: 1718.0s (~28.6 min)

## Reward Progression

| Timesteps | Reward |
|-----------|--------|
| 0 | -9.580 |
| 10.8M | 134.475 |
| 21.6M | 151.964 |
| 32.4M | 153.912 |
| 43.3M | 153.750 |
| 54.1M | 162.022 |
| 64.9M | 166.180 |
| 75.7M | 156.989 |
| 86.5M | 163.520 |
| 97.3M | 159.351 |
| 108.1M | 150.180 |
| 118.9M | 155.422 |
| 129.8M | 152.727 |
| 140.6M | 161.763 |
| 151.4M | 163.039 |
| 162.2M | 165.878 |
| 173.0M | 195.070 |
| 183.8M | 209.989 |
| 194.6M | 245.954 |
| 205.5M | 237.948 |

## Comparison with Baseline

| | Baseline | DR |
|---|---|---|
| Final reward | 431 | 246 |
| Training time | 28.5 min | 28.6 min |
| Reward at 100M | 203 | 159 |

DR reward is ~43% lower — expected due to randomized physical parameters forcing the policy to generalize.
