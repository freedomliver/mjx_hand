# LeapCubeReorient Task: Reward Structure

## Overview

The LeapCubeReorient task trains a dexterous manipulation policy to reorient a cube in-hand to match target orientations. The agent controls the 24-DOF LEAP Hand to manipulate the cube. Success is achieved when the orientation error falls below 0.1 radians. Each episode is 1000 steps, and upon success, a new random goal orientation is sampled for continued learning.

## Reward Components

| Reward Term | Weight | Description | Purpose |
|-------------|--------|-------------|---------|
| orientation | 5.0 | Negative orientation error between cube and target | Primary task objective: incentivizes accurate orientation matching |
| position | 0.5 | Negative position error of cube center | Maintains cube position stability during reorientation |
| success_reward | 100.0 | Bonus reward on task completion | Strong signal for successful episode completion |
| termination | -100.0 | Penalty when episode is terminated | Discourages early termination or failure states |
| hand_pose | -0.5 | Penalty for deviating from default hand pose | Encourages energy-efficient configurations |
| action_rate | -0.001 | Penalty proportional to control action magnitude | Smooths trajectory and reduces unnecessary motion |
| energy | -0.001 | Penalty for joint velocity magnitudes | Minimizes energy consumption and joint stress |
| joint_vel | 0.0 | Penalty for joint velocities (disabled) | Currently unused in reward calculation |

## Training Dynamics

- Primary focus: Orientation accuracy (weight: 5.0)
- Secondary stability: Position control (weight: 0.5)
- Efficiency penalties: Hand pose, action smoothing, energy (combined weight: ~0.502)
- Task completion: Strong success bonus (100.0) with heavy termination penalty (-100.0)

## Success Criteria

- Orientation error < 0.1 radians
- Episode length: Up to 1000 steps
- Continuous goal resampling after each success
