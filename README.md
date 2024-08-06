# INFO

This project implements DRL algorithms to gym environment under uncertainity. In real-world scenarios, agents often have to make decisions based on incomplete or noisy information. This project explores how reinforcement learning can be applied to such partially observable environments, specifically using the CartPole problem as a testbed.  

## Key Features

- Implementation of SAC and A2C algorithms with LSTM networks to handle temporal dependencies
- POMDP framework to model uncertainty in observations
- Custom environment wrapper to introduce partial observability to CartPole
- Analysis of agent performance under varying degrees of observation noise

## How it Works

1. The CartPole environment is modified to provide only partial state information (position and angle, omitting velocities)
2. An LSTM-based architecture is used to implicitly model the belief state over time
3. The algorithm is employed to learn a policy that maximizes expected return while accounting for uncertainty.
4. Performance is evaluated based on episode length and cumulative reward, with a focus on robustness to partial observability