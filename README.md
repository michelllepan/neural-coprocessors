# Neural Coprocessors

Accompanying code for the paper [Coprocessor Actor Critic: A Model-Based Reinforcement Learning Approach For Adaptive Brain Stimulation](https://arxiv.org/abs/2406.06714).

## Environment setup

Use the following commands to create a Conda environment with the required packages:
``` bash
PIP_NO_DEPS=1 conda env create -f environment.yml
conda activate coproc
pip install -e .
pip install -e myosuite
```

## Train healthy brain policies

To train a healthy brain policy, run `train/train_brain.py`.
```bash
python train/train_brain.py --gym_env myoHandReachFixed-v0 --brain michaels --timesteps 500000
```
In our paper, we use Michaels model brains for MyoSuite environments and SAC for other environments.

## Train coprocessor policies

#### CopAC
To train a coprocessor using CopAC, first train an optimal policy with `train/train_sac_env.py`.
```bash
python train/train_sac_env.py --gym_env myoHandReachFixed-v0 --timesteps 5000000
```
Then run `train/train_action_conv.py`.
```bash
python train/train_action_conv.py --gym_env myoHandReachFixed-v0 --brain michaels --region M1 --pct_lesion 0.9 --stim_dim 2 -action_conv qmax --num_q_update_traj 5
```
To run without Q-updates, run with `--num_q_update_traj 0`. To run without both Q-updates and Q-max, use `--action_conv random`.

#### SAC
The following command trains a SAC coprocessor:
```bash
python train/train_sac_coproc.py --gym_env myoHandReachFixed-v0 --brain michaels --region M1 --pct_lesion 0.9 --stim_dim 2
```

#### MBPO
Due to dependency incompatibilities, MBPO must be run in a separate environment. First set up the environment:
```bash
conda create -n coproc-mbrl python=3.9
conda activate coproc-mbrl
sh install_mbrl_deps.sh
```
Then train the MBPO coprocessor:
```bash
python -m mbrl-lib.mbrl.examples.main algorithm=mbpo overrides=mbpo_coproc_myo-hand overrides.env_cfg.pct_lesion=0.9 overrides.env_cfg.stim_dim=2
```

#### Offline
To learn the optimal policy from offline data, first collect healthy brain rollouts.
```bash
python coprocessors/utils/collect_offline_data.py --gym_env myoHandReachFixed-v0 --brain michaels --data_size 1000 
```
Then train the policy.
```bash
python train/train_offline.py --env myoHandReachFixed-v0 --data_size 1000 --episodes 5000
```
Finally, run CoPAC with the offline policy.
```bash
python train/train_action_conv.py --gym_env myoHandReachFixed-v0 --brain michaels --region M1 --pct_lesion 0.9 --stim_dim 2 --action_conv qmax_offline
```