import argparse
import os

import numpy as np
import stable_baselines3
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from coprocessors.configs import coproc_config
from coprocessors.utils.setup_env import setup_coproc_env


class Callback(BaseCallback):
    '''
    Snippet skeleton from Stable baselines3 documentation here:
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
    '''

    def __init__(self, max_episodes, test_env):
        super(Callback, self).__init__()
        self.total_reward = 0
        self.steps = 0
        self.episodes = 0
        self.max_episodes = max_episodes
        self.test_env = test_env

        wandb.define_metric("global_episode")
        wandb.define_metric("rollout/ep_rew_mean", step_metric="global_episode")

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.total_reward += reward
        if self.steps == 0:
            test_reward = self.test_policy()
            wandb.log({"test/ep_rew_mean": test_reward, "global_episode": self.episodes})
            wandb.log({"rollout/ep_rew_mean": test_reward, "global_episode": self.episodes})

        if self.locals['dones']:
            self.episodes += 1
            test_reward = self.test_policy()
            wandb.log({"rollout/ep_rew_mean": self.total_reward, "global_episode": self.episodes})
            wandb.log({"test/ep_rew_mean": test_reward, "global_episode": self.episodes})
            self.total_reward = 0

        self.steps += 1
        if self.episodes == self.max_episodes:
            return False

    def test_policy(self):
        episode_rewards, _ = evaluate_policy(self.model, self.test_env, n_eval_episodes=25)
        return np.mean(episode_rewards)


def main(config, project_name, num_episodes, wandb_config={}):
    model_path = config["coproc_path"]
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    if project_name is None:
        project_name = f"train sac recovery {config['gym_env']}"

    run_name = "sac_recovery_lesion-{}_stim_dim-{}".format(config["pct_lesion"], config["stim_dim"])
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=wandb_config,
        monitor_gym=True,
        save_code=True,
    )

    coproc_env = setup_coproc_env(config, device)
    coproc_env_test = setup_coproc_env(config, device)
    model = stable_baselines3.SAC(
        'MlpPolicy',
        coproc_env,
        verbose=1,
        tensorboard_log=f"runs/sac_recovery",
        device=device,
    )
    episode_steps = config["max_episode_steps"] if config["max_episode_steps"] else 1000
    model.learn(
        total_timesteps=num_episodes * episode_steps,
        callback=Callback(num_episodes, coproc_env_test),
    )
    model.save(model_path)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, help="wandb project name")
    parser.add_argument("--gym_env", type=str, required=True, help="gym environment")
    parser.add_argument("--method", type=str, required=True, help="method for wandb")
    parser.add_argument("--brain", type=str, required=True, choices={"michaels", "sac"},
                        help="brain environment")
    parser.add_argument("--pct_lesion", type=float, required=True, help="percent to injure brain")
    parser.add_argument("--region", type=str, default=None, choices=("M1", "F5", "AIP"),
                        help="region of brain to injure (Michaels)")
    parser.add_argument("--stim_dim", type=int, required=True, help="stimulation dimension")
    parser.add_argument("--brain_obs", action="store_true", help="include brain observations")
    parser.add_argument("--temporal", action="store_true", help="apply temporal stimulation (Michaels)")
    parser.add_argument("--data_size", type=int, default=10, help='number of training episodes')
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("--max_episode_steps", default=None, type=int, help="maximum episode length")
    parser.add_argument("--env_seed", type=int,default=0, help="seed for Gym environments")
    parser.add_argument("--brain_seed", type=int, default=0, help="seed for brain")
    parser.add_argument("--injury_seed", type=int, default=0, help="seed for injury selection")
    parser.add_argument("--stim_seed", type=int, default=0, help="seed for stimulation selection")
    args = parser.parse_args()

    args_dict = vars(args)
    config = coproc_config(coproc="sac", **args_dict)
    main(config, args.project_name, args.data_size, args_dict)
