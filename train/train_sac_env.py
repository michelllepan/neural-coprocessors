import argparse
import os

import stable_baselines3
import torch
import wandb

from coprocessors.configs import coproc_config
from coprocessors.utils.setup_env import make_gym_env
from myosuite import myosuite


def main(config, project_name, timesteps, wandb_config={}):
    save_path = config["env_actor_critic_path"]
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    if project_name is None:
        project_name = "train SAC env"

    wandb.init(
        project=project_name,
        name=config["gym_env"],
        config=wandb_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,       # auto-upload the videos of agents playing the game
    )

    gym_env = make_gym_env(config)
    model = stable_baselines3.SAC(
        "MlpPolicy",
        gym_env,
        verbose=1,
        tensorboard_log=f"runs/myo",
        device=device,
    )

    model.learn(total_timesteps=timesteps)
    model.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, help="wandb project name")
    parser.add_argument("--gym_env", type=str, required=True, help="gym environment")
    parser.add_argument('--timesteps', type=int, help="number of env timesteps to train for")
    args = parser.parse_args()

    args_dict = vars(args)
    config = coproc_config(brain="michaels", **args_dict)
    main(config, args.project_name, args.timesteps, args_dict)