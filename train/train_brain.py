import argparse
import math
import os

import gin
import torch
import wandb
from stable_baselines3.common.callbacks import EvalCallback

from coprocessors.configs import coproc_config
from coprocessors.utils.injured_sac import InjuredSAC
from coprocessors.utils.setup_env import make_gym_env
from michaels.experiment.experiment import get_config
from offpcc.algorithms_recurrent import RecurrentSAC
from offpcc.basics import run_fns
from offpcc.basics.replay_buffer_recurrent import RecurrentReplayBuffer


def main(config, project_name, timesteps, wandb_config={}):
    brain_path = config["brain_path"]
    brain_dir = os.path.dirname(brain_path)
    os.makedirs(brain_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    if project_name is None:
        project_name = f"train healthy brain {config['gym_env']}"

    run_name = f"{config['gym_env']} {timesteps} steps"
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=wandb_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    gym_env = make_gym_env(config)
    eval_gym_env = make_gym_env(config)

    if config["brain"] == "sac":
        eval_callback = EvalCallback(
            eval_gym_env,
            best_model_save_path=brain_dir,
            log_path=brain_dir,
            eval_freq=5e4,
            deterministic=True,
            render=False,
        )
        brain = InjuredSAC(
            policy="MlpPolicy",
            env=gym_env,
            verbose=1,
            tensorboard_log=f"runs/sac_brain",
            seed=config["brain_seed"],
            device=device,
        )
        brain.learn(
            callback=eval_callback,
            total_timesteps=timesteps,
        )
        brain.save(brain_path)

    elif config["brain"] == "michaels":
        os.makedirs(brain_path, exist_ok=True)
        gin.parse_config_file("offpcc/configs/test/template_recurrent_500k.gin")
        cfg = get_config(cuda=device)
        
        algorithm = RecurrentSAC(
            input_dim=gym_env.observation_space.shape[0],
            action_dim=gym_env.action_space.shape[0],
            michaels=True,
            cfg=cfg,
        )
        buffer = RecurrentReplayBuffer(
            o_dim=gym_env.observation_space.shape[0],
            a_dim=gym_env.action_space.shape[0],
            max_episode_len=gym_env.spec.max_episode_steps,
        )
        run_fns.train(
            env_fn=lambda: gym_env,
            algorithm=algorithm,
            buffer=buffer,
            num_epochs=math.ceil(timesteps / 1000),
            num_steps_per_epoch=1000,
            save_path=brain_path,
        )

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, help="wandb project name")
    parser.add_argument("--gym_env", type=str, required=True, help="gym environment")
    parser.add_argument("--brain", type=str, required=True, choices={"michaels", "sac"},
                        help="brain environment")
    parser.add_argument("--timesteps", type=int, default=75000, help="number of training timesteps")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("--env_seed", type=int, default=0, help="seed for Gym environments")
    parser.add_argument("--brain_seed", type=int, default=0, help="seed for brain")
    args = parser.parse_args()

    args_dict = vars(args)
    config = coproc_config(**args_dict) 
    main(config, args.project_name, args.timesteps, args_dict)
