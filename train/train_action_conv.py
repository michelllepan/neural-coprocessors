import argparse
import os

import torch
import wandb

from coprocessors.configs import coproc_config
from coprocessors.utils.setup_action_conv import setup_action_conv


def main(config, project_name, wandb_config={}):
    model_dir = config["action_conv_dir"]
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    if project_name is None:
        project_name = f"train action converter {config['gym_env']}"

    wandb.init(
        project=project_name,
        name="{}_pct-lesion-{}_data-size-{}_num-update-{}".format(
            config["action_conv"],
            config["pct_lesion"],
            config["data_size"],
            config["num_q_update_traj"],
        ),
        config=wandb_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    wandb.define_metric("global_episode")
    wandb.define_metric("rollout/ep_rew_mean", step_metric="global_episode")
    wandb.define_metric("test/ep_rew_mean", step_metric="global_episode")

    action_converter_policy = setup_action_conv(config, device)
    action_converter_policy.collect_and_train(
        num_episodes=config["data_size"],
        num_q_update_traj=config["num_q_update_traj"],
        test_policy=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, help="wandb project name")
    parser.add_argument("--gym_env", type=str, required=True, help="gym environment")
    parser.add_argument("--method", type=str, required=True, help="method for wandb")
    parser.add_argument("--brain", type=str, required=True, choices={"michaels", "sac"}, 
                        help="brain environment")
    parser.add_argument("--pct_lesion", type=float, default=0.0, help="percent to injure brain")
    parser.add_argument("--region", type=str, default=None, choices=("M1", "F5", "AIP"),
                        help="region of brain to injure (Michaels)")
    parser.add_argument("--stim_dim", type=int, default=0, help="stimulation dimension")
    parser.add_argument("--brain_obs", action="store_true", help="include brain observations")
    parser.add_argument("--temporal", action="store_true",
                        help="apply temporal stimulation (Michaels)")
    parser.add_argument("--action_conv", type=str, default="qmax", help="action converter type",
                        choices={"random", "qmax", "qmax_offline", "inverse"})
    parser.add_argument("--data_size", type=int, default=10,
                        help="number of episodes to collect for training action converter")
    parser.add_argument("--opt_method", type=str, default="grid_search",
                        help="method for optimizing objective",
                        choices={"lbfgs", "sim_anneal", "grid_search"})
    parser.add_argument("--num_q_update_traj", type=int, default=0,
                        help="number of Q-update episodes for each action converter training episode")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for Q-update")
    parser.add_argument("--q_lr", type=float, default=1e-4, help="learning rate for Q-update")
    parser.add_argument("--tau", type=float, default=1e-3, help="soft update coefficient for Q-update")
    parser.add_argument("--train_frequency", type=int, default=10,
                        help="step frequency for retraining f network")
    parser.add_argument("--update_start", type=int, default=3,
                        help="number of steps to retrain f network")
    parser.add_argument("--f_lr", type=float, default=5e-3, help="learning rate for training f network")
    parser.add_argument("--max_episode_steps", default=None, type=int, help="maximum episode length")
    parser.add_argument("--env_seed", type=int,default=0, help="seed for Gym environments")
    parser.add_argument("--brain_seed", type=int,default=0, help="seed for brain")
    parser.add_argument("--injury_seed", type=int, default=0, help="seed for injury selection")
    parser.add_argument("--stim_seed", type=int, default=0, help="seed for stimulation selection")
    parser.add_argument("--action_conv_seed", type=int, default=0, help="seed for action converter initialization")
    args = parser.parse_args()

    args_dict = vars(args)
    config = coproc_config(**args_dict) 
    main(config, args.project_name, args_dict)
    