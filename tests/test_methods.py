import argparse

import torch
import wandb

from coprocessors.configs import coproc_config
from coprocessors.utils.setup_policy import setup_policy
from tests.basics import run_policy


def main(config, project_name, num_tests, render, camera_id, wandb_config={}):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    if project_name is None:
        project_name = f"test recovery methods {config['gym_env']}"

    if config["coproc"] is None:
        if config["pct_lesion"] > 0:
            run_name = f"injured_brain_lesion-{config['pct_lesion']}"
        else:
            run_name = "healthy_brain"
    elif config["coproc"] == "sac":
        run_name= "sac_recovery_lesion-{}_stim-dim-{}".format(config["pct_lesion"], config["stim_dim"])
    elif config["coproc"] in ("pmax", "qmax"):
        run_name = "{}_lesion-{}_stim-dim{}_action-conv-{}_data-size-{}".format(
            config["coproc"], config["pct_lesion"], config["stim_dim"], config["action_conv"], config["data_size"]
        )
    elif config["coproc"] == "random":
        run_name = "random_lesion-{}_stim-dim-{}".format(config["pct_lesion"], config["stim_dim"])

    run = wandb.init(
        project=project_name,
        name=run_name,
        config=wandb_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    policy, test_env, reset_policy = setup_policy(config, device)
    run_policy(
        policy=policy,
        env=test_env,
        reset_policy=reset_policy,
        num_tests=num_tests,
        render=render,
        camera_id=camera_id,
        seed=config["env_seed"],
        return_std=True,
        device=device,
    )

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, help="wandb project name")
    parser.add_argument("--gym_env", type=str, required=True, help="gym environment")
    parser.add_argument("--brain", type=str, required=True, choices={"michaels", "sac"},
                        help="brain environment")
    parser.add_argument("--pct_lesion", type=float, default=0, help="percent to injure brain")
    parser.add_argument("--region", type=str, default=None, choices=("M1", "F5", "AIP"),
                        help="region of brain to injure (Michaels)")
    parser.add_argument("--coproc", type=str, default=None, choices={"sac", "pmax", "qmax", "random"},
                        help="coprocessor type")
    parser.add_argument("--stim_dim", type=int, default=2, help="stimulation dimension")
    parser.add_argument("--brain_obs", action="store_true", help="include brain observations")
    parser.add_argument("--temporal", action="store_true", help="apply temporal stimulation (Michaels)")
    parser.add_argument("--action_conv", type=str, default="true_f",
                        choices={"true_f", "random", "qmax", "inverse"}, help="action converter type")
    parser.add_argument("--data_size", type=int, default=0,
                        help="number of episodes to collect for training action converter")
    parser.add_argument("--opt_method", type=str, default="sim_anneal",
                        help="optimization used for action converter")
    parser.add_argument("--num_q_update_traj", type=int, default=0,
                        help="number of Q-update episodes for each action converter training episode")
    parser.add_argument("--train_frequency", type=int, default=10,
                        help="step frequency for retraining f network")
    parser.add_argument("--update_start", type=int, default=3,
                        help="number of steps to retrain f network")
    parser.add_argument("--num_tests", type=int, default=50, help="number of episodes to test")
    parser.add_argument("--disc", type=int, default=51, help="discretization for grid search")
    parser.add_argument("--render", action="store_true", help="render environment and save frames")
    parser.add_argument("--camera_id", type=int, default=0, help="camera id for Myosuite rendering")
    parser.add_argument("--max_episode_steps", default=None, type=int, help="maximum episode length")
    parser.add_argument("--env_seed", type=int,default=0, help="seed for Gym environments")
    parser.add_argument("--brain_seed", type=int, default=0, help="seed for brain")
    parser.add_argument("--injury_seed", type=int, default=0, help="seed for injury selection")
    parser.add_argument("--stim_seed", type=int, default=0, help="seed for stimulation selection")
    args = parser.parse_args()

    args_dict = vars(args)
    config=coproc_config(**args_dict)
    main(config, args.project_name, args.num_tests, args.render, args.camera_id, args_dict)