import argparse
import os
import pickle
from tqdm import tqdm

import torch

from coprocessors.configs import coproc_config
from coprocessors.utils.setup_policy import setup_healthy_brain_policy


def main(config, data_size):
    data_path = config["offline_data_path"]
    data_dir = os.path.dirname(data_path)
    os.makedirs(data_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    brain, env, reset_policy = setup_healthy_brain_policy(config, device)

    data = {
        "observations": [],
        "next_observations": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "timeouts": [],
        "infos": [],
    }

    for ep in tqdm(range(args.data_size)):
        obs, _ = env.reset(seed=ep+1)
        if reset_policy:
            brain.reset()

        for i in range(env.spec.max_episode_steps):
            act = brain.act(obs, deterministic=True).reshape(-1,)
            next_obs, reward, truncated, terminated, info = env.step(act)

            data["observations"].append(obs)
            data["next_observations"].append(next_obs)
            data["actions"].append(act)
            data["rewards"].append(reward)
            data["terminals"].append(terminated)
            data["timeouts"].append(truncated)
            data["infos"].append(info)

            obs = next_obs
            done = truncated or terminated
            if done:
                break

    with open(data_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym_env", type=str, help="gym environment", required=True)
    parser.add_argument("--brain", type=str, help="brain environment", required=True, 
                        choices={"michaels", "sac"})
    parser.add_argument("--data_size", type=int, default=10, 
                        help="number of episodes to collect data for")
    parser.add_argument("--brain_seed", type=int, default=0, help="seed for brain")
    args = parser.parse_args()

    args_dict = vars(args)
    config = coproc_config(**args_dict) 
    main(config, args.data_size)