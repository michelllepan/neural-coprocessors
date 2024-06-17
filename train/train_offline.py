# adapted from https://github.com/BY571/CQL/blob/main/CQL-SAC/train_offline.py

import argparse
import glob
import pickle
import random
from collections import deque

import gym
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset

from coprocessors.configs import coproc_config
from coprocessors.utils.setup_env import make_gym_env
from CQL_SAC.agent import CQLSAC
from CQL_SAC.utils import save


def prep_dataloader(config, batch_size=256):
    dataset = {
        "observations": [],
        "next_observations": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "timeouts": [],
        "infos": [],
    }

    file_path = config["offline_data_path"]
    with open(file_path, "rb") as handle:
        b = pickle.load(handle)
        for key in dataset.keys():
            dataset[key] += b[key]

    tensors = {}
    for k, v in dataset.items():
        if k in ("actions", "observations", "next_observations", "rewards"):
            tensors[k] = torch.tensor(v).float()
        elif k == "terminals":
            tensors[k] = torch.tensor(v).long()

    tensor_data = TensorDataset(
        tensors["observations"],
        tensors["actions"],
        tensors["rewards"][:, None],
        tensors["next_observations"],
        tensors["terminals"][:, None],
    )
    dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True)

    eval_env = make_gym_env(config)
    return dataloader, eval_env


def scale_world_action(action, world_env):
    min_action = -1
    max_action = 1
    return min_action + (max_action - min_action) * (
        action - world_env.action_space.low
    ) / (world_env.action_space.high - world_env.action_space.low)


def rescale_world_action(action, world_env):
    min_action = -1
    max_action = 1
    action = world_env.action_space.low + (
        world_env.action_space.high - world_env.action_space.low
    ) * ((action - min_action) / (max_action - min_action))
    return action


def evaluate(env, policy, eval_runs=5):
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    for i in range(eval_runs):
        obs, _ = env.reset()
        rewards = 0
        while True:
            action = policy.get_action(obs, eval=True)
            obs, reward, truncated, terminated, _ = env.step(
                rescale_world_action(action, env)
            )
            rewards += reward

            done = terminated or truncated
            if done:
                break
        reward_batch.append(rewards)

    return np.mean(reward_batch)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="CQL",
                        help="Run name, default: CQL")
    parser.add_argument("--env", type=str, default="halfcheetah-medium-v2",
                        help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--data_size", type=int, default=1000,
                        help="Size of offline dataset")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0,
                        help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="")
    parser.add_argument("--target_action_gap", type=float, default=10, help="")
    parser.add_argument("--with_lagrange", type=int, default=0, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    
    args = parser.parse_args()

    config = coproc_config(
        gym_env=args.env,
        brain="michaels" if "myo" in args.env else "sac",
        data_size=args.data_size,
        brain_seed=0,
    )

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataloader, env = prep_dataloader(config=config, batch_size=args.batch_size)

    env.action_space.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batches = 0
    average10 = deque(maxlen=10)

    with wandb.init(project="CQL offline", name=args.run_name, config=vars(args)):
        agent = CQLSAC(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.shape[0],
            tau=args.tau,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            temp=args.temperature,
            with_lagrange=args.with_lagrange,
            cql_weight=args.cql_weight,
            target_action_gap=args.target_action_gap,
            device=device,
        )

        wandb.watch(agent, log="gradients", log_freq=10)
        if args.log_video:
            env = gym.wrappers.Monitor(env, "./video", video_callable=lambda x: x % 10 == 0, force=True)

        eval_reward = evaluate(env, agent)
        wandb.log({"Test Reward": eval_reward, "Episode": 0, "Batches": batches}, step=batches)

        for i in range(1, args.episodes + 1):

            for batch_idx, experience in enumerate(dataloader):
                states, actions, rewards, next_states, dones = experience
                states = states.to(device)
                actions = scale_world_action(actions, env)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                (
                    policy_loss,
                    alpha_loss,
                    bellmann_error1,
                    bellmann_error2,
                    cql1_loss,
                    cql2_loss,
                    current_alpha,
                    lagrange_alpha_loss,
                    lagrange_alpha,
                ) = agent.learn((states, actions, rewards, next_states, dones))
                batches += 1

            if i % args.eval_every == 0:
                eval_reward = evaluate(env, agent)
                wandb.log({"Test Reward": eval_reward, "Episode": i, "Batches": batches}, step=batches)

                average10.append(eval_reward)
                print("Episode: {} | Reward: {} | Polciy Loss: {} | Batches: {}".format(
                    i, eval_reward, policy_loss, batches,
                ))

            wandb.log({
                "Average10": np.mean(average10),
                "Policy Loss": policy_loss,
                "Alpha Loss": alpha_loss,
                "Lagrange Alpha Loss": lagrange_alpha_loss,
                "CQL1 Loss": cql1_loss,
                "CQL2 Loss": cql2_loss,
                "Bellman error 1": bellmann_error1,
                "Bellman error 2": bellmann_error2,
                "Alpha": current_alpha,
                "Lagrange Alpha": lagrange_alpha,
                "Batches": batches,
                "Episode": i,
            })

            if (i % 10 == 0) and args.log_video:
                mp4list = glob.glob("video/*.mp4")
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({
                        "gameplays": wandb.Video(mp4, caption="episode: " + str(i - 10), fps=4, format="gif"),
                        "Episode": i,
                    })

            if i % args.save_every == 0:
                save(config["offline_policy_dir"], agent.actor_local, agent.critic1, agent.critic2, wandb, 0)


if __name__ == "__main__":
    train()
