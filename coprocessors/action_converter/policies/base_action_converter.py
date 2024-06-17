import copy
import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

from coprocessors.action_converter.basics import ActionDataset, ActionNet
from coprocessors.utils.optimization_methods import *


class ActionConverterPolicy(ABC):

    def __init__(
        self,
        world_env,
        coproc_env,
        coproc_env_test,
        action_net,
        save_dir,
        stim_dim,
        temporal,
        f_lr,
        device="cpu",
        seed=0,
    ):
        self.cop_env = coproc_env
        self.cop_env_test = coproc_env_test
        self.world_env = world_env
        self.save_dir = save_dir
        self.device = device
        self.stim_dim = stim_dim
        self.temporal = temporal
        self.f_lr = f_lr
        self.training_data = {
            "obs": [],
            "cp_action": [],
            "next_obs": [],
            "w_action": [],
            "next_cp_action": [],
            "history": [],
            "reward": [],
        }
        self.sim_data = copy.deepcopy(self.training_data)
        self.seed = seed
        self.net = action_net
        self.min_action = -1
        self.max_action = 1
        self.steps = 0

    def act(self, obs=None):
        if obs is None:
            opt_action = self.select_stim(self.obs)
        else:
            opt_action = self.select_stim(obs)
        return opt_action

    @abstractmethod
    def select_stim(self, obs, disc=100):
        pass

    def step(self, action, obs1, sim=False):
        env = self.world_env if sim else self.cop_env
        training_data = self.sim_data if sim else self.training_data
        if sim:
            world_action = (
                self.predict_world_action(
                    torch.from_numpy(obs1).float().to(self.device).reshape(1, -1),
                    torch.from_numpy(action).float().to(self.device).reshape(1, -1),
                )
                .detach()
                .cpu()
                .numpy()
            )
            world_action = world_action.reshape(
                -1,
            )
            obs2, reward, done, _, _ = env.step(world_action)
            next_cp_action = self.select_stim(obs2, disc=50)
            training_data["next_cp_action"].append(next_cp_action)
        else:
            obs2, reward, done, _, _, world_action = env.step(action.flatten())
            world_action = self.scale_world_action(world_action)
            training_data["next_cp_action"].append(0)

        training_data["obs"].append(obs1)
        training_data["next_obs"].append(obs2)
        training_data["reward"].append(reward)
        training_data["cp_action"].append(action)
        training_data["w_action"].append(world_action)
        training_data["history"].append(0)
        return obs2, done, reward

    def scale_world_action(self, action):
        return self.min_action + (self.max_action - self.min_action) * (
            action - self.world_env.action_space.low
        ) / (self.world_env.action_space.high - self.world_env.action_space.low)

    def rescale_world_action(self, action):
        action = self.world_env.action_space.low + (
            self.world_env.action_space.high - self.world_env.action_space.low
        ) * ((action - self.min_action) / (self.max_action - self.min_action))
        return action

    def save_network(self, net):
        torch.save(net.state_dict(), os.path.join(self.save_dir, "model.pth"))

    def save_data(self):
        with open(
            os.path.join(self.save_dir, "action_hidden_data_train.pickle"), "wb"
        ) as fp:
            pickle.dump(self.training_data, fp)

    def get_net(self):
        net = ActionNet(
            obs_size=self.cop_env.observation_space.shape[0],
            stim_size=self.stim_dim,
            act_size=self.cop_env.env.action_space.shape[0],
            temporal=self.temporal,
        )
        net.to(self.device)
        return net

    def train(self, training_data=None):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.f_lr)

        if training_data is None:
            train_data = ActionDataset(data=self.training_data)
        else:
            train_data = ActionDataset(data=training_data)

        train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

        train_loss, val_loss = [], []
        average_change = []
        prev_loss = 10000
        for epoch in range(75):
            running_loss = 0.0
            num_samples = 0

            for data in train_dataloader:
                obs, cp_action, world_action, hist = map(
                    lambda x: x.to(self.device), data
                )
                optimizer.zero_grad()
                outputs = self.net(obs, cp_action, 0)
                loss = criterion(outputs, world_action)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(obs)
                num_samples += len(obs)

            train_loss.append(running_loss / num_samples)

            average_change.append(prev_loss - running_loss / num_samples)
            prev_loss = running_loss / num_samples

            if epoch % 10 == 0:
                if np.array(np.mean(average_change)) < 0.001:
                    break
                else:
                    average_change = []
        return self.net

    @abstractmethod
    def collect_and_train(self, num_episodes, num_q_update_traj):
        pass

    def get_random_data(self):
        return self.cop_env.action_space.sample()

    def predict_world_action(self, o, cp_action):
        env_actions = self.net(o.to(self.device), cp_action.to(self.device), 0)
        return env_actions

    def reset_sim_dataset(self):
        self.sim_data = {
            "obs": [],
            "cp_action": [],
            "next_obs": [],
            "w_action": [],
            "next_cp_action": [],
            "history": [],
            "reward": [],
        }

    def test_policy(self, episode, num_tests):
        all_reward = []
        for _ in range(num_tests):
            obs, _ = self.cop_env_test.reset()

            reward = 0
            done = False
            while not done:
                action = self.select_stim(obs=obs)
                obs, r, truncated, terminated, _ = self.cop_env_test.step(action)
                done = terminated or truncated
                reward += r
            all_reward.append(reward)
        all_reward = np.mean(np.array(all_reward))
        wandb.log({
            "test/ep_rew_mean": all_reward,
            "global_episode": episode,
        })
        return all_reward
