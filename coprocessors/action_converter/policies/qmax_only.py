import pickle
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

from coprocessors.action_converter.basics import ActionDataset
from coprocessors.action_converter.policies import ActionConverterPolicy
from coprocessors.utils.setup_brain import *
from coprocessors.utils.optimization_methods import *


class QMaxOnlyPolicy(ActionConverterPolicy):

    def __init__(
        self,
        world_env,
        coproc_env,
        coproc_test_env,
        action_net,
        save_dir,
        stim_dim,
        temporal,
        gamma,
        q_lr,
        tau,
        use_target,
        train_frequency,
        update_start,
        f_lr,
        healthy_actor,
        healthy_q,
        opt_method="grid_search",
        device="cpu",
    ):
        super().__init__(
            world_env=world_env,
            coproc_env=coproc_env,
            coproc_env_test=coproc_test_env,
            action_net=action_net,
            save_dir=save_dir,
            stim_dim=stim_dim,
            temporal=temporal,
            f_lr=f_lr,
            device=device,
        )

        self.trained = False
        self.sim_steps = 0
        self.world_env = world_env
        self.train_frequency = train_frequency
        self.update_start = update_start
        self.healthy_actor = healthy_actor
        self.healthy_q = healthy_q
        self.gamma = gamma
        self.q_lr = q_lr
        self.tau = tau
        self.opt_method = opt_method

        if use_target: 
            self.target_net = deepcopy(self.healthy_q)

        reward = self.test_policy(0, num_tests=15)
        wandb.log({
            "rollout/ep_rew_mean": reward,
            "global_episode": 0,
        })

    def train_q(self, training_data=None):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.healthy_q.parameters(), lr=self.q_lr)
        train_data = ActionDataset(
            data=training_data or self.sim_data, include_trans=True
        )
        train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

        train_loss, val_loss = [], []
        average_change = []
        prev_loss = 10000

        for epoch in range(50):
            running_loss = 0.0
            num_samples = 0

            for data in train_dataloader:
                obs, next_obs, reward, cp_action, next_cp_action, world_action, _ = map(
                    lambda x: x.to(self.device), data
                )
                optimizer.zero_grad()
                next_world_action = self.predict_world_action(next_obs, next_cp_action)
                next_q_1, next_q_2 = self.target_net(next_obs, next_world_action)
                next_q = torch.minimum(next_q_1, next_q_2)
                q_target = reward[:, None] + self.gamma * next_q
                q_target = q_target[None].expand(2, -1, -1)
                q_pred1, q_pred2 = self.healthy_q(obs, world_action)
                q_pred = torch.stack([q_pred1, q_pred2], dim=0)

                assert q_pred.shape == (2, len(obs), 1)
                assert q_target.shape == (2, len(obs), 1)

                loss = criterion(q_pred, q_target.detach())
                loss.backward()
                optimizer.step()

                # soft update
                current_dict = self.healthy_q.state_dict()
                target_dict = self.target_net.state_dict()
                with torch.no_grad():
                    for name, param in target_dict.items():
                        new_param = current_dict[name]
                        param.copy_(self.tau * new_param + (1-self.tau) * param)

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
        return self.healthy_q

    def objective(self, actions, *args):

        o = args[0].reshape(-1, self.cop_env.observation_space.shape[0]).to(self.device)
        actions = actions.reshape(-1, self.cop_env.action_space.shape[0])
        env_actions = self.predict_world_action(o, torch.from_numpy(actions).float())

        q_vals = self.healthy_q(o, env_actions)
        if type(q_vals) is tuple:
            q_vals = torch.minimum(*q_vals)
        return q_vals.detach().cpu().numpy() * -1

    def select_stim(self, obs, disc=100):
        return eval(self.opt_method)(
            self.cop_env, obs, self.objective, device=self.device, disc=disc
        )

    def execute_q_tune_episode(self, train=True):
        self.obs, _ = self.world_env.reset()
        total_reward = 0

        for j in range(self.cop_env.spec.max_episode_steps):
            action = self.act()
            self.obs, done, reward = self.step(action, self.obs, sim=True)
            total_reward += reward
            if self.sim_steps % 50 == 0 and train:
                self.healthy_q = self.train_q()
            if done:
                break
            self.sim_steps += 1
        return total_reward

    def execute_explore_exploit_episode(self):
        self.obs, _ = self.cop_env.reset()
        pretrain = 5
        total_reward = 0
        for j in range(self.cop_env.spec.max_episode_steps):
            action = self.act()
            self.obs, done, reward = self.step(action, self.obs)
            total_reward += reward
            if j % self.train_frequency == 0 and self.steps >= pretrain:
                self.net = self.train()
                self.trained = True
            self.steps += 1
            if done:
                break
        return total_reward

    def collect_and_train(self, num_episodes, num_q_update_traj, test_policy=False):
        episode = 0
        while episode < num_episodes:
            episode += 1
            train_reward = self.execute_explore_exploit_episode()
            wandb.log({
                "rollout/ep_rew_mean": train_reward,
                "global_episode": episode,
            })
            print(f"Episode {episode}: reward {train_reward}")
            if test_policy:
                self.test_policy(episode, num_tests=15)
            self.save_data()

            if episode > self.update_start and num_q_update_traj > 0:
                self.reset_sim_dataset()
                while self.sim_steps < 500:
                    self.execute_q_tune_episode(train=False)
                for j in range(num_q_update_traj):
                    self.execute_q_tune_episode()

        net = ActionConverterPolicy.train(self)
        self.save_network(net)
