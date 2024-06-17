import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

from coprocessors.action_converter.basics import ActionDataset
from coprocessors.action_converter.policies import ActionConverterPolicy
from coprocessors.utils.setup_brain import *
from coprocessors.utils.optimization_methods import *


class FInversePolicy(ActionConverterPolicy):

    def __init__(
        self,
        world_env,
        coproc_env,
        coproc_test_env,
        action_net,
        save_dir,
        stim_dim,
        temporal,
        f_lr,
        train_frequency,
        update_start,
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
        self.train_frequency = train_frequency
        self.opt_method = opt_method
        self.cop_env_test = coproc_test_env
        self.world_env = world_env
        self.healthy_actor = healthy_actor

        self.update_start = update_start
        self.healthy_q = healthy_q
        reward = self.test_policy(0, num_tests=15)

        wandb.log({
            "rollout/ep_rew_mean": reward,
            "global_episode": 0,
        })

    def select_stim(self, obs, disc=100):
        opt_action = self.healthy_actor(obs)
        if isinstance(opt_action, tuple):
            opt_action = opt_action[0]
        scaled_action = self.scale_world_action(opt_action)
        stim = self.net(
            obs=torch.from_numpy(obs).float().reshape(1, -1).to(self.device),
            action=torch.from_numpy(scaled_action).float().reshape(1, -1).to(self.device),
            hist=0,
        )
        return stim

    def test_policy(self, episode, num_tests):
        inv_reward = []
        for _ in range(num_tests):
            obs, _ = self.cop_env_test.reset()
            reward = 0
            done = False
            while not done:
                stim = self.act(obs)
                obs, r, truncated, terminated, info = self.cop_env_test.step(stim)
                done = terminated or truncated
                reward += r
            inv_reward.append(reward)
        inv_reward = np.mean(np.array(inv_reward))

        wandb.log({"test/ep_rew_mean": inv_reward, "global_episode": episode})
        return inv_reward

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
        self.episode = 0
        while self.episode < num_episodes:
            self.episode += 1
            train_reward = self.execute_explore_exploit_episode()
            print(f"Episode {episode}: reward {train_reward}")
            wandb.log({
                "rollout/ep_rew_mean": train_reward,
                "global_episode": self.episode,
            })
            if test_policy:
                self.test_policy(self.episode, num_tests=15)
            self.save_data()

        net = self.train()
        self.save_network(net)

    def train(self, training_data=None):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.f_lr)

        if training_data is None:
            train_data = ActionDataset(data=self.training_data)
        else:
            train_data = ActionDataset(data=training_data)

        train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

        train_loss = []
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
                outputs = self.net(obs, world_action, 0)
                loss = criterion(outputs, cp_action.squeeze(dim=1))
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

        wandb.log({
            "global_episode": self.episode,
            "rollout/inverse_train_loss": np.mean(train_loss),
        })
        return self.net
