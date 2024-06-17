import torch

from coprocessors.actors import BaseStimTester
from coprocessors.utils.optimization_methods import *


class QMaxActionConv(BaseStimTester):
    
    def __init__(
        self,
        cop_env,
        healthy_q,
        stims,
        act_conv_net,
        opt_method="grid_search",
        brain_obs=False,
        device="cpu",
    ):
        super().__init__(
            env=cop_env, 
            healthy_q=healthy_q,
            stims=stims, 
            min_max=False,
            device=device,
        )

        self.action_space = cop_env.action_space
        self.cop_env = cop_env
        self.healthy_q = healthy_q
        self.action_conv_net = act_conv_net
        self.history = torch.zeros((5, self.stimulations.shape[1]))
        self.step = 0
        self.brain_obs = brain_obs
        self.opt_method = opt_method
        self.device = device

    def objective(self, actions, *args):
        obs_batch = (args[0].reshape(-1, self.cop_env.observation_space.shape[0]).to(self.device))
        actions = actions.reshape(-1, self.cop_env.action_space.shape[0])
        env_actions = self.action_conv_net(
            obs_batch, torch.from_numpy(actions).float().to(self.device), 0
        )
        if self.brain_obs:
            q_vals_1, q_vals_2 = self.healthy_q(
                obs_batch[:, :9], torch.FloatTensor(env_actions)
            )
        else:
            q_vals_1, q_vals_2 = self.healthy_q(obs_batch, env_actions)
        q_vals = torch.minimum(q_vals_1, q_vals_2)
        return q_vals.detach().cpu().numpy() * -1

    def select_stim(self, obs, disc=51):
        return eval(self.opt_method)(
            self.cop_env, obs, self.objective, device=self.device, disc=disc
        )

    def act(self, obs, deterministic):
        opt_action = self.select_stim(obs)
        self.history = torch.roll(self.history, shifts=(-1))
        self.history[-1, :] = torch.FloatTensor(opt_action)
        self.step += 1
        return opt_action

    def reset(self):
        self.step = 0
        self.history = torch.zeros((5, self.stimulations.shape[1]))
