import numpy as np
import torch

from coprocessors.actors import BaseStimTester


class PMaxActionConv(BaseStimTester):
    def __init__(
        self,
        cop_env,
        healthy_q,
        healthy_actor,
        stims,
        act_conv_net,
        min_max=False,
        device="cpu",
        brain_obs=False,
    ):
        super().__init__(
            env=cop_env,
            healhty_q=healthy_q, 
            stims=stims,
            min_max=False,
            device=device,
        )
        
        self.action_space = cop_env.action_space
        self.cop_env = cop_env
        self.healthy_q = healthy_q.to(device)
        self.action_conv_net = act_conv_net.to(device)
        self.history = torch.zeros((5, self.stimulations.shape[1]))
        self.step = 0
        self.brain_obs = brain_obs
        self.healthy_policy = healthy_actor.to(device)

        self.stimulations = torch.FloatTensor(self.stimulations).to(device)

    def act(self, o, deterministic):
        obs_batch = np.tile(o, (self.stimulations.shape[0], 1))
        obs_batch = torch.Tensor(obs_batch).to(self.device)

        hist_batch = torch.tile(
            self.history.unsqueeze(0), (self.stimulations.shape[0], 1, 1)
        )
        env_actions = self.action_conv_net(obs_batch, self.stimulations, hist_batch)
        if self.brain_obs:
            healthy_action = self.healthy_policy(
                torch.FloatTensor(o[:9]).to(self.device), deterministic=True
            )
        else:
            healthy_action = self.healthy_policy(
                torch.FloatTensor(o).reshape(1, -1).to(self.device), deterministic=True
            )

        healthy_action = healthy_action.detach().cpu().numpy()
        env_actions = env_actions.detach().cpu().numpy()
        opt_action = self.stimulations[np.linalg.norm(healthy_action - env_actions, axis=1).argmin()]

        self.step += 1
        return opt_action

    def reset(self):
        self.step = 0
