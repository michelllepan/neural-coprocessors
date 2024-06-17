import numpy as np
import torch

from coprocessors.actors import BaseStimTester


class QMaxTrueFStim(BaseStimTester):
    
    def __init__(
        self,
        cop_env,
        healthy_q,
        disc,
        min_max=False,
        device="cpu",
        opt_method="grid_search",
    ):
        super().__init__(
            env=cop_env,
            healthy_q=healthy_q,
            min_max=min_max,
            device=device,
        )

        self.opt_method = opt_method
        self.disc = disc

    def objective(self, actions, *args):
        actions = torch.FloatTensor(actions).reshape(-1, self.env.action_space.shape[0])
        o = torch.FloatTensor(args[0]).reshape(-1, self.env.observation_space.shape[0])
        if hasattr(self.env, "simulate_multiple_stimulations"):
            env_actions = self.env.simulate_multiple_stimulations(torch.FloatTensor(actions))
        else:
            env_actions = np.zeros((actions.shape[0], self.env.env.action_space.shape[0]))
            for i in range(actions.shape[0]):
                e_a = self.env.simulate_stimulation(torch.FloatTensor(actions[i, :]))
                env_actions[i, :] = e_a
            env_actions = torch.FloatTensor(env_actions).to(self.device)

        q_vals_1, _ = self.healthy_q(o, env_actions)
        return q_vals_1.detach().numpy() * -1
