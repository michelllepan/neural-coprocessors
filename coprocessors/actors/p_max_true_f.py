import numpy as np
import torch

from coprocessors.actors import BaseStimTester


class PMaxTrueFStim(BaseStimTester):
    def __init__(
        self,
        cop_env,
        healthy_q,
        healthy_policy,
        stims,
        min_max=False,
        device="cpu",
    ):
        super().__init__(
            env=cop_env,
            healthy_q=healthy_q,
            stims=stims,
            min_max=min_max,
            device=device,
        )

        self.action_space = cop_env.action_space
        self.cop_env = cop_env
        self.healthy_q = healthy_q
        self.healthy_policy = healthy_policy
        self.stimulations = stims
        self.min_max = min_max

    def act(self, obs, deterministic):
        if hasattr(self.env, "simulate_multiple_stimulations"):
            env_actions = self.env.simulate_multiple_stimulations(
                torch.FloatTensor(self.stimulations)
            )
            env_actions = env_actions.detach().cpu().numpy()
        else:
            env_actions = np.zeros(
                (self.stimulations.shape[0], self.env.env.action_space.shape[0])
            )
            for i in range(self.stimulations.shape[0]):
                e_a = self.env.simulate_stimulation(
                    torch.FloatTensor(self.stimulations[i, :])
                )
                env_actions[i, :] = e_a

        o_tensor = torch.FloatTensor(obs).reshape(1, -1).to(self.device)
        healthy_action = self.healthy_policy(o_tensor, deterministic=True)

        opt_action = self.stimulations[
            np.linalg.norm(healthy_action - env_actions, axis=1).argmin()
        ]
        return opt_action
