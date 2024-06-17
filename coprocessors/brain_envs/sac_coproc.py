import gymnasium as gym
import numpy as np
import torch


class SACCoprocessorWrapper(gym.Wrapper):

    def __init__(
        self,
        env,
        brain,
        stim_dim,
        brain_obs=False,
        return_world_action=False,
        device="cpu",
    ):
        super().__init__(env)
        self.env = env
        self.brain = brain
        self.brain_obs = brain_obs
        self.return_world_action = return_world_action
        self.device = device

        if brain_obs:
            raise NotImplementedError

        self.action_space = gym.spaces.Box(
            low=-5,
            high=5,
            shape=(stim_dim,),
        )
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=self.env.observation_space.shape,
        )

    def reset(self, seed=None):
        self.obs, _ = self.env.reset(seed=seed)
        return self.obs, _

    def seed(self, seed=None):
        self.env.reset(seed=seed)

    def step(self, coprocessor_action):
        layer_obs = self.brain.predict_one(
            self.obs,
            deterministic=True,
        )
        action, _ = self.brain.predict_two(
            layer_obs,
            coprocessor_action=coprocessor_action,
            deterministic=True,
        )

        step_result = self.env.step(action)
        self.obs = step_result[0]

        if self.return_world_action:
            return tuple(list(step_result) + [action])
        return step_result

    def simulate_stimulation(self, coprocessor_action):
        layer_obs = self.brain.predict_one(
            self.obs,
            deterministic=True,
        )
        action, _ = self.brain.predict_two(
            layer_obs,
            coprocessor_action=coprocessor_action,
            deterministic=True,
        )
        return action

    def simulate_multiple_stimulations(self, coprocessor_actions):
        obs_batch = np.tile(self.obs, (coprocessor_actions.shape[0], 1))
        layer_obs = self.brain.policy.actor.forward_one(
            torch.from_numpy(obs_batch).to(self.device),
            deterministic=True,
        )
        self.brain.policy.actor.coprocessor_action = coprocessor_actions.to(self.device)
        actions = self.brain.policy.actor.forward_two(
            layer_obs,
            deterministic=True,
        )
        return actions
