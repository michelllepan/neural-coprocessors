from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch

from michaels.experiment.experiment import get_config


DEFAULT_OBS_DIM = 10
DEFAULT_STIM_SIGMA = 0.1
DEFAULT_OBS_SIGMA = 0.1


class MikeCoprocessorWrapper(gym.Wrapper):
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

        cfg = get_config(
            coadapt=False,
            cuda=self.device,
            obs_out_dim=DEFAULT_OBS_DIM,
            num_stim_channels=stim_dim,
            stim_sigma=DEFAULT_STIM_SIGMA,
            obs_sigma=DEFAULT_OBS_SIGMA,
        )
        self.observer = cfg.observer_instance

        if self.brain_obs:
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3 * self.observer.out_dim + self.env.observation_space.shape[0]),
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=self.env.observation_space.low,
                high=self.env.observation_space.high,
                shape=self.env.observation_space.shape,
            )

        self.action_space = gym.spaces.Box(
            low=-5,
            high=5,
            shape=(stim_dim,),
        )
        self._max_episode_steps = self.env._max_episode_steps

    def reset(self, seed=None):
        self.state, _ = self.env.reset(seed=seed)

        self.brain.actor_summarizer.rnn.reset()
        self.brain.reinitialize_hidden()
        
        if self.brain_obs:
            summarizer_obs = self.brain.actor_summarizer.rnn.observe(self.observer)
            mike_obs = torch.cat(summarizer_obs).flatten().cpu().numpy()
            return np.concatenate([self.state, mike_obs])
        else:
            return self.state, {}

    def simulate_stimulation(self, action):
        old_mike = deepcopy(self.brain)
        old_mike.actor_summarizer.rnn.stimulate(action)
        mike_action = old_mike.simulate(self.state)
        return mike_action

    def simulate_healthy(self):
        old_mike = deepcopy(self.brain)
        old_mike.actor_summarizer.rnn.lesion = None
        mike_action = old_mike.act(self.state, deterministic=True)

        old_state = self.env.unwrapped.clone_full_state()
        state, reward, terminated, truncated, _ = self.env.step(mike_action)
        self.env.unwrapped.restore_full_state(old_state)
        return reward

    def step(self, action):
        action = torch.Tensor(action).float().to(self.device)
        self.brain.actor_summarizer.rnn.stimulate(action)
        mike_action = self.brain.act(self.state, deterministic=True)

        self.state, reward, terminated, truncated, _ = self.env.step(mike_action)
        done = terminated or truncated
        if done:
            self.brain.reinitialize_hidden()
        
        summarizer_obs = self.brain.actor_summarizer.rnn.observe(self.observer)
        mike_obs = torch.cat(summarizer_obs).flatten().cpu().numpy()
        if self.brain_obs:
            state = np.concatenate([self.state, mike_obs])
        else:
            state = self.state

        if self.return_world_action:
            return state, reward, done, False, {}, mike_action
        else:
            return state, reward, done, False, {}
