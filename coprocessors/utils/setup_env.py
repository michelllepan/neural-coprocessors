import os

import gym
import gymnasium
import stable_baselines3

from coprocessors.actors import *
from coprocessors.brain_envs import MikeCoprocessorWrapper, SACCoprocessorWrapper
from coprocessors.utils import setup_brain
from CQL_SAC.agent import CQLSAC
from myosuite import myosuite


def make_gym_env(config):
    env_name = config["gym_env"]
    if "myo" in env_name:
        gym_env = gym.make(env_name, apply_api_compatibility=True)
    elif env_name == "LunarLander-v2":
        gym_env = gymnasium.make(env_name, continuous=True, render_mode="rgb_array")
    else:
        gym_env = gymnasium.make(env_name, render_mode="rgb_array")
    if config["max_episode_steps"] is not None:
        gym_env._max_episode_steps = config["max_episode_steps"]
    return gym_env


def make_coproc_env(config, gym_env, brain, return_world_action, device):
    if config["brain"] == "michaels":
        coproc_env = MikeCoprocessorWrapper(
            env=gym_env,
            brain=brain,
            stim_dim=config["stim_dim"],
            brain_obs=config["brain_obs"],
            return_world_action=return_world_action,
            device=device,
        )
    else:
        coproc_env = SACCoprocessorWrapper(
            env=gym_env,
            brain=brain,
            stim_dim=config["stim_dim"],
            brain_obs=config["brain_obs"],
            return_world_action=return_world_action,
            device=device,
        )
    if config["max_episode_steps"] is not None:
        coproc_env._max_episode_steps = config["max_episode_steps"]
    return coproc_env


def load_healthy_brain(config, gym_env, device):
    if config["brain"] == "michaels":
        brain = setup_brain.get_michaels_brain(
            env=gym_env,
            brain_path=config["brain_path"],
            device=device,
        )
    else:
        brain = setup_brain.get_sac_brain(
            env=gym_env,
            brain_path=config["brain_path"],
            device=device,
        )
    return brain


def load_injured_brain(config, gym_env, device, path_prefix=""):
    if config["brain"] == "michaels":
        brain = setup_brain.get_michaels_brain(
            env=gym_env,
            brain_path=os.path.join(path_prefix, config["brain_path"]),
            pct_lesion=config["pct_lesion"],
            lesion_path=os.path.join(path_prefix, config["lesion_path"]),
            region=config["region"],
            injury_seed=config["injury_seed"],
            stim_dim=config["stim_dim"],
            stim_seed=config["stim_seed"],
            temporal=config["temporal"],
            device=device,
        )
    else:
        brain = setup_brain.get_sac_brain(
            env=gym_env,
            brain_path=os.path.join(path_prefix, config["brain_path"]),
            pct_lesion=config["pct_lesion"],
            injury_seed=config["injury_seed"],
            stim_dim=config["stim_dim"],
            stim_seed=config["stim_seed"],
            device=device,
        )
    return brain


def load_opt_actor(config, gym_env, device):
    if config["brain"] == "michaels":
        sac = stable_baselines3.SAC("MlpPolicy", gym_env, verbose=1)
        env_actor_critic = sac.load(config["env_actor_critic_path"])
        actor = env_actor_critic.predict
    else:
        brain = load_healthy_brain(config, gym_env, device)
        actor = brain.act
    return actor


def load_opt_q(config, gym_env, device):
    if config["action_conv"] == "qmax_offline":
        offline_policy = CQLSAC(
            state_size=gym_env.observation_space.shape[0],
            action_size=gym_env.action_space.shape[0],
            tau=5e-3,
            hidden_size=256,
            learning_rate=3e-4,
            temp=1.0,
            with_lagrange=0,
            cql_weight=1.0,
            target_action_gap=10,
            device=device,
        )
        offline_policy.load_critic(config["offline_policy_dir"], device=device)
        q = lambda s, a: (offline_policy.critic1(s, a), offline_policy.critic2(s, a))
    elif config["brain"] == "michaels":
        sac = stable_baselines3.SAC("MlpPolicy", gym_env, verbose=1)
        env_actor_critic = sac.load(config["env_actor_critic_path"])
        q = env_actor_critic.critic.to(device)
    else:
        brain = load_healthy_brain(config, gym_env, device)
        q = brain.critic.to(device)
    return q


def setup_coproc_env(config, device, path_prefix=""):
    gym_env = make_gym_env(config)
    brain = load_injured_brain(config, gym_env, device, path_prefix)
    coproc_env = make_coproc_env(config, gym_env, brain, False, device)
    return coproc_env
