import numpy as np
import wandb
from PIL import Image

from coprocessors.utils.setup_brain import *


def render_frame(env, camera_id):
    if hasattr(env, 'sim'):
        frame = env.sim.renderer.render_offscreen(
            width=400,
            height=400,
            camera_id=camera_id)
    elif hasattr(env.unwrapped, 'env') and hasattr(env.unwrapped.env, 'sim'):
        frame = env.unwrapped.env.sim.renderer.render_offscreen(
            width=400,
            height=400,
            camera_id=camera_id)
    else:
        frame = env.render()
    return frame


def run_policy(
    policy,
    env,
    reset_policy=True,
    num_tests=50,
    render=False,
    camera_id=0,
    seed=None,
    return_std=False,
):
    env.reset(seed=seed)
    frames, rewards = [], []

    for episode in range(num_tests):
        obs, _ = env.reset()
        if reset_policy:
            policy.reset()

        total_reward = 0
        for _ in range(env.spec.max_episode_steps):
            if render:
                frame = render_frame(env, camera_id)
                frames.append(frame)

            act = policy.act(obs, deterministic=True).reshape(-1,)
            obs, reward, truncated, terminated, _ = env.step(act)

            total_reward += reward
            done = truncated or terminated
            if done: break

        rewards.append(total_reward)
        wandb.log({
            "reward": np.mean(rewards),
            "std": np.std(rewards),
        })
        print(f"episode {episode + 1} of {num_tests}, running average: {np.mean(rewards)}")

    env.close()
    if render:
        frames = [Image.fromarray(f) for f in frames]
        frames[0].save(
            f"viz.gif",
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=20,
            loop=0,
        )

    mean_reward, std_reward = np.mean(rewards), np.std(rewards)
    print("AVERAGE REWARD: {}".format(mean_reward))
    if return_std:
        return mean_reward, std_reward, frames
    else:
        return mean_reward, frames
