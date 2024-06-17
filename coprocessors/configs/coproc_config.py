import os
from typing import Optional


def coproc_config(
    gym_env: str,
    brain: str,
    pct_lesion: float = 0.0,
    region: Optional[str] = None,
    coproc: Optional[str] = None,
    stim_dim: int = 0,
    brain_obs: bool = False,
    temporal: bool = False,
    action_conv: Optional[str] = "true_f",
    data_size: int = 0,
    opt_method: Optional[str] = None,
    num_q_update_traj: int = 0,
    gamma: float = 0.99,
    q_lr: float = 1e-4,
    tau: float = 1e-3,
    train_frequency: int = 0,
    update_start: int = 0,
    f_lr: float = 5e-3,
    disc: int = 51,
    max_episode_steps: int = None,
    env_seed: int = 0,
    brain_seed: int = 0,
    injury_seed: int = 5,
    stim_seed: int = 5,
    action_conv_seed: int = 5,
    **kwargs,
):
    # configure brain path
    if brain == "michaels":
        brain_path = os.path.join(
            "models",
            gym_env, 
            f"{brain}_brain",
            "brain_models",
            f"policy_{brain_seed}")
        lesion_path = os.path.join(
            "models",
            gym_env,
            f"{brain}_brain",
            "lesions",
            "lesion-{}_region-{}.pickle".format(str(pct_lesion).split('.')[-1], region),
        )
    else:
        brain_path = os.path.join(
            "models",
            gym_env,
            f"{brain}_brain",
            "brain_models",
            f"policy_{brain_seed}",
        )
        lesion_path = None

    # configure SAC coprocessor path
    if coproc == "sac":
        model_dir = os.path.join(
            "models",
            gym_env,
            f"{brain}_brain",
            "sac_coproc_models",
        )
        model_file = "lesion-{}_region-{}_stim-{}_brain-obs{}_seeds-{}-{}.zip".format(
            pct_lesion,
            region,
            stim_dim,
            brain_obs,
            injury_seed,
            stim_seed,
        )
        coproc_path = os.path.join(model_dir, model_file)
    else:
        coproc_path = None

    # configure action converter path
    if action_conv == "true_f":
        action_conv_dir = None
        action_conv_path = None
    else:
        model_dir = os.path.join(
            "models",
            gym_env,
            f"{brain}_brain",
            "action_conv_models",
            action_conv,
        )
        model_subdir = "lesion-{}_region-{}_stim-{}_brain-obs-{}_data-{}_seeds-{}-{}".format(
            pct_lesion,
            region,
            stim_dim,
            brain_obs,
            data_size,
            injury_seed,
            stim_seed,
        )
        action_conv_subdir = "opt-method-{}_num-update-{}_train-freq-{}_update-start-{}_seed-{}".format(
            opt_method,
            num_q_update_traj,
            train_frequency,
            update_start,
            action_conv_seed,
        )
        action_conv_dir = os.path.join(
            model_dir, model_subdir, action_conv_subdir
        )
        action_conv_path = os.path.join(
            model_dir, model_subdir, action_conv_subdir, "model.pth"
        )

    # configure environment actor path, used for opt q and opt actor
    env_actor_critic_path = None
    if brain == "michaels":
        env_actor_critic_path = os.path.join(
            "models",
            gym_env,
            "env_actor_critic",
            "model"
        )

    # configure path for offline RL data and and policy
    offline_data_path = os.path.join(
        "data",
        gym_env,
        f"brain-{brain_seed}_data-{data_size}.pickle",
    )
    offline_policy_dir = os.path.join(
        "models",
        gym_env,
        "offline",
    )

    config = {
        "gym_env": gym_env,
        "max_episode_steps": max_episode_steps,
        "disc": disc,
        "brain": brain,
        "brain_path": brain_path,
        "brain_seed": brain_seed,
        "pct_lesion": pct_lesion,
        "lesion_path": lesion_path,
        "region": region,
        "env_seed": env_seed,
        "injury_seed": injury_seed,
        "coproc": coproc,
        "coproc_path": coproc_path,
        "stim_dim": stim_dim,
        "temporal": temporal,
        "brain_obs": brain_obs,
        "action_conv": action_conv,
        "action_conv_dir": action_conv_dir,
        "action_conv_path": action_conv_path,
        "action_conv_seed": action_conv_seed,
        "data_size": data_size,
        "opt_method": opt_method,
        "num_q_update_traj": num_q_update_traj,
        "gamma": gamma,
        "q_lr": q_lr,
        "tau": tau,
        "train_frequency": train_frequency,
        "update_start": update_start,
        "f_lr": f_lr,
        "stim_seed": stim_seed,
        "env_actor_critic_path": env_actor_critic_path,
        "offline_data_path": offline_data_path,
        "offline_policy_dir": offline_policy_dir,
    }
    return config
