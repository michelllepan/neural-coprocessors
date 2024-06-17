import stable_baselines3

from coprocessors.actors import *
from coprocessors.utils import get_stimulation_sweep
from coprocessors.utils.setup_env import *
from coprocessors.utils.setup_action_conv import *
from myosuite import myosuite


def setup_healthy_brain_policy(config, device="cpu"):
    gym_env = make_gym_env(config)
    brain = load_healthy_brain(config, gym_env, device=device)
    if config["brain"] == "michaels":
        brain_policy = VanillaPolicyRecurrent(gym_env, brain, device=device)
        return brain_policy, gym_env, True
    else:
        brain_policy = VanillaPolicy(gym_env, brain.predict, device=device)
        return brain_policy, gym_env, False


def setup_injured_brain_policy(config, device="cpu"):
    gym_env = make_gym_env(config)
    brain = load_injured_brain(config, gym_env, device)
    if config["brain"] == "michaels":
        brain_policy = VanillaPolicyRecurrent(gym_env, brain, device=device)
    else:
        brain_policy = VanillaPolicy(gym_env, brain.predict, device=device)
    return brain_policy, gym_env, False


def setup_coproc_env(config, device="cpu"):
    gym_env = make_gym_env(config)
    brain = load_injured_brain(config, gym_env, device)
    coproc_env = make_coproc_env(config, gym_env, brain, return_world_action=False, device=device)
    return coproc_env


def setup_sac_coproc_policy(config, device="cpu"):
    gym_env = make_gym_env(config)
    brain = load_injured_brain(config, gym_env, device)
    coproc_env = make_coproc_env(config, gym_env, brain, return_world_action=False, device=device)

    sac = stable_baselines3.SAC("MlpPolicy", coproc_env, verbose=1)
    sac_policy = sac.load(
        path=config["coproc_path"],
        env=coproc_env,
        custom_objects={
            "observation_space": coproc_env.observation_space,
            "action_space": coproc_env.action_space,
        },
    )
    coproc = VanillaPolicy(coproc_env, sac_policy.predict)
    return coproc, coproc_env, False

def setup_pmax_coproc_policy(config, device="cpu"):
    gym_env = make_gym_env(config)
    brain = load_injured_brain(config, gym_env, device)
    coproc_env = make_coproc_env(config, gym_env, brain, return_world_action=False, device=device)

    opt_actor = load_opt_actor(config, gym_env, device)
    opt_q = load_opt_q(config, gym_env, device)
    stimulations = get_stimulation_sweep(coproc_env, config["disc"])
    if config["action_conv"] == "true_f":
        coproc = PMaxTrueFStim(
            coproc_env, opt_q, opt_actor, stimulations, device
        )
    else:
        action_conv = load_action_conv(config, coproc_env, device)
        coproc = PMaxActionConv(
            coproc_env,
            opt_q,
            opt_actor,
            stimulations,
            action_conv,
            config["brain_obs"],
            device,
        )
    return coproc, coproc_env, True


def setup_qmax_coproc_policy(config, device="cpu"):
    gym_env = make_gym_env(config)
    brain = load_injured_brain(config, gym_env, device)
    coproc_env = make_coproc_env(config, gym_env, brain, return_world_action=False, device=device)

    opt_q = load_opt_q(config, gym_env, device)
    stimulations = get_stimulation_sweep(coproc_env, config["disc"])
    if config["action_conv"] == "true_f":
        coproc = QMaxTrueFStim(coproc_env, opt_q, stimulations, device)
    else:
        action_conv = load_action_conv(config, coproc_env, device)
        coproc = QMaxActionConv(
            cop_env=coproc_env,
            healthy_q=opt_q,
            stims=stimulations,
            act_conv_net=action_conv,
            opt_method=config["opt_method"],
            brain_obs=config["brain_obs"],
            device=device,
        )
    return coproc, coproc_env, True


def setup_random_coproc_policy(config, device="cpu"):
    gym_env = make_gym_env(config)
    brain = load_injured_brain(config, gym_env, device)
    coproc_env = make_coproc_env(config, gym_env, brain, return_world_action=False, device=device)
    coproc = RandomActor(coproc_env, device)
    return coproc, coproc_env, False


def setup_policy(config, device="cpu"):
    if config["coproc"] is None:
        if config["pct_lesion"] == 0:
            return setup_healthy_brain_policy(config, device)
        else:
            return setup_injured_brain_policy(config, device)
    elif config["coproc"] == "sac":
        return setup_sac_coproc_policy(config, device)
    elif config["coproc"] == "pmax":
        return setup_pmax_coproc_policy(config, device)
    elif config["coproc"] == "qmax":
        return setup_qmax_coproc_policy(config, device)
    elif config["coproc"] == "random":
        return setup_random_coproc_policy(config, device)
    else:
        raise NotImplementedError
