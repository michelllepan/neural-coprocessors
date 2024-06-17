from coprocessors.action_converter.basics import ActionNet, InverseActionNet
from coprocessors.action_converter.policies import *
from coprocessors.utils.setup_env import *
import torch


def make_action_net(config, coproc_env, device):
    obs_size = coproc_env.observation_space.shape[0]
    act_size = coproc_env.env.action_space.shape[0]
    net_class = InverseActionNet if config["action_conv"] == "inverse" else ActionNet

    torch.manual_seed(config["action_conv_seed"])
    if config["gym_env"] in ("Hopper-v4", "BipedalWalker-v3"):
        net = net_class(
            obs_size=obs_size,
            stim_size=config["stim_dim"],
            hid_size1=64,
            hid_size2=128,
            hid_size3=32,
            act_size=act_size,
            temporal=config["temporal"],
        )
    else:
        net = net_class(
            obs_size=obs_size,
            stim_size=config["stim_dim"],
            act_size=act_size,
            temporal=config["temporal"],
        )

    net.to(device)
    return net


def make_action_conv_coproc_env(config, device):
    gym_env = make_gym_env(config)
    brain = load_injured_brain(config, gym_env, device)
    coproc_env = make_coproc_env(config, gym_env, brain, True, device)
    return coproc_env


def setup_random_action_conv(config, device):
    gym_env = make_gym_env(config)
    coproc_env = make_action_conv_coproc_env(config, device)
    coproc_test_env = setup_coproc_env(config, device)
    action_net = make_action_net(config, coproc_env, device)
    opt_actor = load_opt_actor(config, gym_env, device)
    opt_q = load_opt_q(config, gym_env, device)

    return RandomActionConverterPolicy(
        world_env=gym_env,
        coproc_env=coproc_env,
        coproc_test_env=coproc_test_env,
        action_net=action_net,
        save_dir=config["action_conv_dir"],
        stim_dim=config["stim_dim"],
        temporal=config["temporal"],
        gamma=config["gamma"],
        train_frequency=config["train_frequency"],
        update_start=config["update_start"],
        f_lr=config["f_lr"],
        healthy_actor=opt_actor,
        healthy_q=opt_q,
        opt_method=config["opt_method"],
        device=device,
    )


def setup_qmax_action_conv(config, device):
    gym_env = make_gym_env(config)
    coproc_env = make_action_conv_coproc_env(config, device)
    coproc_test_env = setup_coproc_env(config, device)
    action_net = make_action_net(config, coproc_env, device)
    opt_actor = load_opt_actor(config, gym_env, device)
    opt_q = load_opt_q(config, gym_env, device)

    return QMaxOnlyPolicy(
        world_env=gym_env,
        coproc_env=coproc_env,
        coproc_test_env=coproc_test_env,
        action_net=action_net,
        save_dir=config["action_conv_dir"],
        stim_dim=config["stim_dim"],
        temporal=config["temporal"],
        gamma=config["gamma"],
        q_lr=config["q_lr"],
        tau=config["tau"],
        use_target=config["num_q_update_traj"] > 0,
        train_frequency=config["train_frequency"],
        update_start=config["update_start"],
        f_lr=config["f_lr"],
        healthy_actor=opt_actor,
        healthy_q=opt_q,
        opt_method=config["opt_method"],
        device=device,
    )


def setup_inverse_action_conv(config, device):
    gym_env = make_gym_env(config)
    coproc_env = make_action_conv_coproc_env(config, device)
    coproc_test_env = setup_coproc_env(config, device)
    action_net = make_action_net(config, coproc_env, device)
    opt_actor = load_opt_actor(config, gym_env, device)
    opt_q = load_opt_q(config, gym_env, device)

    return FInversePolicy(
        world_env=gym_env,
        coproc_env=coproc_env,
        coproc_test_env=coproc_test_env,
        action_net=action_net,
        save_dir=config["action_conv_dir"],
        stim_dim=config["stim_dim"],
        temporal=config["temporal"],
        f_lr = config["f_lr"],
        train_frequency=config["train_frequency"],
        update_start=config["update_start"],
        healthy_actor=opt_actor,
        healthy_q=opt_q,
        device=device,
    )


def setup_action_conv(config, device):
    if config["action_conv"] == "random":
        return setup_random_action_conv(config, device)
    elif config["action_conv"] in ("qmax", "qmax_offline"):
        return setup_qmax_action_conv(config, device)
    elif config["action_conv"] == "inverse":
        return setup_inverse_action_conv(config, device)
    else:
        raise NotImplementedError


def load_action_conv(config, coproc_env, device):
    net = make_action_net(config, coproc_env, device)
    net.to(device)
    net.load_state_dict(torch.load(config["action_conv_path"]))
    return net
