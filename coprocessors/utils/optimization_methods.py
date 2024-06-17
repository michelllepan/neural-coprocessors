import numpy as np
import torch
from scipy import optimize


def grid_search(env, obs, objective, **kwargs):
    out = np.meshgrid(
        *[
            np.linspace(env.action_space.low[i], env.action_space.high[i], kwargs["disc"])
            for i in range(env.action_space.shape[0])
        ]
    )
    actions = np.vstack(tuple(map(np.ravel, out))).T
    obs_batch = np.tile(obs, (kwargs["disc"] ** env.action_space.shape[0], 1))
    obs_batch = torch.Tensor(obs_batch).to(kwargs["device"])

    q_vals_1 = objective(actions, obs_batch)
    return actions[q_vals_1.argmin()]


def sim_anneal(env, obs, objective, **kwargs):
    opt_action = optimize.dual_annealing(
        objective,
        bounds=list(zip(env.action_space.low, env.action_space.high)),
        maxiter=100,
        args=(torch.Tensor(obs),),
    )
    return opt_action.x


def lbfgs(env, obs, objective, **kwargs):
    opt_action = optimize.minimize(
        objective,
        [0] * env.action_space.shape[0],
        bounds=list(zip(env.action_space.low, env.action_space.high)),
        args=torch.Tensor(obs),
    )
    return opt_action.x
