import numpy as np


def get_stimulation_sweep(cop_env, disc):
    out = np.meshgrid(
        *[
            np.linspace(cop_env.action_space.low[i], cop_env.action_space.high[i], disc)
            for i in range(cop_env.action_space.shape[0])
        ]
    )
    stimulations = np.vstack(tuple(map(np.ravel, out))).T
    return stimulations
