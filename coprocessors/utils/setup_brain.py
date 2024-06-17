import dill
import itertools
import numpy as np
import os
import torch

import michaels.experiment as experiment
from coprocessors.utils.injured_sac import InjuredSAC
from offpcc.algorithms_recurrent import RecurrentSAC
from michaels.experiment.experiment import get_config
from myosuite import myosuite


DEFAULT_OBS_DIM = 10
DEFAULT_STIM_SIGMA = .1
DEFAULT_OBS_SIGMA = .1

def get_michaels_brain(
    env,
    brain_path,
    set_predefined_lesion=True,
    pct_lesion=0.0,
    lesion_path=None,
    region=None,
    injury_seed=5,
    stim_dim=2,
    stim_seed=5,
    temporal=False,
    device="cpu",
):
    assert pct_lesion >= 0 and pct_lesion <= 1, "pct_lesion must be in [0,1]"

    cfg = get_config(coadapt=False, cuda=device, obs_out_dim=DEFAULT_OBS_DIM,
                     num_stim_channels=stim_dim, stim_sigma=DEFAULT_STIM_SIGMA,
                     obs_sigma=DEFAULT_OBS_SIGMA)
    brain = RecurrentSAC(
        input_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        michaels=True,
        cfg=cfg,
    )
    brain.actor_summarizer.rnn.stimulus._decay = 1
    if temporal: brain.actor_summarizer.rnn.stimulus._decay = .3

    brain.load_actor(brain_path)
    brain.load_q(brain_path)

    if pct_lesion > 0:
        if set_predefined_lesion:
            if os.path.exists(lesion_path):
                lesion = dill.load(file=open(lesion_path, "rb"))
                lesion.lesion_mask = lesion.lesion_mask.to(device)
            else:
                lesion = experiment.lesion.LesionOutputs(
                    brain.actor_summarizer.rnn.num_neurons_per_module,
                    region, pct_lesion, cuda=device,
                )
                lesion_dir = os.path.dirname(lesion_path)
                os.makedirs(lesion_dir, exist_ok=True)
                print(f"New lesion created, saving to path {lesion_path}")
                dill.dump(lesion, file=open(lesion_path, "wb"))
        else:
            lesion = experiment.lesion.LesionOutputs(
                brain.actor_summarizer.rnn.num_neurons_per_module,
                region, pct_lesion, cuda=device,
            )
        brain.actor_summarizer.rnn.set_lesion(lesion)
    return brain


def get_sac_brain(
    env,
    brain_path,
    pct_lesion=0.0,
    injury_seed=5,
    stim_dim=None,
    stim_seed=5,
    device="cpu",
):
    assert pct_lesion >= 0 and pct_lesion <= 1, "pct_lesion must be in [0,1]"

    brain = InjuredSAC.load(brain_path)
    brain_size = brain.policy.net_arch
    indices = list(itertools.product(range(brain_size[0]), range(brain_size[1])))

    np.random.seed(injury_seed)
    np.random.shuffle(indices)
    burns = np.array(indices)[:int(pct_lesion * brain_size[0] * brain_size[1])]
    b = brain.policy.actor.latent_pi[2].weight.detach()
    for i,j in burns:
        b[i,j] = 0
    brain.policy.actor.latent_pi[2].weight = torch.nn.Parameter(b)

    if stim_dim is not None:
        np.random.seed(stim_seed)
        shuffled_neurons = np.random.permutation(range(brain_size[1]))
        brain.policy.actor.stimulations = np.sort(shuffled_neurons[:stim_dim])
    brain.policy.to(device)
    return brain
