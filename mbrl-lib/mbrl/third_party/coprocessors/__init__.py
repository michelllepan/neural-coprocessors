import torch

from coprocessors.configs import coproc_config
from coprocessors.utils.setup_env import setup_coproc_env
# from mbrl.third_party.coprocessors.env_utils import *


def make(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = coproc_config(**kwargs)
    coproc_env = setup_coproc_env(config, device, path_prefix="../../../../../..")
    return coproc_env