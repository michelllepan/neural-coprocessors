import enum

import torch

from . import utils


class ActivationType(enum.Enum):
    ReLU = torch.nn.ReLU
    ReTanh = utils.ReTanh
    Tanh = torch.nn.Tanh
    Linear = utils.NonAct
