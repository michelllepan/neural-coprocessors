import enum

import attr
import torch

from experiment import stats
from experiment.utils import render_none_or_float


class EpochType(enum.Enum):
    EN = 0
    EN_OFFLINE = 1
    CPN = 2
    CPN_OFFLINE = 3
    CPN_AND_EN = 4
    VAL_EN = 5
    VAL_CPN = 6


@attr.s(auto_attribs=True)
class CPNENStats(stats.UserData):
    epoch_type: EpochType
    train_loss: float
    train_val_loss: float
    pred_loss: float
    pred_val_loss: float

    def render(self):
        return {
            "epoch_type": self.epoch_type.name,
            "train_loss": render_none_or_float(self.train_loss),
            "pred_loss": render_none_or_float(self.pred_loss),
            "pred_val_loss": render_none_or_float(self.pred_val_loss),
        }


_G_MSELOSS = torch.nn.MSELoss()


def calc_pred_loss(preds, actuals):
    return _G_MSELOSS(preds, actuals)


def calc_train_loss(preds, targets):
    return _G_MSELOSS(preds, targets[:, 1:, :])
