import copy
import json
import logging

import attr
import torch.nn

from . import class_stdevs
from .utils import render_none_or_float, HAND_MUSCLE_START_IDX

_G_MSELOSS = torch.nn.MSELoss()

g_logger = logging.getLogger("stats")


def calc_task_loss(actuals, targets):
    return _G_MSELOSS(actuals, targets[:, 1:, :])


@attr.s(auto_attribs=True)
class UserData:
    # Just a string which shows up in the logs, for user convenience
    msg: str

    def render(self):
        raise NotImplementedError()


@attr.s(auto_attribs=True)
class LossRec:
    eidx: int
    task_loss: float
    task_loss_hand: float
    task_val_loss: float
    pct_recov: float
    pct_recov_hand: float
    class_separation: float
    user_data: UserData


class LossHistory:
    def __init__(
        self,
        lesioned_loss,
        lesioned_loss_hand,
        healthy_loss,
        healthy_loss_hand,
        var_whole_healthy,
        var_within_healthy,
        adv_stats_report_cadence=7,
    ):
        self._recs = []

        self.lesioned_loss = lesioned_loss
        self.lesioned_loss_hand = lesioned_loss_hand
        self.healthy_loss = healthy_loss
        self.healthy_loss_hand = healthy_loss_hand

        g_logger.info(
            "Healthy loss: %0.8f, Lesioned loss: %0.8f", healthy_loss, lesioned_loss
        )

        self.eidx = 0

        self._healthy_class_separation = var_whole_healthy / var_within_healthy

        self._adv_stats_report_cadence = adv_stats_report_cadence

        self._max_pct_recov = 0

    def calc_pct_recov(self, task_loss):
        dh = task_loss.item() - self.healthy_loss
        dl = self.lesioned_loss - self.healthy_loss
        recov_pct = 1.0 - (dh / dl)
        return recov_pct

    def calc_pct_recov_hand(self, task_loss_hand):
        dh = task_loss_hand.item() - self.healthy_loss_hand
        dl = self.lesioned_loss_hand - self.healthy_loss_hand
        recov_pct = 1.0 - (dh / dl)
        return recov_pct

    def calc_class_vars(self, actuals, labels):
        var_whole, var_within = class_stdevs.calc_class_vars(actuals, labels)
        return var_whole, var_within

    def calc_class_separation(self, actuals, labels):
        if (self.eidx % self._adv_stats_report_cadence) == 0:
            var_whole, var_within = self.calc_class_vars(actuals, labels)
            class_separation = var_whole / var_within
            class_separation -= self._healthy_class_separation
        else:
            class_separation = None

        return class_separation

    def render_rec(self, rec):
        rec_rendered = {
            "eidx": rec.eidx,
            "task_loss": render_none_or_float(rec.task_loss),
            "task_loss_hand": render_none_or_float(rec.task_loss_hand),
            "task_val_loss": render_none_or_float(rec.task_val_loss),
            "pct_recov": render_none_or_float(rec.pct_recov, fmt="%0.3f"),
            "pct_recov_hand": render_none_or_float(rec.pct_recov_hand, fmt="%0.3f"),
            "class_separation": render_none_or_float(rec.class_separation, fmt="%0.3f"),
        }

        if rec.user_data is not None:
            rec_rendered["user"] = rec.user_data.render()

        return rec_rendered

    def render(self):
        rendered = []

        for rec in self._recs:
            rec_rendered = self.render_rec(rec)
            rendered.append(rec_rendered)

        return rendered

    def dump(self):
        return json.dumps(self.render())

    def dump_to_file(self, fname):
        with open(fname, "w") as f:
            json.dump(self.render(), f)

    def report(
        self,
        task_loss,
        task_loss_hand,
        task_val_loss=None,
        class_separation=None,
        user_data=None,
    ):
        if task_val_loss is None:
            if len(self._recs) > 0:
                task_val_loss = self._recs[-1].task_val_loss
            else:
                task_val_loss = float("nan")
        else:
            task_val_loss = task_val_loss.item()

        if class_separation is None:
            if len(self._recs) > 0:
                class_separation = self._recs[-1].class_separation
            else:
                class_separation = float("nan")

        rec = LossRec(
            self.eidx,
            None,
            None,
            task_val_loss,
            None,
            None,
            class_separation,
            user_data,
        )

        if task_loss is not None:
            rec.task_loss = task_loss.item()
            pct_recov = self.calc_pct_recov(task_loss)
            if pct_recov > self._max_pct_recov:
                self._max_pct_recov = pct_recov
            rec.pct_recov = pct_recov
        elif len(self._recs) > 0:
            rec.task_loss = self._recs[-1].task_loss
            rec.pct_recov = self._recs[-1].pct_recov

        if task_loss_hand is not None:
            rec.task_loss_hand = task_loss_hand.item()
            pct_recov_hand = self.calc_pct_recov_hand(task_loss_hand)
            rec.pct_recov_hand = pct_recov_hand
        elif len(self._recs) > 0:
            rec.task_loss_hand = self._recs[-1].task_loss_hand
            rec.pct_recov_hand = self._recs[-1].pct_recov_hand

        self._recs.append(rec)
        self.eidx += 1

    def report_by_result(
        self, actuals, dout, labels, update_task_loss=True, user_data=None
    ):
        if update_task_loss:
            class_separation = self.calc_class_separation(actuals, labels)
            task_loss = calc_task_loss(actuals, dout)
            task_loss_hand = calc_task_loss(
                actuals[:, :, HAND_MUSCLE_START_IDX:],
                dout[:, :, HAND_MUSCLE_START_IDX:],
            )
        else:
            class_separation = None
            task_loss = None
            task_loss_hand = None

        self.report(
            task_loss,
            task_loss_hand,
            class_separation=class_separation,
            user_data=user_data,
        )

    def report_val_last_result(
        self, actuals, dout, update_task_loss=True, user_data=None
    ):
        if update_task_loss:
            task_val_loss = calc_task_loss(actuals, dout)
        else:
            task_val_loss = None

        self.report(None, None, task_val_loss=task_val_loss, user_data=user_data)

    def report_user_data(self, user_data):
        if user_data is not None:
            self.recent_record.user_data = user_data

    @property
    def recent_task_loss(self):
        last_rec = self.recent_record
        if last_rec is not None:
            return last_rec.task_loss
        return None

    @property
    def recent_task_val_loss(self):
        last_rec = self.recent_record
        if last_rec is not None:
            return last_rec.task_val_loss
        return None

    @property
    def recent_record(self):
        try:
            return self._recs[-1]
        except IndexError:
            return None

    def get_recent_record(self, idx):
        try:
            return self._recs[idx]
        except IndexError:
            return None

    @property
    def records(self):
        return copy.copy(self._recs)

    @property
    def max_pct_recov(self):
        return self._max_pct_recov

    def log(self, logger, msg=None):
        recent = self.recent_record

        if recent is not None:
            rendered = self.render_rec(recent)
            del rendered["task_val_loss"]

            eidx = rendered["eidx"]
            del rendered["eidx"]

            out = f"{eidx} " + " ".join([f"{k}: {v}" for k, v in rendered.items()])

            if msg:
                out = msg + " " + out

            logger.info(out)
