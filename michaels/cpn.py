import logging
import os
import json
import uuid

import torch
from torch.optim import AdamW


import cpn_model
import cpn_epoch_cpn
import cpn_epoch_en
from cpn_utils import EpochType
from experiment import experiment
from experiment import observer
import stim_model

MODEL_FILENAME_CPN = "cpn.model"
MODEL_FILENAME_EN = "en.model"
MODEL_FILENAME_MIKE = "mike.model"
MODEL_FILENAME_OBS = "obs.model"

g_logger = logging.getLogger("cpn")


class CPN_EN_CoProc(experiment.CoProc):
    def __init__(self, cfg, log_dir=None, recycle_thresh=None):
        self.cfg = cfg

        in_dim, stim_dim, out_dim, cuda = cfg.unpack()

        self.uuid = uuid.uuid4()
        if log_dir is not None:
            if cfg.dont_train:
                self.log_dir = os.path.join(log_dir, "phase2")
                os.makedirs(self.log_dir)
            else:
                self.log_dir = os.path.join(log_dir, cfg.cfg_str, str(self.uuid))
                os.makedirs(self.log_dir)

            print("Log dir will be:", self.log_dir)
        else:
            self.log_dir = None

        self.cpn = cpn_model.CPNModelLSTM(
            in_dim,
            stim_dim,
            num_neurons=in_dim,
            activation_func=cfg.cpn_activation,
            cuda=cuda,
        )

        if cfg.recover_after_lesion:
            self.opt_cpn = AdamW(self.cpn.parameters(), lr=4e-3)
        else:
            self.opt_cpn = AdamW(self.cpn.parameters(), lr=1e-3)

        # We are not training the coproc, but the obs function has drifting bias.
        # Load current bias into the obs instance.
        if cfg.dont_train and cfg.drifting_obs:
            assert isinstance(
                cfg.observer_instance, observer.ObserverGaussian1dDrifting
            )
            assert log_dir is not None
            obs_path = os.path.join(log_dir, MODEL_FILENAME_OBS)
            print("obs_path:", obs_path)
            obs_state = torch.load(obs_path)

            obs_bias = obs_state["bias"].cuda(cuda)
            obs_means = obs_state["means"].cuda(cuda)
            obs_stdevs = obs_state["stdevs"].cuda(cuda)

            cfg.observer_instance.bias = obs_bias
            cfg.observer_instance.means = obs_means
            cfg.observer_instance.stdevs = obs_stdevs
            print("Observer loaded as:", str(cfg.observer_instance))

        en_in_dim = cfg.observer_instance.out_dim + stim_dim + 1
        self.en, self.opt_en = stim_model.get_stim_model(
            en_in_dim, out_dim, activation=cfg.en_activation, cuda=cfg.cuda
        )

        self.stims = None
        self.brain_data = None

        self.epoch_type = EpochType.EN
        for param in self.cpn.parameters():
            param.requires_grad = False
        for param in self.en.parameters():
            param.requires_grad = True

        self.en_epoch = cpn_epoch_en.CPNEpochEN(
            self.en, self.opt_en, self.cpn, self.opt_cpn, self.cfg
        )

        # If we aren't training, it means we trained already. At
        # lease that is the interface for now. It also means we
        # are loading an existing cpn.model from a fully qualified
        # log dir path. Again - just a very assumption heavy
        # cfg.
        if cfg.dont_train:
            cpn_model_path = os.path.join(log_dir, MODEL_FILENAME_CPN)
        else:
            cpn_model_path = None

        self.cpn_epoch = cpn_epoch_cpn.CPNEpochCPN(
            self.en,
            self.opt_en,
            self.cpn,
            self.opt_cpn,
            self.cfg,
            model_path=cpn_model_path,
        )

        self.recycle_thresh = recycle_thresh
        # A list of up to recycle_thresh length
        #  Each item is a tuple (actuals, targets, trial_end, stim list, obs list)
        #   The list is trial_len length.
        #     Each item is a stim or obs
        self.saved_data = []

        self.reset_soft()

    def reset_soft(self):
        self.stims = []
        self.brain_data = []
        self.en_epoch.reset()
        self.cpn_epoch.reset()

    def reset(self):
        self.cpn_epoch.reset_period()

        if self.cfg.dont_train:
            self.epoch_type = EpochType.CPN
        else:
            self.epoch_type = EpochType.EN

        self.en, self.opt_en = self.en_epoch.reset_en()
        self.cpn_epoch.set_en(self.en, self.opt_en)

        if self.cfg.dont_train:
            for param in self.cpn.parameters():
                param.requires_grad = False
            for param in self.en.parameters():
                param.requires_grad = False
        else:
            for param in self.cpn.parameters():
                param.requires_grad = False
            for param in self.en.parameters():
                param.requires_grad = True

        self.reset_soft()

    def forward(self, brain_data, loss_history):
        if self.epoch_type in (EpochType.EN, EpochType.VAL_EN):
            new_stim = self.en_epoch.forward(
                brain_data, loss_history, self.epoch_type == EpochType.VAL_EN
            )
        elif self.epoch_type in (EpochType.CPN, EpochType.VAL_CPN):
            new_stim = self.cpn_epoch.forward(
                brain_data, loss_history, self.epoch_type == EpochType.VAL_CPN
            )
        else:
            # For now: no other thing
            raise ValueError(self.epoch_type)

        self.stims.append(new_stim.detach())
        self.brain_data.append(brain_data)

        return new_stim

    def feedback(self, actuals, targets, trial_end, loss_history):
        if self.epoch_type in (EpochType.EN, EpochType.VAL_EN):
            self.en_epoch.feedback(
                actuals,
                targets,
                trial_end,
                loss_history,
                is_validation=self.epoch_type == EpochType.VAL_EN,
            )
            update_task_loss = False

            if self.epoch_type == EpochType.EN and self.recycle_thresh:
                new_save = (actuals, targets, trial_end, self.stims, self.brain_data)
                self.saved_data.append(new_save)

        elif self.epoch_type in (EpochType.CPN, EpochType.VAL_CPN):
            self.cpn_epoch.feedback(
                actuals,
                targets,
                trial_end,
                loss_history,
                is_validation=self.epoch_type == EpochType.VAL_CPN,
            )
            update_task_loss = True

        return update_task_loss

    def train_en_closed_loop(self, loss_history, user_data, next_is_validation):
        # Unpack, aka form a single 'actuals', 'targets', 'trial_end',
        #  and pack a list of concatenated stims.

        trial_len = len(self.saved_data[0][4])

        is_validation = next_is_validation

        last_pred_loss = 1
        if user_data is not None:
            if user_data.pred_loss == user_data.pred_loss:
                last_pred_loss = user_data.pred_loss
        if last_pred_loss > 0.005:
            self.saved_data = []
            return False

        checkpoint_eidx = 0
        en_is_ready = False
        while not en_is_ready and checkpoint_eidx < 10:
            for bidx in range(self.recycle_thresh):
                actuals, targets, trial_end, stims, brain_data = self.saved_data[bidx]

                for tidx in range(trial_len):
                    bd = brain_data[tidx]
                    stim = stims[tidx]

                    self.en_epoch.forward(
                        bd, loss_history, is_validation=is_validation, stim=stim
                    )

                self.en_epoch.feedback(
                    actuals, targets, trial_end, loss_history, is_validation
                )

                if is_validation:
                    loss_history.report_val_last_result(
                        actuals,
                        targets,
                        update_task_loss=False,
                    )
                else:
                    loss_history.report_by_result(
                        actuals,
                        targets,
                        labels=None,
                        update_task_loss=False,
                    )

                (
                    self.en,
                    self.opt_en,
                    next_is_validation,
                    en_is_ready,
                    user_data,
                ) = self.en_epoch.finish(loss_history, is_validation, reused_data=True)

                loss_history.report_user_data(user_data)
                loss_history.log(g_logger, msg=user_data.msg)

                is_validation = next_is_validation

                if en_is_ready:
                    break

                checkpoint_eidx += 1

            if en_is_ready:
                break

        self.saved_data = []
        return en_is_ready

    def finish(self, loss_history):
        if self.epoch_type in (EpochType.EN, EpochType.VAL_EN):
            (
                self.en,
                self.opt_en,
                next_is_validation,
                en_is_ready,
                user_data,
            ) = self.en_epoch.finish(loss_history, self.epoch_type == EpochType.VAL_EN)

            if en_is_ready:
                self.epoch_type = EpochType.CPN
                self.cpn_epoch.set_en(self.en, self.opt_en)

                for param in self.cpn.parameters():
                    param.requires_grad = True
                for param in self.en.parameters():
                    param.requires_grad = False

                self.saved_data = []

            elif next_is_validation:
                self.epoch_type = EpochType.VAL_EN
            else:
                self.epoch_type = EpochType.EN

            if len(self.saved_data) == self.recycle_thresh and not en_is_ready:
                en_is_ready = self.train_en_closed_loop(
                    loss_history, user_data, next_is_validation
                )

                if en_is_ready:
                    self.epoch_type = EpochType.CPN
                    self.cpn_epoch.set_en(self.en, self.opt_en)

                    for param in self.cpn.parameters():
                        param.requires_grad = True
                    for param in self.en.parameters():
                        param.requires_grad = False
                else:
                    self.epoch_type = EpochType.EN

            we_are_done = False

        else:
            (
                we_are_done,
                next_is_validation,
                en_is_ready,
                user_data,
            ) = self.cpn_epoch.finish(loss_history, self.epoch_type == EpochType.VAL_EN)

            if en_is_ready:
                if next_is_validation:
                    self.epoch_type = EpochType.VAL_CPN
                else:
                    self.epoch_type = EpochType.CPN
            else:
                for param in self.cpn.parameters():
                    param.requires_grad = False

                self.en, self.opt_en = self.en_epoch.reset_en()

                self.epoch_type = EpochType.EN
                next_is_validation = False

        result = experiment.EpochResult(
            we_are_done,
            next_is_validation,
            user_data,
        )

        self.reset_soft()
        return result

    def report(self, loss_history, mike):
        if self.log_dir is not None:
            if (loss_history.eidx % 50) == 0:
                log_path = os.path.join(self.log_dir, "log")

                with open(log_path, "w") as f:
                    json.dump(loss_history.render(), f)
                    f.flush()

                cpn_path = os.path.join(self.log_dir, MODEL_FILENAME_CPN)
                en_path = os.path.join(self.log_dir, MODEL_FILENAME_EN)
                mike_path = os.path.join(self.log_dir, MODEL_FILENAME_MIKE)
                obs_path = os.path.join(self.log_dir, MODEL_FILENAME_OBS)

                torch.save(self.cpn.state_dict(), cpn_path)
                torch.save(self.en.state_dict(), en_path)
                torch.save(mike.state_dict(), mike_path)

                if isinstance(
                    self.cfg.observer_instance, observer.ObserverGaussian1dDrifting
                ):
                    out = {
                        "bias": self.cfg.observer_instance.bias.cpu(),
                        "means": self.cfg.observer_instance.means.cpu(),
                        "stdevs": self.cfg.observer_instance.stdevs.cpu(),
                    }
                    torch.save(out, obs_path)

                torch.save(mike.state_dict(), mike_path)
