import torch

from cpn_utils import CPNENStats, calc_pred_loss, calc_train_loss, EpochType
from experiment import utils


class CPNEpochCPN:
    def __init__(self, en, opt_en, cpn, opt_cpn, cfg, model_path=None):
        self.en = en
        self.opt_en = opt_en

        self.cpn = cpn
        self.opt_cpn = opt_cpn
        for p in opt_cpn.param_groups:
            p["lr"] = 1e-3

        if model_path is not None:
            cpn.load_state_dict(torch.load(model_path))

        self.cfg = cfg

        self.preds = None
        self.tidx = 0

        # This tracks how many epochs we've been training the CPN
        self.checkpoint_eidx = 0

        self.recent_task_losses = []

        self.recent_train_loss = 0.05
        self.recent_pred_loss = 0.05
        self.recent_train_val_loss = 0.05

        # Weight for stim regularizer
        # self.reg_stim_weight = 2e-8
        self.reg_stim_weight = None
        self.stims = []

        if cfg.recover_after_lesion:
            self.set_opt_lr = self.lr_sched_aggressive_refine3
        else:
            self.set_opt_lr = self.lr_sched_standard

        self.reset()

    def reset(self):
        self.preds = None
        self.tidx = 0
        self.stims = []
        self.reset_models()

    def reset_models(self):
        self.en.reset()
        self.cpn.reset()
        self.opt_en.zero_grad()
        self.opt_cpn.zero_grad()

    def reset_period(self):
        self.checkpoint_eidx = 0
        self.recent_task_losses = []

    def set_en(self, en, opt_en):
        self.en = en
        self.opt_en = opt_en
        self.en.reset()
        self.opt_en.zero_grad()

    def ensure_preds(self, batch_size):
        if self.preds is None:
            self.preds = []

    def forward(self, brain_data, loss_history, is_validation):
        batch_size = brain_data[0].shape[0]
        self.ensure_preds(batch_size)

        cpn_in = torch.cat(brain_data, axis=1)

        new_stim = self.cpn(cpn_in)

        # en receives (obs, stims, trial_end)
        new_obs_en = brain_data[0]
        en_in = torch.cat((new_obs_en, new_stim, brain_data[-1]), axis=1)
        cur_pred = self.en(en_in)

        self.preds.append(cur_pred.unsqueeze(dim=1))

        self.tidx += 1
        self.stims.append(new_stim)

        return new_stim

    def feedback(self, actuals, targets, trial_end, loss_history, is_validation):

        preds = torch.cat(self.preds, axis=1)
        preds = utils.trunc_to_trial_end(preds, trial_end[:, :-1, :])

        pred_loss = calc_pred_loss(preds, actuals)
        train_loss = calc_train_loss(preds, targets)

        if self.reg_stim_weight is not None:
            # Regularization for stimulation applied
            train_loss += self.reg_stim_weight * sum(
                [torch.linalg.norm(s) for s in self.stims]
            )

        if is_validation:
            self.recent_train_val_loss = train_loss.item()
        else:
            self.recent_train_loss = train_loss.item()
            if not self.cfg.dont_train:
                train_loss.backward(inputs=list(self.cpn.parameters()))

        self.recent_pred_loss = pred_loss.item()

    def lr_sched_aggressive_refine(self, rtl, eidx):
        """
        Args:
            rtl - recent training loss, which we use to determine the learning rate
        """
        if rtl is None or rtl >= 0.005:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-3
        elif rtl >= 0.004:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-4
        elif rtl >= 0.002:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 2e-6
        else:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-6

    def lr_sched_aggressive_refine2(self, rtl, eidx):
        """
        Args:
            rtl - recent training loss, which we use to determine the learning rate
        """
        if rtl is None or rtl >= 0.004:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 5e-4
        elif rtl >= 0.003:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-5
        elif rtl >= 0.002:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 2e-6
        else:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-6

    def lr_sched_aggressive_refine3(self, rtl, eidx):
        """
        Seems to work for M1 co-adapt.
        Args:
            rtl - recent training loss, which we use to determine the learning rate
        """
        if rtl is None or eidx < 4000:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-3
        elif rtl >= 0.008:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-3
        elif rtl >= 0.006:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 5e-4
        elif rtl >= 0.005:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-5
        elif rtl >= 0.004:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 2e-6
        else:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-6

    def lr_sched_standard(self, rtl, eidx):
        """
        Args:
            rtl - recent training loss, which we use to determine the learning rate
        """
        if rtl is None or rtl >= 0.008:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-3
        elif rtl >= 0.006:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 5e-5
        elif rtl >= 0.002:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 2e-6
        else:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-6

    def finish(self, loss_history, is_validation):

        rtl = self.recent_train_loss
        self.set_opt_lr(rtl, loss_history.eidx)

        # Every 10 epochs let's validate/test
        next_is_validation = not is_validation and (loss_history.eidx % 10) == 0

        last_rec = loss_history.get_recent_record(-2)
        pred_val_loss_out = float("nan")
        if last_rec is not None:
            last_user_data = last_rec.user_data
            if last_user_data is not None:
                pred_val_loss_out = last_user_data.pred_val_loss

        rtl = loss_history.recent_task_loss
        self.recent_task_losses.append(rtl)
        # if (loss_history.max_pct_recov > 0.99 and not self.cfg.dont_train and
        if (
            loss_history.max_pct_recov > 0.9
            and not self.cfg.dont_train
            and loss_history.lesioned_loss > loss_history.healthy_loss
        ):
            # The pct recov makes sense only in the typical case that lesions cause
            # worse performance, due to the way it's calculated. This is always the case
            # unless we are using a pre-recovered model.
            we_are_done = True
            en_is_ready = False
        # For now: just run for awhile
        elif self.cfg.dont_train and loss_history.eidx == 200000:
            we_are_done = True
            en_is_ready = False
        else:
            we_are_done = False

            if self.cfg.dont_train:
                en_is_ready = True
                self.reset_period()
            elif self.recent_pred_loss > max(rtl / 10, 6e-4):
                en_is_ready = False
                self.reset_period()
            # elif self.checkpoint_eidx >= 200:
            elif self.checkpoint_eidx >= 100:
                en_is_ready = False
                self.reset_period()
            elif self.checkpoint_eidx > 30:
                num_reg = 0
                for l in self.recent_task_losses[-30:]:
                    if l < rtl:
                        num_reg += 1

                if num_reg > 15:
                    en_is_ready = False
                    self.reset_period()
                else:
                    en_is_ready = True
                    self.opt_cpn.step()
                    self.checkpoint_eidx += 1
            else:
                en_is_ready = True
                self.opt_cpn.step()
                self.checkpoint_eidx += 1

        user_data = CPNENStats(
            "cpn",
            EpochType.CPN,
            self.recent_train_loss,
            self.recent_train_val_loss,
            self.recent_pred_loss,
            pred_val_loss_out,
        )

        self.reset()
        return we_are_done, next_is_validation, en_is_ready, user_data
