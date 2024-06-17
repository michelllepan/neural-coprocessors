import torch

import cpn_model
from cpn_utils import CPNENStats, calc_pred_loss, EpochType
from experiment import utils
import stim_model


class CPNEpochEN:
    def __init__(self, en, opt_en, cpn, opt_cpn, cfg):
        self.en = en
        self.opt_en = opt_en
        for p in opt_en.param_groups:
            p["lr"] = 4e-3

        self.cpn = cpn
        self.opt_cpn = opt_cpn

        self.cfg = cfg

        self.preds = None
        self.tidx = 0

        # This tracks how many epochs we've been training the EN
        self.checkpoint_eidx = 0

        self.cpn_noise = None

        self.recent_train_loss = 0.05
        self.recent_pred_loss = 0.05
        self.recent_pred_val_loss = 0.05

        self.reset()

    def reset(self):
        self.preds = None
        self.tidx = 0
        self.reset_models()

    def reset_period(self):
        self.checkpoint_eidx = 0

    def reset_models(self):
        self.en.reset()
        self.cpn.reset()

        self.cpn_noise = None

        self.opt_en.zero_grad()
        self.opt_cpn.zero_grad()

    def reset_en(self):
        self.recent_pred_val_loss = 0.05
        self.en, self.opt_en = self.new_en(self.en)

        return self.en, self.opt_en

    def new_en(self, old_en):
        en, opt_en = stim_model.get_stim_model(
            old_en.in_dim,
            old_en.out_dim,
            num_neurons=old_en.num_neurons,
            activation=old_en.activation_func_t,
            cuda=self.cfg.cuda,
        )
        return en, opt_en

    def ensure_noisey_cpn(self, batch_size):
        if self.cpn_noise is None or self.cpn_noise.batch_size != batch_size:
            self.cpn_noise = cpn_model.CPNNoiseyLSTMCollection(
                self.cpn,
                noise_var=1.0 * self.recent_train_loss,
                white_noise_pct=0.3,
                white_noise_var=6,
                cuda=self.cfg.cuda,
            )
            self.cpn_noise.setup(batch_size)

    def ensure_preds(self, batch_size):
        if self.preds is None:
            self.preds = []

    def forward(self, brain_data, loss_history, is_validation, stim=None):
        batch_size = brain_data[0].shape[0]
        self.ensure_preds(batch_size)

        if stim is None:
            cpn_in = torch.cat(brain_data, axis=1).detach()

            if is_validation:
                new_stim = self.cpn(cpn_in)
            else:
                self.ensure_noisey_cpn(batch_size)
                new_stim = self.cpn_noise(cpn_in)
        else:
            new_stim = stim

        # en receives (obs, stims, trial_end)
        # (detaching just in case; we don't want to backprop here)
        new_obs_en = brain_data[0]
        en_in = torch.cat((new_obs_en, new_stim, brain_data[-1]), axis=1)
        cur_pred = self.en(en_in)

        self.preds.append(cur_pred.unsqueeze(dim=1))

        self.tidx += 1

        return new_stim

    def feedback(self, actuals, targets, trial_end, loss_history, is_validation):
        preds = torch.cat(self.preds, axis=1)
        preds = utils.trunc_to_trial_end(preds, trial_end[:, :-1, :])

        pred_loss = calc_pred_loss(preds, actuals)

        if is_validation:
            self.recent_pred_val_loss = pred_loss.item()
        else:
            self.recent_pred_loss = pred_loss.item()

            back_params = list(self.en.parameters())
            pred_loss.backward(inputs=back_params)

            self.opt_en.step()

    def finish(self, loss_history, is_validation, reused_data=False):

        vl = self.recent_pred_val_loss
        for p in self.opt_en.param_groups:
            if vl < 0.0007:
                p["lr"] = 1e-4
            elif vl < 0.005:
                p["lr"] = 3e-3
            else:
                p["lr"] = 4e-3

        last_rec = loss_history.get_recent_record(-2)
        train_loss_out = float("nan")
        train_val_loss_out = float("nan")
        if last_rec is not None:
            last_user_data = last_rec.user_data
            if last_user_data is not None:
                if last_user_data.train_loss == last_user_data.train_loss:
                    self.recent_train_loss = last_user_data.train_loss
                    train_loss_out = last_user_data.train_loss
                train_val_loss_out = last_user_data.train_val_loss

        if vl != vl or vl == float("inf") or vl > 1.5:
            self.en, self.opt_en = self.new_en(self.en)
            en_is_ready = False
            self.reset_period()
        elif (
            vl < max(0.02 * self.recent_train_loss, 0.0003)
            and self.checkpoint_eidx > 100
        ) or self.checkpoint_eidx >= 3000:
            en_is_ready = True
            self.reset_period()
        else:
            self.checkpoint_eidx += 1
            en_is_ready = False

        # Every 10 epochs let's validate/test
        next_is_validation = not en_is_ready and not is_validation and (self.checkpoint_eidx % 10) == 0

        user_data = CPNENStats(
            "en_offline" if reused_data else "en",
            EpochType.EN_OFFLINE if reused_data else EpochType.EN,
            train_loss_out,
            train_val_loss_out,
            self.recent_pred_loss,
            self.recent_pred_val_loss,
        )

        self.reset()
        return self.en, self.opt_en, next_is_validation, en_is_ready, user_data
