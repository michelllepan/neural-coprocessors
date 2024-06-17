# NOTE: this is an old file

import logging

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)


import torch
from torch.optim import AdamW

# Local imports
import cpn_model
import stats
import stim_model
import utils


RECENT_EN = None


def unroll(cpn, mike, en, din, trial_end, observer, retain_stim_grads=False, cuda=None):
    """
    This runs a trial on a batch.
    Args:
        cpn: a CPN network (a torch Module)
        mike: a Michaels modular RNN (a torch Module)
        en: an EN network (a torch Module)
        din: batch of inputs to mike (i.e. VGGNet features)
             tensor (batch, time, feat_dim)
        trial_end: tensor with 1/0 indicating if the current time step
                   is beyond the trial end. This is how we handle trials
                   of varying lengths.
        observer: and observer.Observer. Used to observe mike's activity.
        retain_stim_grads: bool indicating we should keep stim vector gradients
        cuda: something that can be passed to a tensor.cuda()
    """

    batch_size = din.shape[0]
    steps = din.shape[1]

    stims = []
    preds = torch.zeros(batch_size, steps - 1, en.out_dim)
    actuals = torch.zeros(batch_size, steps - 1, en.out_dim)

    if cuda is not None:
        preds = preds.cuda(cuda)
        actuals = actuals.cuda(cuda)

    for tidx in range(steps - 1):
        obs = mike.observe(observer)
        new_obs_cpn = torch.cat(obs, axis=1).detach()
        new_obs_en = obs[0].detach()

        # cpn recieves (obs, trial_end)
        cpn_in = torch.cat((new_obs_cpn, trial_end[:, tidx, :]), axis=1)
        # output is (batch_size, num_stim_channels)
        new_stim = cpn(cpn_in)
        assert new_stim.shape == (batch_size, mike.stimulus.num_stim_channels)
        if retain_stim_grads:
            new_stim.retain_grad()
        stims.append(new_stim)

        # en receives (obs, stims, trial_end)
        en_in = torch.cat((new_obs_en, new_stim, trial_end[:, tidx, :]), axis=1)
        cur_pred = en(en_in)
        preds[:, tidx, :] = cur_pred[:, :]

        # new_stim will be cloned in here, to prevent accidentally backprop-ing
        # through the "brain", aka mike.
        mike.stimulate(new_stim)

        # Note that 'preds' lags 'actual' by a time step, hence
        # 'pred' is a prediction of the actual en activity
        mike_out = mike(din[:, tidx + 1, :].T)
        actuals[:, tidx, :] = mike_out[:, :]

    actuals = utils.trunc_to_trial_end(actuals, trial_end[:, :-1, :])
    preds = utils.trunc_to_trial_end(preds, trial_end[:, :-1, :])
    return actuals, preds, stims


def train_loop(
    cpn,
    mike,
    en,
    din,
    dout,
    trial_end,
    labels,
    observer,
    loss_history,
    epoch_type,
    is_val=False,
    retain_stim_grads=False,
    cuda=None,
):

    cpn.reset()
    mike.reset()
    en.reset()
    actuals, preds, stims = unroll(
        cpn,
        mike,
        en,
        din,
        trial_end,
        observer,
        retain_stim_grads=retain_stim_grads,
        cuda=cuda,
    )

    if is_val:
        loss_history.report_val_last_result(actuals, preds, dout)
    else:
        loss_history.report_by_result(epoch_type, actuals, preds, dout, labels)

    return actuals, preds, stims


def train_en(
    mike,
    observer,
    cpn,
    data_loader,
    loss_history,
    en=None,
    opt_en=None,
    en_num_neurons=None,
    cuda=None,
):
    """
    mike: a Michaels modular RNN (a torch Module)
    observer: and observer.Observer. Used to observe mike's activity.
    cpn: a CPN network (a torch Module)
    data_loader: a DataLoader which contains the training data we are
                 using.
    loss_history: a stats.LossHistory
    cuda: something that can be passed to a tensor.cuda()
    """
    # the last EN we were working on training, for debugging
    global RECENT_EN

    obs_dim = observer.out_dim * 1
    # Stim: mike.stimulus.num_stim_channels
    # +1 for trial_end
    en_in_dim = obs_dim + mike.stimulus.num_stim_channels + 1

    if en is None:
        en = stim_model.StimModelLSTM(
            en_in_dim,
            mike.output_dim,
            num_neurons=en_num_neurons or (en_in_dim + 50),
            activation_func=torch.nn.Tanh,
            cuda=cuda,
        )

        assert opt_en is None

        opt_en = AdamW(en.parameters(), lr=9e-3, weight_decay=0.04)

    RECENT_EN = en
    vl = torch.tensor(1.0)

    checkpoint_eidx = 0
    eidx = -1
    while True:
        for batch in data_loader:
            din, trial_end, _, dout, labels = batch
            eidx += 1
            batch_size = din.shape[0]
            opt_en.zero_grad()

            remaining_loss = loss_history.recent_train_loss
            if remaining_loss is None:
                # Just some high-ish number for the first time
                # through this function.
                remaining_loss = 0.05
            else:
                remaining_loss = remaining_loss.item()

            # Silly lr schedule; basically works
            for p in opt_en.param_groups:
                if vl.item() < 0.0007:
                    p["lr"] = 1e-4
                elif vl.item() < 0.005:
                    p["lr"] = 3e-3
                else:
                    p["lr"] = 4e-3

            cpn.reset()
            cpn_noise = cpn_model.CPNNoiseyLSTMCollection(
                cpn,
                noise_var=2 * remaining_loss,
                white_noise_pct=0.3,
                white_noise_var=6,
                cuda=cuda,
            )
            cpn_noise.setup(batch_size)

            train_loop(
                cpn_noise,
                mike,
                en,
                din,
                dout,
                trial_end,
                labels,
                observer,
                loss_history,
                stats.LossRecType.EN,
                retain_stim_grads=False,
                cuda=cuda,
            )

            # Update en
            rl = loss_history.recent_pred_loss
            rl.backward()
            opt_en.step()

            # Verify against the actual CPN
            if (eidx % 10) == 0:
                train_loop(
                    cpn,
                    mike,
                    en,
                    din,
                    dout,
                    trial_end,
                    labels,
                    observer,
                    loss_history,
                    stats.LossRecType.EN,
                    retain_stim_grads=False,
                    is_val=True,
                    cuda=cuda,
                )
                vl = loss_history.recent_pred_val_loss

            loss_history.log(logger, "training en:")

            if (
                torch.isnan(vl)
                or torch.isinf(vl)
                or vl.item() > 1.5
                or (eidx - checkpoint_eidx) > 5000
            ):
                en = stim_model.StimModelLSTM(
                    en.in_dim,
                    en.out_dim,
                    num_neurons=en.num_neurons,
                    activation_func=en.activation_func_t,
                    cuda=cuda,
                )

                RECENT_EN = en
                opt_en = AdamW(en.parameters(), lr=1e-3, weight_decay=0.04)
                checkpoint_eidx = eidx

            if (vl.item() < max(0.02 * remaining_loss, 0.0003) and eidx > 200) or (
                eidx - checkpoint_eidx
            ) == 2000:
                done = True
                break
            else:
                done = False

        if done:
            break

    opt_en.zero_grad()
    return en, opt_en


def refine_en(
    cpn,
    mike,
    en,
    opt_en,
    data_loader,
    observer,
    loss_history,
    cuda=None,
):

    for p in opt_en.param_groups:
        p["lr"] = 1e-4

    batch_size = din.shape[0]

    while loss_history.recent_pred_loss > max(loss_history.recent_task_loss / 10, 6e-4):
        for batch in data_loader:

            din, trial_end, _, dout, labels = batch

            cpn.reset()
            cpn_noise = cpn_model.CPNNoiseyLSTMCollection(
                cpn,
                noise_var=0.002,
                white_noise_pct=0.3,
                white_noise_var=6,
                cuda=cuda,
            )
            cpn_noise.setup(batch_size)

            opt_en.zero_grad()
            _ = train_loop(
                cpn,
                mike,
                en,
                din,
                dout,
                trial_end,
                labels,
                observer,
                loss_history,
                stats.LossRecType.EN,
                cuda=cuda,
            )

            pred_loss = loss_history.recent_pred_loss
            pred_loss.backward(inputs=list(en.parameters()))

            loss_history.log(logger, "adjusting en:")

            opt_en.step()
