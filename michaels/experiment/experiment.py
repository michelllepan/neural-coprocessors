import errno
import logging
import os

import attr
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
torch.manual_seed(0)


# Local imports
from . import class_stdevs
from . import config
from . import lesion
from . import stats
from . import michaels_load
from . import mRNN
from . import utils

g_logger = logging.getLogger("experiment")

# Rel path pointing to a pre-recovered mRNN
RECOV_MODEL_PATH = "experiment/recovered"


@attr.s(auto_attribs=True)
class EpochResult:
    stop: bool = False
    next_is_validation: bool = False
    user_data: stats.UserData = None

    def unpack(self):
        return self.stop, self.next_is_validation, self.user_data


def get_vanilla_config(cuda=None, **kwargs):
    return get_config(cuda=cuda, **kwargs)


def get_coadapt_config(cuda=None, **kwargs):
    return get_config(cuda=cuda, coadapt=True, **kwargs)


def get_m1_lesion_config(cuda=None, coadapt=True, **kwargs):
    lesion_type = lesion.LesionType.outputs
    # Lesion the neurons with idxs 0-50, which are in M1
    lesion_args = (0, 50)
    return get_config(
        cuda=cuda,
        coadapt=coadapt,
        drop_m1=True,
        lesion_type=lesion_type,
        lesion_args=lesion_args,
        num_stim_neurons=50,
        stim_pad_left_neurons=50,
        stim_pad_right_neurons=200,
        **kwargs,
    )


def get_aip_lesion_config(cuda=None, coadapt=True, **kwargs):
    lesion_type = lesion.LesionType.outputs
    # Lesion the neurons with idxs 200-250, which are in AIP
    lesion_args = (200, 250)

    return get_config(
        cuda=cuda,
        coadapt=coadapt,
        drop_m1=True,
        lesion_type=lesion_type,
        lesion_args=lesion_args,
        **kwargs,
    )


def get_config(
    recover_after_lesion=False,
    coadapt=False,
    dont_train=False,
    num_stim_neurons=None,
    stim_pad_right_neurons=config.DEFAULT_STIM_PAD_RIGHT_NEURONS,
    cuda=None,
    **kwargs,
):
    """
    Args:
        - cuda (str, torch.device, or None): None for CPU, or a string like "0" specifying a GPU
                              device. This will be passed to the CUDA_VISIBLE_DEVICES env var.
                              Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
    Returns:
        A config.Config, which contains the parameterization of this experiment.
    """
    if cuda is None or cuda=='cpu':
        cuda_out = None
    elif isinstance(cuda, str):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        cuda_out = torch.device(0)

    else:
        cuda_out = cuda
    cfg = config.get(
        recover_after_lesion=recover_after_lesion,
        num_stim_neurons=num_stim_neurons,
        stim_pad_right_neurons=stim_pad_right_neurons,
        coadapt=coadapt,
        dont_train=dont_train,
        cuda=cuda_out,
        **kwargs,
    )

    return cfg


def get_raw_data(cuda=None, **kwargs):
    # Just a passthrough, since 'experiment' is our simple interface
    return config.get_data_raw(cuda=cuda ** kwargs)


class CoProc:
    def reset(self):
        """
        Called at the start and continuation of a run.
        """
        pass

    def forward(self, brain_data, loss_history):
        """
        Args:
            brain_data: 4-tuple (M1 data, F5 data, AIP data, trial_end signal)
                        The first three are Tensors (batch_size, cfg.observer_instance.out_dim)
                        The last is (batch_size, 1)
                        The sum of second dimensions is in_dim.
        Returns:
                the stim vector: Tensor(batch_size, stim_dim)
        """
        raise NotImplementedError()

    def feedback(self, actuals, targets, loss_history):
        """
        Args:
            actuals: torch.Tensor(batch_size, out_dim)
            targets: torch.Tensor(batch_size, out_dim)
            loss_history: a stats.LossHistory
        Returns:
            True if we should update the task loss; False if we should just
                carry it over from the prior epoch
        """
        raise NotImplementedError()

    def finish(self, loss_history):
        """
        Called between batches. User can add any cleanup logic here, e.g.
        resetting internal state of RNNs.
        Returns:
            A tuple: (
                True to stop training; False otherwise,
                True to update task loss log based on this epoch,
                True if this is a validation epoch; report it as such,
                A UserData)
            If it's hard to remember the order of those, you may find
            it easier to remember by instead returning a
            EpochResult (above).
        """
        pass

    def report(self, loss_history, mike):
        """
        Final chance to report/calc stats, records, logs, etc., after finish.
        This step is separate from finish(), since the most recent loss history
        record now holds the user data from finish().
        """
        pass


def stage(coproc, cfg, mike_load_path=None):
    return Experiment(coproc, cfg, mike_load_path=mike_load_path)


class Experiment:
    def __init__(self, coproc, cfg, mike_load_path=None):
        self._coproc = coproc

        self._cfg = cfg

        mike = mRNN.MichaelsRNN(
            init_data_path=michaels_load.get_default_path(),
            stimulus=cfg.stim_instance,
            cuda=cfg.cuda,
        )
        mike.set_lesion(cfg.lesion_instance)

        self.mike = mike

        if cfg.recover_after_lesion:
            model_path = os.path.join(
                RECOV_MODEL_PATH, "recovered_mrnn_%s.model" % str(cfg.lesion_instance)
            )

            try:
                self.mike.load_weights_from_file(model_path)
            except OSError as e:
                if e.errno == errno.ENOENT:
                    raise ValueError(
                        "No pre-generated recovered mRNN available "
                        "for a lesion of type %s; looked for path %s"
                        % (str(cfg.lesion_instance), model_path)
                    )
                raise

        elif mike_load_path is not None:
            self.mike.load_weights_from_file(mike_load_path)

        if self.cfg.coadapt:
            for param in self.mike.parameters():
                param.requires_grad = True
        else:
            for param in self.mike.parameters():
                param.requires_grad = False
        self.opt_mike = AdamW(self.mike.parameters(), lr=1e-7)

        self.observer = cfg.observer_instance

        (
            self.comp_loss_healthy,
            self.comp_loss_healthy_hand,
            self.comp_loss_lesioned,
            self.comp_loss_lesioned_hand,
            self.var_whole_healthy,
            self.var_within_healthy,
        ) = self._get_healthy_vs_lesioned_stats()

        self.loss_history = stats.LossHistory(
            self.comp_loss_lesioned.item(),
            self.comp_loss_lesioned_hand.item(),
            self.comp_loss_healthy.item(),
            self.comp_loss_healthy_hand.item(),
            self.var_whole_healthy,
            self.var_within_healthy,
        )

        if cfg.drop_m1:
            self.obs_drop_module_idx = 0
        else:
            self.obs_drop_module_idx = None

    @property
    def cfg(self):
        return self._cfg

    def _get_healthy_vs_lesioned_stats(self):
        cuda = self.cfg.cuda
        comp_loss = torch.nn.MSELoss()

        dset = self.cfg.dataset
        dset_size = len(dset)
        dset_samp_len = dset.sample_len

        loader_comp = DataLoader(dset, batch_size=dset_size, shuffle=True)

        for s in loader_comp:
            din, trial_end, _, dout, labels = s

        comp_preds_healthy = torch.zeros((dset_size, dset_samp_len, self.cfg.out_dim))
        if cuda is not None:
            comp_preds_healthy = comp_preds_healthy.to(cuda)

        self.mike.set_lesion(None)
        try:
            self.mike.reset()
            for tidx in range(dout.shape[1]):
                cur_din = din[:, tidx, :].T
                p = self.mike(cur_din)
                comp_preds_healthy[:, tidx, :] = p[:, :]
            comp_preds_healthy = utils.trunc_to_trial_end(comp_preds_healthy, trial_end)
            comp_loss_healthy = comp_loss(comp_preds_healthy, dout)
            comp_loss_healthy_hand = comp_loss(
                comp_preds_healthy[:, :, utils.HAND_MUSCLE_START_IDX :],
                dout[:, :, utils.HAND_MUSCLE_START_IDX :],
            )

        finally:
            self.mike.set_lesion(self.cfg.lesion_instance)

        var_whole, var_within = class_stdevs.calc_class_vars(comp_preds_healthy, labels)

        comp_preds_lesioned = torch.zeros(comp_preds_healthy.shape)
        if cuda is not None:
            comp_preds_lesioned = comp_preds_lesioned.to(cuda)

        self.mike.reset()
        for tidx in range(dout.shape[1]):
            cur_din = din[:, tidx, :].T
            p = self.mike(cur_din)
            comp_preds_lesioned[:, tidx, :] = p[:, :]
        comp_preds_lesioned = utils.trunc_to_trial_end(comp_preds_lesioned, trial_end)
        comp_loss_lesioned = comp_loss(comp_preds_lesioned, dout)
        comp_loss_lesioned_hand = comp_loss(
            comp_preds_lesioned[:, :, utils.HAND_MUSCLE_START_IDX :],
            dout[:, :, utils.HAND_MUSCLE_START_IDX :],
        )

        self.mike.reset()

        return (
            comp_loss_healthy,
            comp_loss_healthy_hand,
            comp_loss_lesioned,
            comp_loss_lesioned_hand,
            var_whole,
            var_within,
        )

    @property
    def coproc(self):
        return self._coproc

    def _coproc_forward(self, brain_data):
        stim = self.coproc.forward(brain_data, self.loss_history)

        # Un-pythonic to assert a type, but let's be strict here...
        if not isinstance(stim, torch.Tensor):
            raise TypeError("Stim vector must be a Tensor")

        expected_stim_shape = (brain_data[0].shape[0], self.cfg.stim_dim)
        if stim.shape != expected_stim_shape:
            raise ValueError(
                f"Expected stim vector to have shape {expected_stim_shape}"
            )

        return stim

    def _coproc_feedback(self, actuals, targets, trial_end, loss_history):
        return self.coproc.feedback(actuals, targets, trial_end, loss_history)

    def _coproc_finish(self, loss_history):
        return self.coproc.finish(loss_history)

    def _coproc_report(self, loss_history, mike):
        return self.coproc.report(loss_history, mike)

    def run(self):
        is_validation = False
        self.coproc.reset()

        while True:
            if is_validation:

                assert (
                    len(self.cfg.loader_test[0].dataset)
                    == self.cfg.loader_test[0].batch_size
                )
                batch = next(iter(self.cfg.loader_test[0]))
            else:
                assert (
                    len(self.cfg.loader_train[0].dataset)
                    == self.cfg.loader_train[0].batch_size
                )
                batch = next(iter(self.cfg.loader_train[0]))

            din, trial_end, _, dout, labels = batch
            steps = din.shape[1]
            self.mike.reset()
            self.opt_mike.zero_grad()

            actuals = []

            mike_out = self.mike(din[:, 0, :].T)

            for tidx in range(steps - 1):
                obs_raw = self.mike.observe(
                    self.observer, drop_module_idx=self.obs_drop_module_idx
                )
                obs = obs_raw + (trial_end[:, tidx, :],)

                stim = self._coproc_forward(obs)

                # new_stim will be cloned in here, to prevent accidentally
                # backprop-ing through the "brain", aka mike.
                self.mike.stimulate(stim)

                mike_out = self.mike(din[:, tidx + 1, :].T)
                actuals.append(mike_out.unsqueeze(dim=1))

            actuals = torch.cat(actuals, axis=1)
            actuals = utils.trunc_to_trial_end(actuals, trial_end[:, :-1, :])

            # Give the user the result
            update_task_loss = self._coproc_feedback(
                actuals, dout, trial_end, self.loss_history
            )

            # Calc losses / stats
            if is_validation:
                self.loss_history.report_val_last_result(
                    actuals,
                    dout,
                    update_task_loss=update_task_loss,
                )
            else:
                self.loss_history.report_by_result(
                    actuals,
                    dout,
                    labels,
                    update_task_loss=update_task_loss,
                )

            # Pass the loss back to the user, who can decide what to
            # do now.
            result = self._coproc_finish(self.loss_history)
            should_stop, next_is_validation, user_data = result.unpack()

            if self.cfg.coadapt and (
                (self.loss_history.eidx % 5) == 0 or self.cfg.dont_train
            ):
                loss = torch.nn.MSELoss()(actuals, dout[:, 1:, :])
                loss.backward(inputs=list(self.mike.parameters()))
                if self.cfg.lesion_instance.application == "connection":
                    self.mike.set_connection_coadap_grads()
                else:
                    self.mike.set_end_to_end_coadap_grads()
                self.opt_mike.step()

            if self.cfg.drifting_obs and ((self.loss_history.eidx % 5) == 0):
                self.observer.step()

            self.loss_history.report_user_data(user_data)

            if user_data is not None:
                msg = user_data.msg
            else:
                msg = None

            self.loss_history.log(g_logger, msg=msg)

            self._coproc_report(self.loss_history, self.mike)

            if should_stop:
                break

            is_validation = next_is_validation

            # self.cfg.shuffle_dataset()

        return self.loss_history
