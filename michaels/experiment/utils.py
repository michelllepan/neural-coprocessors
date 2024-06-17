import math

import torch
from torch import nn
from scipy.io import loadmat
import scipy.stats
import numpy as np


LOSS_RENDER_FMT = "%0.6f"


def render_none_or_float(val, fmt=LOSS_RENDER_FMT):
    if val is None or val != val:
        return "-"

    return fmt % val


# Identifies hand vs other muscles
# Based on muscle IDs: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4282829/
HAND_MUSCLE_START_IDX = 32  # "FDSL" == Flexor digitorum profundus, Digit 5


def array_weights(in_dim, out_dim, distance_func, normalize=False, **dfunckwargs):
    """
    Returns a matrix used for applying a convolution which represents some sort
    of weighted averaged based on distance. This is used for both observation
    functions and stimulation functions.

    For example, we may have virtual "electrodes" which measure neural activity
    of the neurons in some "area".  The location of the electrode is treated as
    a point on a single spatial dimension, whose cardinality is given by in_dim.
    A Gaussian or other function is applied at the location of every such
    electrode, zero padded, to calculate a weighted average measurement for it.
    The matrix returned by this function calculates the values for every such
    electrode, given the neural activity vector.

    Thus: the return value is a matrix (out_dim, in_dim)
    """
    if out_dim >= in_dim and out_dim > 1:
        raise ValueError(
            "Input dimension (%d) must exceed output dimension "
            "(%d)" % (in_dim, out_dim)
        )

    interval = in_dim / (out_dim + 1)
    centers = np.array([interval * i for i in range(1, out_dim + 1)])

    weights = np.zeros((out_dim, in_dim))
    for c_idx, m in enumerate(centers):
        dists = [i - m for i in range(in_dim)]
        cur_weights = distance_func(dists, **dfunckwargs)
        weights[c_idx, :] = cur_weights

    if normalize:
        # Re-normalize, so each channels's weights sum to 1
        weights = weights / weights.sum(axis=1, keepdims=True)

    return weights


def gaussian_array_weights(in_dim, out_dim, sigma, normalize=False):
    norm = scipy.stats.norm(0, sigma)
    return norm, array_weights(
        in_dim, out_dim, normalize=normalize, distance_func=norm.pdf
    )


def alpha(gmax, t, tau=5, onset=0, time_scalar=1):
    """
    Implementation of the alpha function.
    Inspired by: https://www.sas.upenn.edu/LabManuals/BBB251/NIA/NEUROLAB/HELP/alphasyn.htm

    gmax: (batch_size, num_stim_channels)
    out: (batch_size, num_stim_channels)
    """
    v = gmax * time_scalar * (t - onset) / tau
    v *= np.exp(-1 * time_scalar * (t - onset - tau) / tau)
    return v


def fill_jagged_array(x):
    # Pad arrays with NaNs s.t. all arrays in the list are equal length
    feat_dim = x[0].shape[0]
    max_len = max([a.shape[-1] for a in x])
    make_nan_pad = lambda l: np.full((feat_dim, max_len - l), np.nan)
    x = [np.concatenate([a, make_nan_pad(a.shape[-1])], -1) for a in x]
    x = np.stack(x).transpose(0, 2, 1)
    return torch.tensor(x, dtype=torch.float32, requires_grad=False)


def init_from_michaels_model(
    init_data_path, num_input_features, num_neurons_per_module, output_dim
):
    """
    Loads parameters for a modular RNN and returns weight matrices and
    a Linear model. These can be used to instantiate an mRNN.MichaelsRNN
    """

    data = loadmat(init_data_path)
    data = {k: v for k, v in data.items() if "__" not in k}

    npm = num_neurons_per_module

    J = torch.tensor(data["J"].copy()).float()

    assert data["B"].shape[1] - 1 == num_input_features
    I, S = np.split(data["B"], [num_input_features], axis=1)

    I = torch.tensor(I.copy()).float()
    S = torch.tensor(S.copy()).float()

    B = data["bx"]
    B = torch.tensor(B.copy()).float()

    x0 = data["x0"]
    x0 = torch.tensor(x0.copy()).float().squeeze()

    fc = nn.Linear(npm, output_dim)
    fc.load_state_dict(
        {
            "weight": torch.tensor(data["W"], dtype=torch.float32)[:, :npm],
            "bias": torch.tensor(data["bz"].squeeze(), dtype=torch.float32),
        }
    )

    return J, I, S, B, fc, x0


class ReTanh(nn.Module):
    """
    ReTanh activation function
    """

    def forward(self, x):
        return torch.tanh(torch.clamp(x, min=0))


class NonAct(nn.Module):
    """
    A passthrough/no-op activation function
    """

    def forward(self, x):
        return x


# Adapted from:
#  https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
#  Access 07/27/21
#  *tip of the hat*
# I've unrolled it to keep the interface simple.
class LSTMModel(nn.Module):
    """
    An LSTM implementation that uses only PyTorch primitives, so it's easy to
    tinker with it later (e.g. in CPNNoiseyLSTMCollection).
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        num_neurons=None,
        activation_func=torch.nn.Tanh,
        cuda=None,
    ):
        """
        Args:
            num_neurons: if None, is set to out_dim
            cuda: something that can be passed nn.Module.cuda
            activation_func: should be the class, not an instance. e.g. give
                             torch.nn.Tanh
        """
        super().__init__()

        if num_neurons is None:
            num_neurons = out_dim

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_neurons = num_neurons
        self.W = nn.Parameter(torch.Tensor(in_dim, num_neurons * 4))
        self.U = nn.Parameter(torch.Tensor(num_neurons, num_neurons * 4))
        self.bias = nn.Parameter(torch.Tensor(num_neurons * 4))
        self.init_weights()

        self.activation_func_t = activation_func
        self.activation_func = activation_func()

        self.fc = nn.Linear(num_neurons, out_dim)

        self.ht = None
        self.ct = None

        self._cuda = cuda
        if cuda is not None:
            self.cuda(cuda)

        # Used purely for dropping gradients on the ground.
        #  We do this to reset between learning epochs where
        #  the optimizer/thing we are learning isn't this
        #  network, but we are using this network.
        self._opt = torch.optim.SGD(self.parameters(), lr=1e-3)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.num_neurons)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def detach_inplace(self):
        for p in self.parameters():
            p.detach_()

        if self.ht is not None:
            self.ht.detach_()

        if self.ct is not None:
            self.ct.detach_()

        for p in self.fc.parameters():
            p.detach_()

    def reset(self):
        self.ht = None
        self.ct = None
        self._opt.zero_grad()

    def forward(self, x_t):
        # Old: Assumes x is of shape (batch, sequence, feature)
        # New: Assumes x_t is of shape (batch, feature)

        batch_size, in_dim = x_t.shape
        assert in_dim == self.in_dim, str((in_dim, self.in_dim))

        if self.ht is None:
            self.ht = torch.zeros(batch_size, self.num_neurons)
            self.ct = torch.zeros(batch_size, self.num_neurons)

            if self._cuda is not None:
                self.ht = self.ht.cuda(self._cuda)
                self.ct = self.ct.cuda(self._cuda)

        HS = self.num_neurons
        # batch the computations into a single matrix multiplication
        gates = x_t @ self.W + self.ht @ self.U + self.bias
        i_t, f_t, g_t, o_t = (
            torch.sigmoid(gates[:, :HS]),  # input
            torch.sigmoid(gates[:, HS : HS * 2]),  # forget
            torch.tanh(gates[:, HS * 2 : HS * 3]),
            torch.sigmoid(gates[:, HS * 3 :]),  # output
        )

        self.ct = f_t * self.ct + i_t * g_t
        self.ht = o_t * torch.tanh(self.ct)

        activation = self.activation_func(self.ht)
        out = self.fc(activation)

        return out


def trunc_to_trial_end(data, trial_end):
    """
    Truncates data to its trial end indicator. See mRNN.MichaelsDataset
    """
    return data * trial_end


def loss_regressed(data, window_start=7, window_size=40):
    end = len(data)

    if end < window_start:
        return False

    ews = min(window_size, end)
    ec = ews // 2
    p0 = end - ews
    p1 = end - ec
    e0 = end - ec
    e1 = end

    m1 = sum([cv for cv in data[p0:p1]]) / (p1 - p0)
    m2 = sum([cv for cv in data[e0:e1]]) / (e1 - e0)

    if m1 < m2:
        return True

    return False
