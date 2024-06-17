import uuid

import numpy as np
import torch
import torch.autograd
from torch import nn

from experiment import utils


class CPNModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation_func=utils.ReTanh, num_neurons=None):
        super(CPNModel, self).__init__()

        if num_neurons is None:
            num_neurons = in_dim + 10

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_func_t = activation_func

        if activation_func is nn.PReLU:
            self.activation_func = activation_func(num_neurons)
        else:
            self.activation_func = activation_func()

        self.num_neurons = num_neurons

        with torch.no_grad():
            # Inter-neuron hidden recurrent weights
            # (num_neurons, num_neurons)
            self.W = nn.Parameter(torch.zeros((1, num_neurons, num_neurons)))
            nn.init.normal_(self.W[:, :, :], mean=0.0, std=(1.0 / np.sqrt(num_neurons)))

            # Neuron response to input
            # (num_neurons, in_dim)
            self.I = nn.Parameter(torch.zeros((1, num_neurons, in_dim)))
            nn.init.normal_(self.I[:, :, :], mean=0.0, std=(1.0 / np.sqrt(in_dim)))

            # Neuron biases
            self.b = nn.Parameter(torch.zeros((num_neurons,)))
            nn.init.uniform_(self.b, -1.0, 1.0)

            self.fc = nn.Linear(num_neurons, out_dim)

        # (batch, num_neurons)
        self.x = None
        # (batch, num_neurons)
        self.prev_output = None
        self.x0 = None

        # Used purely for dropping gradients on the ground.
        #  We do this to reset between learning epochs where
        #  the optimizer/thing we are learning isn't this
        #  network, but we are using this network.
        self._opt = torch.optim.SGD(self.parameters(), lr=1e-3)

        self._uuid = uuid.uuid1()

    @property
    def uuid(self):
        return self._uuid.hex

    def reset(self):
        self.x = None
        self.prev_output = None
        self.x0 = None

        self._opt.zero_grad()

    def forward(self, din):
        """
        Args:
            din - (batch, in_dim)
        """
        batch_size = din.shape[0]

        if self.x is None:
            self.x0 = torch.zeros((batch_size, self.num_neurons))
            self.x = self.x0
            self.prev_output = self.activation_func(
                torch.zeros((batch_size, self.num_neurons))
            )

        x = self.W @ self.prev_output.reshape(self.x.shape + (1,))
        assert x.shape == (batch_size, self.num_neurons, 1)

        x = x + self.I @ din.reshape(din.shape + (1,))
        x = x.squeeze() + self.b
        self.x = x

        rnn_output = self.activation_func(x)
        readout = self.fc(rnn_output)

        self.prev_output = rnn_output

        return readout


class CPNNoiseyCollection(nn.Module):
    """
    This acts like a CPN, but in-fact varies its parameters for each sample in
    the batch. For N% of the batch it acts exactly like the provided CPN. For
    the rest, it acts the same, but with some amount of mean 0 noise added to
    every param.
    """

    def __init__(
        self,
        cpn,
        noisey_pct=0.90,
        noise_var=0.3,
        white_noise_pct=0.2,
        white_noise_var=2.0,
    ):
        super(CPNNoiseyCollection, self).__init__()

        self.in_dim = cpn.in_dim
        self.out_dim = cpn.out_dim
        self.activation_func_t = cpn.activation_func_t

        if cpn.activation_func_t is nn.PReLU:
            self.activation_func = cpn.activation_func_t(cpn.num_neurons)
        else:
            self.activation_func = cpn.activation_func_t()

        self.num_neurons = cpn.num_neurons
        self.cpn = cpn
        self.noisey_pct = noisey_pct
        self.noise_var = noise_var

        self.white_noise_pct = white_noise_pct
        self.white_noise_var = white_noise_var

        # (batch, num_neurons)
        self.x = None
        # (batch, num_neurons)
        self.prev_output = None
        self.x0 = None

        self.W = None
        self.I = None
        self.b = None
        self.fc_w = None
        self.fc_b = None

        self._opt = None

    def reset(self):
        self.x = None
        self.prev_output = None
        self.x0 = None

        self._opt.zero_grad()

    def step(self):
        self._opt.step()

    def setup(self, batch_size):
        with torch.no_grad():
            noisey_cnt = int(batch_size * self.noisey_pct)
            cpn = self.cpn

            # Inter-neuron hidden recurrent weights
            # (batch_size, num_neurons, num_neurons)
            self.W = nn.Parameter(
                torch.empty((batch_size, self.num_neurons, self.num_neurons))
            )
            self.W[:, :, :] = cpn.W[:, :, :].repeat(batch_size, 1, 1)
            self.W[:noisey_cnt, :, :] += self.noise_var * (
                torch.rand(noisey_cnt, self.num_neurons, self.num_neurons) - 0.5
            )

            # Neuron response to input
            # (batch_size, num_neurons, in_dim)
            self.I = nn.Parameter(
                torch.empty((batch_size, self.num_neurons, self.in_dim))
            )
            self.I[:, :, :] = cpn.I[:, :, :].repeat(batch_size, 1, 1)
            self.I[:noisey_cnt, :, :] += self.noise_var * (
                torch.rand(noisey_cnt, self.num_neurons, self.in_dim) - 0.5
            )

            # Neuron biases
            self.b = nn.Parameter(torch.empty((batch_size, self.num_neurons)))
            self.b[:, :] = cpn.b[:].reshape(1, self.num_neurons).repeat(batch_size, 1)
            self.b[:noisey_cnt, :] += self.noise_var * (
                torch.rand(noisey_cnt, self.num_neurons) - 0.5
            )

            # This one is harder... Need to re-implement fc...
            self.fc_w = nn.Parameter(
                torch.empty(batch_size, self.out_dim, self.num_neurons)
            )
            self.fc_w[:, :, :] = (
                cpn.fc.weight[:, :]
                .reshape(1, self.out_dim, self.num_neurons)
                .repeat(batch_size, 1, 1)
            )
            self.fc_w[:noisey_cnt, :, :] += self.noise_var * (
                torch.rand(noisey_cnt, self.out_dim, self.num_neurons) - 0.5
            )

            self.fc_b = nn.Parameter(torch.empty(batch_size, self.out_dim))
            self.fc_b[:, :] = (
                cpn.fc.bias[:].reshape(1, self.out_dim).repeat(batch_size, 1)
            )
            self.fc_b[:noisey_cnt, :] += self.noise_var * (
                torch.rand(noisey_cnt, self.out_dim) - 0.5
            )

        # Used purely for dropping gradients on the ground.
        #  We do this to reset between learning epochs where
        #  the optimizer/thing we are learning isn't this
        #  network, but we are using this network.
        self._opt = torch.optim.SGD(self.parameters(), lr=1e-3)

    def forward(self, din):
        """
        Args:
            din - (batch, in_dim)
        """
        batch_size = din.shape[0]

        if self.x is None:
            self.x0 = torch.zeros((batch_size, self.num_neurons))
            self.x = self.x0
            self.prev_output = self.activation_func(
                torch.zeros((batch_size, self.num_neurons))
            )

        x = self.W[:batch_size, :, :] @ self.prev_output.reshape(self.x.shape + (1,))
        assert x.shape == (batch_size, self.num_neurons, 1)

        x = x + self.I[:batch_size, :, :] @ din.reshape(din.shape + (1,))
        x = x.squeeze() + self.b[:batch_size, :]
        self.x = x

        rnn_output = self.activation_func(x)
        assert rnn_output.shape == (batch_size, self.num_neurons)

        weighted = self.fc_w[:batch_size, :, :] @ rnn_output.reshape(
            rnn_output.shape + (1,)
        )
        readout = weighted.squeeze(dim=2) + self.fc_b[:batch_size, :]

        noisey_batch_size = int(batch_size * self.white_noise_pct)
        noise = self.white_noise_var * (
            torch.rand(noisey_batch_size, self.out_dim) - 0.5
        )
        readout[:noisey_batch_size, :] = noise[:, :]

        self.prev_output = rnn_output

        return readout


class CPNModelLSTM(utils.LSTMModel):
    def __init__(self, *args, **kwargs):
        super(CPNModelLSTM, self).__init__(*args, **kwargs)

        self._uuid = uuid.uuid1()

    @property
    def uuid(self):
        return self._uuid.hex


class CPNNoiseyLSTMCollection(nn.Module):
    def __init__(
        self,
        cpn,
        activation_func=torch.nn.Tanh,
        noisey_pct=0.90,
        noise_var=0.7,
        white_noise_pct=0.3,
        white_noise_var=2,
        cuda=None,
    ):
        super(CPNNoiseyLSTMCollection, self).__init__()

        self.cpn = cpn
        self.in_dim = cpn.in_dim
        self.out_dim = cpn.out_dim
        self.num_neurons = cpn.num_neurons
        self.activation_func_t = cpn.activation_func
        self.activation_func = activation_func()
        
        if cuda is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{cuda}')
        self._cuda = cuda

        # See self.setup() for Parameter initialization

        self.noisey_pct = noisey_pct
        self.noise_var = noise_var
        self.white_noise_pct = white_noise_pct
        self.white_noise_var = white_noise_var

        self.ht = None
        self.ct = None
        self.W = None
        self.U = None
        self.bias = None
        self.fc_w = None
        self.fc_b = None

        self.grad_mask_W = None
        self.grad_mask_U = None
        self.grad_mask_bias = None
        self.grad_mask_fc_w = None
        self.grad_mask_fc_b = None

        self._opt = None
        self.batch_size = None

    def reset(self):
        self.ht = None
        self.ct = None
        self._opt.zero_grad()

    def setup(self, batch_size):
        """
        This sets the model up to process the elements of the batch
        differently. Specifically, we add mean 0 noise around the true
        model weights (from self.cpn). That helps us explore the area
        around self.cpn in parameter space.
        """
        with torch.no_grad():
            noisey_cnt = int(batch_size * self.noisey_pct)
            self.batch_size = batch_size
            cpn = self.cpn

            self.W = nn.Parameter(
                torch.Tensor(batch_size, self.in_dim, self.num_neurons * 4)
            )
            self.W[:, :, :] = (
                cpn.W[:, :].reshape((1,) + cpn.W.shape).repeat(batch_size, 1, 1)
            )
            stdev = torch.std(self.W[:noisey_cnt, :, :]) * self.noise_var
            # self.W[:noisey_cnt, :, :] += self.noise_var * (
            # torch.rand(noisey_cnt, self.W.shape[1], self.W.shape[2]) - 0.5
            # )
            self.W[:noisey_cnt, :, :] += torch.normal(
                0.0, stdev, (noisey_cnt, self.W.shape[1], self.W.shape[2])
            )
            self.grad_mask_W = torch.zeros(self.W.shape).to(self.device)
            self.grad_mask_W[:noisey_cnt, :, :] = 1.0

            self.U = nn.Parameter(
                torch.Tensor(batch_size, self.num_neurons, self.num_neurons * 4)
            )
            self.U[:, :, :] = (
                cpn.U[:, :].reshape((1,) + cpn.U.shape).repeat(batch_size, 1, 1)
            )
            stdev = torch.std(self.U[:noisey_cnt, :, :]) * self.noise_var
            # self.U[:noisey_cnt, :, :] += self.noise_var * (
            # torch.rand(noisey_cnt, self.U.shape[1], self.U.shape[2]) - 0.5
            # )
            self.U[:noisey_cnt, :, :] += torch.normal(
                0.0, stdev, (noisey_cnt, self.U.shape[1], self.U.shape[2])
            )
            self.grad_mask_U = torch.zeros(self.U.shape).to(self.device)
            self.grad_mask_U[:noisey_cnt, :, :] = 1.0

            self.bias = nn.Parameter(torch.Tensor(batch_size, self.num_neurons * 4))
            self.bias[:, :] = (
                cpn.bias[:].reshape(1, cpn.bias.shape[0]).repeat(batch_size, 1)
            )
            stdev = torch.std(self.bias[:noisey_cnt, :]) * self.noise_var
            # self.bias[:noisey_cnt, :] += self.noise_var * (
            # torch.rand(noisey_cnt, self.bias.shape[1]) - 0.5
            # )
            self.bias[:noisey_cnt, :] += torch.normal(
                0.0, stdev, (noisey_cnt, self.bias.shape[1])
            )
            self.grad_mask_bias = torch.zeros(self.bias.shape).to(self.device)
            self.grad_mask_bias[:noisey_cnt, :] = 1.0

            # This one is harder... Need to re-implement fc...
            self.fc_w = nn.Parameter(
                torch.empty(batch_size, self.out_dim, self.num_neurons)
            )
            self.fc_w[:, :, :] = (
                cpn.fc.weight[:, :]
                .reshape(1, self.out_dim, self.num_neurons)
                .repeat(batch_size, 1, 1)
            )
            stdev = torch.std(self.fc_w[:noisey_cnt, :, :]) * self.noise_var
            # self.fc_w[:noisey_cnt, :, :] += self.noise_var * (
            # torch.rand(noisey_cnt, self.out_dim, self.num_neurons) - 0.5
            # )
            self.fc_w[:noisey_cnt, :, :] += torch.normal(
                0.0, stdev, (noisey_cnt, self.out_dim, self.num_neurons)
            )
            self.grad_mask_fc_w = torch.zeros(self.fc_w.shape).to(self.device)
            self.grad_mask_fc_w[:noisey_cnt, :, :] = 1.0

            self.fc_b = nn.Parameter(torch.empty(batch_size, self.out_dim))
            self.fc_b[:, :] = (
                cpn.fc.bias[:].reshape(1, self.out_dim).repeat(batch_size, 1)
            )
            stdev = torch.std(self.fc_b[:noisey_cnt, :]) * self.noise_var
            # self.fc_b[:noisey_cnt, :] += self.noise_var * (
            #    torch.rand(noisey_cnt, self.out_dim) - 0.5
            # )
            self.fc_b[:noisey_cnt, :] += torch.normal(
                0.0, stdev, (noisey_cnt, self.out_dim)
            )
            self.grad_mask_fc_b = torch.zeros(self.fc_b.shape).to(self.device)
            self.grad_mask_fc_b[:noisey_cnt, :] = 1.0

        # Used for dropping gradients on the ground.
        #  We do this to reset between learning epochs where
        #  the optimizer/thing we are learning isn't this
        #  network, but we are using this network.
        # Also: we use this for our "wiggle the gradients during EN
        #  training".
        self._opt = torch.optim.SGD(self.parameters(), lr=1e-5)

    # def step(self, only_noisey=True):
    #    if only_noisey:
    #        noisey_cnt = int(self.W.shape[0] * self.noisey_pct)

    #        # W
    #        grad_mean = torch.mean(self.W.grad[:noisey_cnt, :, :])
    #        grad_stdev = torch.std(self.W.grad[:noisey_cnt, :, :])
    #        self.W.grad[:noisey_cnt, :, :] += torch.normal(grad_mean,
    #                grad_stdev, (noisey_cnt, self.W.shape[1], self.W.shape[2])
    #                ).cuda(self._cuda)
    #        self.W.grad *= self.grad_mask_W

    #        # U
    #        grad_mean = torch.mean(self.U.grad[:noisey_cnt, :, :])
    #        grad_stdev = torch.std(self.U.grad[:noisey_cnt, :, :])
    #        self.U.grad[:noisey_cnt, :, :] += torch.normal(grad_mean,
    #                grad_stdev, (noisey_cnt, self.U.shape[1], self.U.shape[2])
    #                ).cuda(self._cuda)
    #        self.U.grad *= self.grad_mask_U

    #        # bias
    #        grad_mean = torch.mean(self.bias.grad[:noisey_cnt, :])
    #        grad_stdev = torch.std(self.bias.grad[:noisey_cnt, :])
    #        self.bias.grad[:noisey_cnt, :] += torch.normal(grad_mean,
    #                grad_stdev, (noisey_cnt, self.bias.shape[1])
    #                ).cuda(self._cuda)
    #        self.bias.grad *= self.grad_mask_bias

    #        # fc_w
    #        grad_mean = torch.mean(self.fc_w.grad[:noisey_cnt, :, :])
    #        grad_stdev = torch.std(self.fc_w.grad[:noisey_cnt, :, :])
    #        self.fc_w.grad[:noisey_cnt, :, :] += torch.normal(grad_mean,
    #                grad_stdev, (noisey_cnt, self.out_dim, self.num_neurons)
    #                ).cuda(self._cuda)
    #        self.fc_w.grad *= self.grad_mask_fc_w

    #        # fc_b
    #        grad_mean = torch.mean(self.fc_b.grad[:noisey_cnt, :])
    #        grad_stdev = torch.std(self.fc_b.grad[:noisey_cnt, :])
    #        self.fc_b.grad[:noisey_cnt, :] += torch.normal(grad_mean,
    #                grad_stdev, (noisey_cnt, self.out_dim)
    #                ).cuda(self._cuda)
    #        self.fc_b.grad *= self.grad_mask_fc_b

    #    self._opt.step()

    def forward(self, x_t):
        """Assumes x_t is of shape (batch, feature)"""

        batch_size, in_dim = x_t.shape
        assert in_dim == self.in_dim

        if self.ht is None:
            self.ht = torch.zeros(batch_size, self.num_neurons)
            self.ct = torch.zeros(batch_size, self.num_neurons)
            self.setup(batch_size)

            if self._cuda is not None:
                self.ht = self.ht.cuda(self._cuda)
                self.ct = self.ct.cuda(self._cuda)
                self.cuda(self._cuda)

        # batch the computations into a single matrix multiplication
        gates = (x_t.unsqueeze(dim=1) @ self.W).squeeze()
        assert gates.shape == (batch_size, 4 * self.num_neurons)

        gates = gates + (self.ht.unsqueeze(dim=1) @ self.U).squeeze()
        assert gates.shape == (batch_size, 4 * self.num_neurons)

        gates = gates + self.bias
        assert gates.shape == (batch_size, 4 * self.num_neurons)

        HS = self.num_neurons
        i_t, f_t, g_t, o_t = (
            torch.sigmoid(gates[:, :HS]),  # input
            torch.sigmoid(gates[:, HS : HS * 2]),  # forget
            torch.tanh(gates[:, HS * 2 : HS * 3]),
            torch.sigmoid(gates[:, HS * 3 :]),  # output
        )

        self.ct = f_t * self.ct + i_t * g_t
        self.ht = o_t * torch.tanh(self.ct)

        activation = self.activation_func(self.ht)

        weighted = self.fc_w @ activation.reshape(activation.shape + (1,))
        out = weighted.squeeze(dim=2) + self.fc_b

        noisey_batch_size = int(batch_size * self.white_noise_pct)
        noise = self.white_noise_var * (
            torch.rand(noisey_batch_size, self.out_dim) - 0.5
        )
        out[:noisey_batch_size, :] = noise[:, :]

        return out
