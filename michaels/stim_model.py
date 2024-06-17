import numpy as np
import torch
import torch.autograd
from torch import nn
from torch.optim import AdamW

from experiment import utils


def get_stim_model(
    in_dim, out_dim, num_neurons=None, activation=torch.nn.Tanh, cuda=None
):

    if num_neurons is None:
        num_neurons = in_dim + 50

    en = StimModelLSTM(
        in_dim,
        out_dim,
        num_neurons=num_neurons,
        activation_func=activation,
        cuda=cuda,
    )

    opt_en = AdamW(en.parameters(), lr=1e-3, weight_decay=0.04)

    return en, opt_en


class StimModel(nn.Module):
    def __init__(
        self, in_dim, out_dim, activation_func=utils.ReTanh, num_neurons=None, cuda=None
    ):
        super(StimModel, self).__init__()

        if num_neurons is None:
            num_neurons = in_dim + 50

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_func_t = activation_func

        self._cuda = cuda

        if activation_func is nn.PReLU:
            self.activation_func = activation_func(num_neurons)
        else:
            self.activation_func = activation_func()

        self.num_neurons = num_neurons

        with torch.no_grad():
            # Inter-neuron hidden recurrent weights
            # (num_neurons, num_neurons)
            self.W = nn.Parameter(torch.zeros((num_neurons, num_neurons)))
            nn.init.normal_(self.W[:, :], mean=0.0, std=(1.0 / np.sqrt(num_neurons)))

            # Neuron response to input
            # (num_neurons, in_dim)
            self.I = nn.Parameter(torch.zeros((num_neurons, in_dim)))
            nn.init.normal_(self.I[:, :], mean=0.0, std=(1.0 / np.sqrt(in_dim)))

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

    def reset(self):
        self.x = None
        self.prev_output = None

        self._opt.zero_grad()

    def load_weights_from_file(self, data_path):
        self.load_state_dict(torch.load(data_path))
        self.eval()

    def forward(self, din):
        """
        Args:
            din - (batch, in_dim)
        """
        batch_size = din.shape[0]

        if self.x is None:
            self.x0 = torch.zeros((batch_size, self.num_neurons))
            self.x = self.x0
            self.prev_output = torch.zeros((batch_size, self.num_neurons))

            if self._cuda is not None:
                self.x0 = self.x0.cuda(self._cuda)
                self.x = self.x.cuda(self._cuda)
                self.prev_output = self.prev_output.cuda(self._cuda)

        x = self.W.reshape((1,) + self.W.shape) @ self.prev_output.reshape(
            self.prev_output.shape + (1,)
        )
        assert x.shape == (batch_size, self.num_neurons, 1)

        x += self.I.reshape((1,) + self.I.shape) @ din.reshape(din.shape + (1,))
        x = x.squeeze() + self.b
        self.x = x

        rnn_output = self.activation_func(x)
        readout = self.fc(rnn_output)

        self.prev_output = rnn_output
        return readout


class StimModelLSTM(utils.LSTMModel):
    pass


def load_from_file(model_path, in_dim, out_dim, pretrained=False, **kwargs):
    ben = StimModel(in_dim, out_dim, **kwargs)
    ben.load_weights_from_file(model_path)

    if pretrained:
        for param in ben.parameters():
            param.requires_grad = False

    return ben
