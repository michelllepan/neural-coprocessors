import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from . import michaels_load
from . import utils
random.seed(0)
torch.manual_seed(0)
# Length of real-world time for a given time step, in ms
tick = 10
dt = 1
tau = 100 / tick
synaptic_scaling_factor = 1.2
num_neurons_per_module = 100
num_modules = 3
num_input_features = 20
num_output_features = 50
sparsity = 0.1


class MichaelsRNN(nn.Module):
    def __init__(
        self,
        tau=tau,
        dt=dt,
        num_neurons_per_module=num_neurons_per_module,
        num_modules=num_modules,
        num_input_features=num_input_features,
        sparsity=sparsity,
        synaptic_scaling_factor=synaptic_scaling_factor,
        activation_func=utils.ReTanh,
        output_dim=num_output_features,
        stimulus=None,
        lesion=None,
        init_data_path=None,
        cuda=None,
    ):

        super(MichaelsRNN, self).__init__()

        self.tau = tau
        self.dt = dt
        self.sparsity = sparsity
        self.num_neurons_per_module = num_neurons_per_module
        self.num_modules = num_modules
        self.num_input_features = num_input_features
        self.num_neurons = num_neurons_per_module * num_modules
        self.synaptic_scaling_factor = synaptic_scaling_factor
        self.activation_func = activation_func()
        self.output_dim = output_dim
        self._cuda = cuda
        self.last_stimulus = None

        npm = self.num_neurons_per_module
        numn = self.num_neurons



        # Recurrent state -----------------
        # Last outputs from the RNN. Setting to 0 for the first time step.
        # Set on first use, since we will keep memory for each trial in the
        # batch, and we don't know the batch size until the first call of
        # forward()
        # Referred to as r_i in the Michaels text.
        # (batch_size, num_neurons) once set
        self.prev_output = None

        # Internal state of the neurons. Referred to as x_i in the text
        self.x = None

        if init_data_path is not None:
            (J, I, S, B, fc, x0,) = utils.init_from_michaels_model(
                init_data_path, num_input_features, num_neurons_per_module, output_dim
            )

            self.J = nn.Parameter(J)
            self.I = nn.Parameter(I)
            self.S = nn.Parameter(S)
            self.B = nn.Parameter(B)
            self.fc = fc
            self.x0 = nn.Parameter(x0)

            self.J_zero_grad_mask = None
            self.I_zero_grad_mask = None
        else:
            # Masks ---------------------------
            # (num_neurons, num_neurons)
            # Set to 1 for pairs which represent sparse connections
            # There are two areas of the matrix with sparse connections:
            #   module 1->2 and module 2->3
            # A '1' is a connection, a '0' is a non-connection
            sparse_mask = torch.zeros((numn, numn))

            # list of (neuron_idx, neuron_idx)
            # Memory inefficient approach, but easy to understand,
            #  and we have a small model...
            possible_connections = []
            for i in range(npm):
                for j in range(npm):
                    possible_connections.append((i, j))
            random.seed(0)
            for c12 in random.sample(possible_connections, int(sparsity * npm ** 2)):
                sparse_mask[c12[0] + npm, c12[1]] = 1
            random.seed(1)
            for c23 in random.sample(possible_connections, int(sparsity * npm ** 2)):
                sparse_mask[c23[0] + (npm * 2), c23[1] + npm] = 1
            random.seed(2)
            for c21 in random.sample(possible_connections, int(sparsity * npm ** 2)):
                sparse_mask[c21[0], c21[1] + npm] = 1
            random.seed(3)
            for c32 in random.sample(possible_connections, int(sparsity * npm ** 2)):
                sparse_mask[c32[0] + npm, c32[1] + (npm * 2)] = 1

            # Zero grad mask: used to 0 out gradients during training,
            # to account for sparsity
            # (num_neurons, num_neurons)
            self.J_zero_grad_mask = torch.zeros((numn, numn))

            # Don't zero the sparse connections
            self.J_zero_grad_mask += sparse_mask

            # Don't zero the FC connections
            fc_mask = torch.ones((npm, npm))
            for i in range(num_modules):
                start = i * npm
                self.J_zero_grad_mask[
                    start : start + npm, start : start + npm
                ] = fc_mask

            self.I_zero_grad_mask = torch.zeros((self.num_neurons, self.num_neurons))
            self.I_zero_grad_mask[-1 * npm :, -1 * npm :] = 1.0

            with torch.no_grad():
                # Params --------------------------
                # Bias: (num_neurons, 1)
                self.B = nn.Parameter(torch.empty((numn, 1)))
                # See: https://github.com/JonathanAMichaels/hfopt-matlab/blob/0a18401c62b555bf799de83aa0d722bc82cf06d2/rnn/init_rnn.m#L146
                nn.init.uniform_(self.B, -1.0, 1.0)

                # Input weights: (num_neurons, num_input_features)
                self.I = nn.Parameter(torch.empty((numn, num_input_features)))
                nn.init.normal_(self.I, mean=0.0, std=(1 / np.sqrt(num_input_features)))

                # zero out those not in the input module
                self.I[-1 * npm :, :] = 0.0

                # Hidden state response to the hold signal
                # (num_neurons, 1)
                self.S = nn.Parameter(torch.empty((numn, 1)))
                nn.init.normal_(self.S, mean=0.0, std=1)

                # Recurrent weights: (num_neurons (current), num_neurons (prev))
                self.J = nn.Parameter(torch.zeros((numn, numn)))

                # Intra-module portions
                for i in range(num_modules):
                    start = i * npm
                    nn.init.normal_(
                        self.J[start : start + npm, start : start + npm],
                        mean=0.0,
                        std=(synaptic_scaling_factor / np.sqrt(npm)),
                    )

                # Inter-module portions
                #  modules 1->2, 2->3, 3->2, 2->1
                for i in (0, 1):
                    start_in = i * npm
                    start_out = start_in + npm
                    nn.init.normal_(
                        self.J[start_out : start_out + npm, start_in : start_in + npm],
                        mean=0.0,
                        std=(1 / np.sqrt(npm)),
                    )

                for i in (1, 2):
                    start_in = i * npm
                    start_out = (i - 1) * npm
                    nn.init.normal_(
                        self.J[start_out : start_out + npm, start_in : start_in + npm],
                        mean=0.0,
                        std=(1 / np.sqrt(npm)),
                    )

                self.J *= self.J_zero_grad_mask

                self.x0 = nn.Parameter(torch.empty(self.num_neurons))
                self.fc = nn.Linear(num_neurons_per_module, output_dim)

        self.recalc_masks_from_weights()
        self.lesion = lesion
        self.stimulus = stimulus

        self.reset_hidden()

        if cuda is not None:
            self.to(cuda)

    def recalc_masks_from_weights(self):
        npm = self.num_neurons_per_module
        J_zero_grad_mask = torch.zeros(self.num_neurons, self.num_neurons)
        J_zero_grad_mask[self.J != 0.0] = 1.0
        J_zero_grad_mask[:npm, :npm] = 1.0
        J_zero_grad_mask[npm : npm * 2, npm : npm * 2] = 1.0
        J_zero_grad_mask[npm * 2 : npm * 3, npm * 2 : npm * 3] = 1.0
        self.J_zero_grad_mask = J_zero_grad_mask

        I_zero_grad_mask = torch.zeros(self.num_neurons, self.num_input_features)
        I_zero_grad_mask[npm:, :] = 1.0
        self.I_zero_grad_mask = I_zero_grad_mask

        # The hold signal response can change, but not input response
        self.I_connection_coadap_zero_grad_mask = torch.zeros(
            self.num_neurons, self.num_input_features
        )

        # Only M1 internal mappings
        self.J_connection_coadap_zero_grad_mask = torch.zeros(
            self.num_neurons, self.num_neurons
        )
        self.J_connection_coadap_zero_grad_mask[:npm, :npm] = 1.0

        # Only M1
        self.S_connection_coadap_zero_grad_mask = torch.zeros(self.num_neurons, 1)
        self.S_connection_coadap_zero_grad_mask[:npm] = 1.0

        # Only M1
        self.B_connection_coadap_zero_grad_mask = torch.zeros(self.num_neurons, 1)
        self.B_connection_coadap_zero_grad_mask[:npm] = 1.0

        if self._cuda is not None:
            self.J_zero_grad_mask = self.J_zero_grad_mask.to(self._cuda)
            self.I_zero_grad_mask = self.I_zero_grad_mask.to(self._cuda)
            self.I_connection_coadap_zero_grad_mask = (
                self.I_connection_coadap_zero_grad_mask.to(self._cuda)
            )
            self.J_connection_coadap_zero_grad_mask = (
                self.J_connection_coadap_zero_grad_mask.to(self._cuda)
            )
            self.S_connection_coadap_zero_grad_mask = (
                self.S_connection_coadap_zero_grad_mask.to(self._cuda)
            )
            self.B_connection_coadap_zero_grad_mask = (
                self.B_connection_coadap_zero_grad_mask.to(self._cuda)
            )

    def load_weights_from_file(self, data_path):
        self.load_state_dict(torch.load(data_path, map_location=torch.device('cpu')))
        self.eval()
        self.recalc_masks_from_weights()

    @property
    def tau_inv(self):
        return 1.0 / self.tau

    def reset_stim(self, batch_size=None):
        if self.stimulus is not None:
            self.stimulus.reset(batch_size=batch_size)

    def reset_hidden(self):
        self.prev_output = None

        # Internal state of the neurons. Referred to as x_i in the text
        self.x = None

    def reset(self):
        self.reset_hidden()

    def set_sparse_grads(self):
        self.J.grad *= self.J_zero_grad_mask
        self.I.grad *= self.I_zero_grad_mask

    def set_connection_coadap_grads(self):
        self.J.grad *= self.J_connection_coadap_zero_grad_mask
        # It seems there is an AdamW bug where these are not always effective...
        # But let's try.
        self.I.grad *= self.I_connection_coadap_zero_grad_mask

        self.S.grad *= self.S_connection_coadap_zero_grad_mask
        self.B.grad *= self.B_connection_coadap_zero_grad_mask

    def set_end_to_end_coadap_grads(self):
        self.set_sparse_grads()

    def set_lesion(self, lesion):
        """
        Args:
            mask: tensor (num_neurons,), with zeros for neurons which
                should not fire, and ones elsewhere.
                Or: pass None to reset
        """
        self.lesion = lesion

    def forward(self, data,prev_output):
        """
        Args:
          data:
            - hold: vector of: double 0.0 or 1.0: (batch_size)
            - image: Tensor((num_input_features, batch_size))
        """
        data=data.squeeze(0).transpose(0,1)
        self.prev_output=prev_output

        #image, hold = np.split(data, [self.num_input_features], axis=0)
        image=data
        if data.get_device()==-1:
            device='cpu'
        else:
            device='cuda'
        hold=torch.zeros((1,data.shape[1])).to(device)
        batch_size = image.shape[1]

        if self.prev_output is None:
            # (batch_size, num_neurons)
            self.x = torch.tile(self.x0, (batch_size, 1))

            if self.lesion is not None and self.lesion.application == "output":
                self.x = self.lesion.lesion(self, self.x)

            self.prev_output = self.activation_func(self.x)
            self.reset_stim(batch_size=batch_size)
        elif batch_size != self.prev_output.shape[0]:
            raise RuntimeError(
                "Must have the same batch size every time step. "
                "Did you forget to reset the module between batches?"
            )

        # Double check myself; can remove this and other asserts after testing
        assert len(image.shape) == 2
        assert image.shape[0] == self.num_input_features
        assert self.prev_output.shape[1] == self.num_neurons

        # Cleared for take-off...
        # Recurrence

        if self.lesion is not None and self.lesion.application == "connection":
            recur = self.prev_output @ self.lesion.lesion(self, self.J).T
        else:
            recur = self.prev_output @ self.J.T
        assert recur.shape == (batch_size, self.num_neurons), str(recur.shape)

        # Input
        inp = image.T @ self.I.T + hold.T * self.S.T
        assert inp.shape == (batch_size, self.num_neurons)

        # x broadcast up from (numn,) to (batch_size, numn)
        tdx = -self.x + recur + inp + self.B.T
        if self.stimulus is not None:
            tdx += self.get_next_stimulus()

        pre_response = self.x + tdx / 10
        assert pre_response.shape == (batch_size, self.num_neurons)

        if self.lesion is not None and self.lesion.application == "output":
            pre_response = self.lesion.lesion(self, pre_response)

        output = self.activation_func(pre_response)
        assert output.shape == (batch_size, self.num_neurons)

        self.x = pre_response
        #self.prev_output = output

        # Return only from the final module
        ret = self.fc(output[:, : self.num_neurons_per_module])
        return ret.unsqueeze(0),output

    def observe(self, obs_model, drop_module_idx=None):
        outputs = []

        for midx in range(self.num_modules):
            if midx == drop_module_idx:
                continue

            act = self.prev_output[
                :,
                midx
                * self.num_neurons_per_module : (midx + 1)
                * self.num_neurons_per_module,
            ]
            out = obs_model(act)
            # aka (batch_size, out_dim)
            assert out.shape == (self.prev_output.shape[0], obs_model.out_dim)
            outputs.append(out.detach())

        # 3-tuple, elements are (batch, obs.out_dim)
        # Possibly fewer than 3, if we are dropping modules
        return tuple(outputs)

    def unroll(self, data_in, cuda=None):
        """
        data_in - (in_dim, time)
        """
        data = data_in.unsqueeze(axis=2)
        steps = data_in.shape[1]
        pred_out = torch.empty((1, steps, self.output_dim))

        if cuda is not None:
            pred_out = pred_out.to(cuda)

        for tidx in range(steps):
            cur = data[:, tidx, :]
            pred_out[0, tidx, :] = self(cur)
        return pred_out

    def stimulate(self, params):
        self.stimulus.add(params)

    def get_next_stimulus(self):
        # (batch_size, num_neurons)
        self.last_stimulus = self.stimulus.get_next()
        return self.last_stimulus


class MichaelsDataset(Dataset):
    def __init__(
        self, data_file_path, with_label=False, limit=None, class_filter=None, cuda=None
    ):
        """
        class_filter: optional iterable of class labels to allowlist
        """

        f = michaels_load.load_from_path(data_file_path)
        inps = f["inp"]
        outs = f["targ"]

        if with_label:
            ti = f["trialInfo"]
        else:
            ti = None

        self.num_samples = inps.shape[0]
        if limit is not None:
            self.num_samples = min(limit, self.num_samples)

        self.sample_len = max([s.shape[1] for s in inps])

        self.with_label = with_label

        self.class_filter = class_filter

        self.data = []
        self._load_data(
            inps, outs, trial_info=ti, class_filter=self.class_filter, cuda=cuda
        )

    def __len__(self):
        return len(self.data)

    def _load_data_single(self, idx, inps, outs, trial_info=None, cuda=None):
        cur_in = inps[idx]
        cur_out = outs[idx]

        din = torch.zeros((self.sample_len, cur_in.shape[0]), dtype=torch.float)
        din[: cur_in.shape[1], :] = torch.tensor(cur_in.T[:, :])
        din[cur_in.shape[1] :, :] = 0.0

        trial_end = torch.zeros((self.sample_len, 1), dtype=torch.float)
        trial_end[: cur_in.shape[1], 0] = 1.0

        trial_len = torch.tensor(cur_in.shape[1])

        dout = torch.zeros((self.sample_len, cur_out.shape[0]), dtype=torch.float)
        dout[: cur_out.shape[1], :] = torch.tensor(cur_out.T[:, :])

        if cuda is not None:
            din = din.to(cuda)
            trial_end = trial_end.to(cuda)
            trial_len = trial_len.to(cuda)
            dout = dout.to(cuda)

        if trial_info is None:
            return din, trial_end, trial_len, dout

        norm = trial_info[idx][0]
        if norm > 80:
            norm -= 10
        norm -= 21
        fact = norm // 10
        norm = fact * 6 + norm % 10
        label = torch.tensor(norm)

        if cuda is not None:
            label = label.to(cuda)

        return din, trial_end, trial_len, dout, label

    def _load_data(self, inps, outs, trial_info=None, class_filter=None, cuda=None):
        if trial_info is None and class_filter is not None:
            raise ValueError("Must provide a trial_info if providing class_filter")

        for i in range(self.num_samples):
            datum = self._load_data_single(
                i, inps, outs, trial_info=trial_info, cuda=cuda
            )

            if class_filter is None or (datum[-1].item() in class_filter):
                self.data.append(datum)

    def __getitem__(self, idx):
        return self.data[idx]


def load_from_file(data_path, pretrained=False, **kwargs):
    mrnn = MichaelsRNN(**kwargs)
    mrnn.load_weights_from_file(data_path)

    if pretrained:
        for param in mrnn.parameters():
            param.requires_grad = False

    return mrnn
