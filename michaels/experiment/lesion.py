import random

import enum

import torch
torch.manual_seed(0)


class LesionModule(enum.Enum):
    M1 = 0
    F5 = 1
    AIP = 2


def module_id_to_idxs(num_neurons_per_module, module_id):
    if isinstance(module_id, str):
        mid = LesionModule[module_id].value
    elif not isinstance(module_id, int):
        raise TypeError("module_id must be an integer index")

    return num_neurons_per_module * mid, num_neurons_per_module * (mid + 1)


class Lesion(object):
    application = "output"

    def lesion(self, network, pre_response):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class LesionOutputs(Lesion):
    def __init__(self, num_neurons_per_module, module_id, pct, cuda=None):
        self.lesion_mask = torch.ones((num_neurons_per_module * 3,))
        start_idx, end_idx = module_id_to_idxs(num_neurons_per_module, module_id)
        kill_idxs = random.sample(
            list(range(start_idx, end_idx)), int(pct * num_neurons_per_module)
        )
        self.lesion_mask[kill_idxs] = 0.0
        self.module_id = module_id
        self.pct = pct

        if cuda is not None:
            self.lesion_mask = self.lesion_mask.to(cuda)

    def lesion(self, network, pre_response):
        batch_size = pre_response.shape[0]
        out = pre_response * torch.tile(self.lesion_mask, (batch_size, 1))
        return out

    def __str__(self):
        return f"outputs{self.module_id}.{self.pct}"


class LesionOutputsByIdxs(Lesion):
    def __init__(self, num_neurons_per_module, start_idx, end_idx, cuda=None):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.lesion_mask = torch.ones((num_neurons_per_module * 3,))
        self.lesion_mask[start_idx:end_idx] = 0.0

        if cuda is not None:
            self.lesion_mask = self.lesion_mask.to(cuda)

    def lesion(self, network, pre_response):
        batch_size = pre_response.shape[0]
        out = pre_response * torch.tile(self.lesion_mask, (batch_size, 1))
        return out

    def __str__(self):
        return f"outputsIdxs{self.start_idx}.{self.end_idx}"


class LesionOutputsByMask(Lesion):
    def __init__(self, num_neurons_per_module, lesion_mask, cuda=None):
        assert lesion_mask.shape == (num_neurons_per_module * 3,)
        self.lesion_mask = lesion_mask

    def lesion(self, network, pre_response):
        batch_size = pre_response.shape[0]
        out = pre_response * torch.tile(self.lesion_mask, (batch_size, 1))
        return out

    def __str__(self):
        return f"outputsIdxs{self.start_idx}.{self.end_idx}"


class LesionConnectionsByIdxs(Lesion):
    application = "connection"

    def __init__(self, num_neurons_per_module, idxs, idxs_are_modules=True, cuda=None):
        # idxs: an iterable of (start_idx_in, end_idx_in, start_idx_out, end_idx_out),
        #  where those indices indicate modules if idxs_are_modules,
        # otherwise indicating neuron ranges
        self.idxs = idxs
        self.idxs_are_modules = idxs_are_modules

        self._cuda = cuda

        num_neurons = num_neurons_per_module * 3
        self.lesion_mask = torch.ones((num_neurons, num_neurons))

        for i in idxs:
            start_in, end_in, start_out, end_out = i

            if idxs_are_modules:
                si = start_in * num_neurons_per_module
                ei = end_in * num_neurons_per_module
                so = start_out * num_neurons_per_module
                eo = end_out * num_neurons_per_module
                self.lesion_mask[so:eo, si:ei] = 0.0
            else:
                self.lesion_mask[start_out:end_out, start_in:end_in] = 0.0

        if cuda is not None:
            self.lesion_mask = self.lesion_mask.to(cuda)

    def lesion(self, network, pre_response):
        out = self.lesion_mask * pre_response
        return out

    def __str__(self):
        return f"connectionsIdxs{self.idxs}"


class LesionType(enum.Enum):
    outputs = LesionOutputsByIdxs
    connection = LesionConnectionsByIdxs
    none = None
