import torch
import torch.nn as nn


class ActionNet(nn.Module):

    def __init__(
            self,
            obs_size=8,
            stim_size=20,
            hid_size1=64,
            hid_size2=32,
            hid_size3=8,
            act_size=2, temporal=False
    ):
        super(ActionNet, self).__init__()
        self.temporal = temporal
        additional_input = 0

        if self.temporal:
            self.lstm = nn.LSTM(stim_size, 64)
            additional_input = 64

        self.layers = nn.Sequential(
            nn.Linear(obs_size + stim_size+additional_input, hid_size1),
            nn.ReLU(),
            nn.Linear(hid_size1, hid_size2),
            nn.ReLU(),
            nn.Linear(hid_size2, hid_size3),
            nn.ReLU(),
            nn.Linear(hid_size3, act_size),
            nn.Tanh(),
        )

    def forward(self, obs, action, hist):
        if self.temporal:
            hist_out, _ = self.lstm(hist)
            hist_out = hist_out[:, -1, :]
            x = torch.cat((obs, action, hist_out), dim=1)
        else:
            x = torch.cat((obs, action), dim=1)

        return self.layers(x)
