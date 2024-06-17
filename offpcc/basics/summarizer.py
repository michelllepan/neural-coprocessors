import gin
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import torch.nn as nn
import torch
import michaels.experiment.experiment as experiment
import michaels.experiment.mRNN as mRNN


@gin.configurable(module=__name__)
class Summarizer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=2, recurrent_type='lstm',cfg=None):

        super().__init__()
        self.michaels=False
        self.hidden_dim=hidden_dim
        if recurrent_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        elif recurrent_type == 'rnn':
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        elif recurrent_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        elif recurrent_type=='michaels':
            self.michaels=True
            self.input_dim=input_dim
            self.hidden_dim=hidden_dim
            if cfg is None:
                cfg = experiment.get_config(coadapt=False, cuda=None, obs_out_dim=10, num_stim_channels=10,
                                            num_neurons_per_module=100)
            self.rnn = mRNN.MichaelsRNN(
                stimulus=cfg.stim_instance,
                cuda=cfg.cuda,
                num_input_features=input_dim,
                output_dim=hidden_dim
            )

        else:
            raise ValueError(f"{recurrent_type} not recognized")


    def forward(self, observations, hidden=None, return_hidden=False):
        #self.rnn.flatten_parameters()
        if self.michaels and observations.shape[1]>1:
            summary,hidden=self.get_michaels_summary_hidden(observations,hidden)
        else:
            summary, hidden = self.rnn(observations, hidden)
        if return_hidden:
            return summary, hidden
        else:
            return summary

    def get_michaels_summary_hidden(self,observations,hidden):
        summary=torch.zeros((observations.shape[0],observations.shape[1],self.hidden_dim))
        for j in range(observations.shape[1]):
            sample=observations[:,j,:]
            s,hidden=self.rnn(sample,hidden)
            summary[:,j,:]=s

        return summary,hidden

