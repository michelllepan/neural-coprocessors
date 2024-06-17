import pickle

import torch
from torch.utils.data import Dataset


class ActionDataset(Dataset):

    def __init__(self, data_path=None, data=[], include_trans=False):
        if data_path is None:
            self.data = data
        else:
            with open(data_path, 'rb') as fp:
                self.data = pickle.load(fp)

        self.obs = self.data['obs']
        self.cp_actions = self.data['cp_action']
        self.w_actions = self.data['w_action']
        self.history = self.data['history']
        self.next_obs = self.data['next_obs']
        self.next_cp_action = self.data['next_cp_action']
        self.reward = self.data['reward']
        self.include_trans = include_trans

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        ob = torch.tensor(self.obs[idx]).float()
        c_a = torch.tensor(self.cp_actions[idx]).float()
        w_a = torch.tensor(self.w_actions[idx]).float()
        h = torch.tensor(self.history[idx]).float()
        nobs = torch.tensor(self.next_obs[idx]).float()
        ncpa = torch.tensor(self.next_cp_action[idx]).float()
        reward = torch.tensor(self.reward[idx]).float()
        if self.include_trans:
            return ob, nobs, reward, c_a,ncpa, w_a, h
        else:
            return ob, c_a, w_a, h
