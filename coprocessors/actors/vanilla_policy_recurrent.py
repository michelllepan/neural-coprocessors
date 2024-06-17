import torch

from coprocessors.actors import BaseStimTester


class VanillaPolicyRecurrent(BaseStimTester):

    def __init__(self, env, policy, device="cpu"):
        super().__init__(env, device=device)
        self.device = device
        self.policy = policy

    def act(self, obs, deterministic):
        act = self.policy.act(
            torch.FloatTensor(obs).to(self.device),
            deterministic=deterministic,
        )
        return act

    def reset(self):
        self.policy.actor_summarizer.rnn.reset()
        self.policy.reinitialize_hidden()
