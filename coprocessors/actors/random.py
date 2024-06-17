import numpy as np

from coprocessors.actors import BaseStimTester


class RandomActor(BaseStimTester):
    
    def __init__(self, env, device="cpu"):
        super().__init__(env, device=device)
        self.device = device

    def act(self, obs, deterministic):
        act = np.random.uniform(
            self.action_space.low,
            self.action_space.high,
            size=self.action_space.shape,
        )
        return act
