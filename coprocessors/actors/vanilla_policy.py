from coprocessors.actors.base_actor import BaseStimTester


class VanillaPolicy(BaseStimTester):
    
    def __init__(self, env, policy, device="cpu"):
        super().__init__(env, [], device=device)
        self.device = device
        self.policy = policy

    def act(self, obs, deterministic):
        act = self.policy(obs, deterministic=deterministic)[0]
        return act
