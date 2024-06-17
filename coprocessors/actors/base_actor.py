class BaseStimTester:
    
    def __init__(
        self,
        env=[],
        healthy_q=[],
        stims=[],
        statistics=[],
        min_max=False,
        device="cpu",
    ):
        self.action_space = env.action_space
        self.env = env
        self.healthy_q = healthy_q
        self.stimulations = stims
        self.min_max = min_max
        self.statistics = statistics
        self.device = device

    def act(self, obs, deterministic):
        pass

    def reset(self):
        pass
