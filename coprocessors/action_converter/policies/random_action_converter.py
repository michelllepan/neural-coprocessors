import wandb

from coprocessors.action_converter.policies import QMaxOnlyPolicy


class RandomActionConverterPolicy(QMaxOnlyPolicy):

    def __init__(
        self,
        world_env,
        coproc_env,
        coproc_test_env,
        action_net,
        save_dir,
        stim_dim,
        temporal,
        gamma,
        train_frequency,
        update_start,
        f_lr,
        healthy_actor,
        healthy_q,
        opt_method="grid_search",
        device="cpu",
    ):
        super().__init__(
            world_env=world_env,
            coproc_env=coproc_env,
            coproc_test_env=coproc_test_env,
            action_net=action_net,
            save_dir=save_dir,
            stim_dim=stim_dim,
            temporal=temporal,
            gamma=gamma,
            q_lr=0.0,
            tau=0.0,
            use_target=False,
            train_frequency=train_frequency,
            update_start=update_start,
            f_lr=f_lr,
            healthy_actor=healthy_actor,
            healthy_q=healthy_q,
            opt_method=opt_method,
            device=device,
        )

    def act(self, obs=None):
        return self.get_random_data()