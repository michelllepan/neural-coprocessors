from typing import Union
import gin

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent

from offpcc.basics.abstract_algorithms import OffPolicyRLAlgorithm
from offpcc.basics.actors_and_critics import MLPGaussianActor, MLPCritic
from offpcc.basics.convnet import ConvNet
from offpcc.basics.replay_buffer import Batch
from offpcc.basics.utils import get_device, create_target, polyak_update, save_net, load_net
from offpcc.basics.lr_scheduler import LRScheduler


@gin.configurable(module=__name__)
class ConvolutionalSAC(OffPolicyRLAlgorithm):

    """
    The autotuning of the entropy coefficient (alpha) follows almost EXACTLY from SB3's SAC implementation, while
    other code follows from spinup's implementation.

    Alpha is called the entropy coefficient, which determines how much the training code cares about
    entropy maximization. Target entropy is NOT a target for alpha; instead, it is a target for the
    policy entropy, which is achieved by tuning alpha appropriately.
    """

    def __init__(
        self,
        input_shape,
        action_dim,
        embedding_dim=50,
        gamma=0.99,
        lr=3e-4,
        lr_schedule=None,
        polyak=0.995,
        alpha=1.0,  # if autotune_alpha, this becomes the initial alpha value
        autotune_alpha: bool = True,
    ):

        # hyperparameters

        self.input_shape = input_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.polyak = polyak

        self.autotune_alpha = autotune_alpha

        if autotune_alpha:

            # Since self.log_alpha must be a leaf variable to be optimized by an optimizer,
            # setting it to require grad is done at last.
            # Ref: see is_leaf attribute at https://pytorch.org/docs/stable/tensors.html

            self.log_alpha = torch.log(torch.ones(1) * alpha).to(get_device()).requires_grad_(True)
            self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        else:
            self.alpha = alpha

        self.target_entropy = - self.action_dim  # int, but it will get broadcasted over a FloatTensor as a float

        # networks

        self.cnn = ConvNet(input_depth=input_shape[0], embedding_dim=embedding_dim).to(get_device())
        self.cnn_targ = create_target(self.cnn)

        self.actor = MLPGaussianActor(input_dim=embedding_dim, action_dim=action_dim).to(get_device())

        self.Q1 = MLPCritic(input_dim=embedding_dim, action_dim=action_dim).to(get_device())
        self.Q1_targ = create_target(self.Q1)

        self.Q2 = MLPCritic(input_dim=embedding_dim, action_dim=action_dim).to(get_device())
        self.Q2_targ = create_target(self.Q2)

        # optimizers

        self.cnn_optimizer = optim.Adam(self.cnn.parameters(), lr=lr)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=lr)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=lr)

        # lr scheduler

        if lr_schedule is not None:
            self.lr_scheduler = LRScheduler(
                optimizers=[self.actor_optimizer, self.Q1_optimizer, self.Q2_optimizer],
                init_lr=lr,
                schedule=lr_schedule
            )

    def sample_action_from_distribution(
            self,
            state: torch.tensor,
            deterministic: bool,
            return_log_prob: bool
    ) -> Union[torch.tensor, tuple]:  # tuple of 2 tensors if return_log_prob is True; else torch.tensor

        # notation (from SAC paper):
        # - mu represents the normal distribution
        # - u represents the un-squashed action; nu stands for next u's

        # reference on log_prob computation code:
        # the following line of code (from SAC paper) is not numerically stable:
        # log_pi_a_given_s = mu_given_s.log_prob(u) - torch.sum(torch.log(1 - torch.tanh(u) ** 2), dim=1)
        # the alternative and equivalent way used in this code is copied from:
        # github.com/vitchyr/rlkit/blob/0073d73235d7b4265cd9abe1683b30786d863ffe/rlkit/torch/distributions.py#L358
        # github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73

        means, stds = self.actor(state)

        if deterministic:
            u = means
        else:
            mu_given_s = Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)
            u = mu_given_s.rsample()

        a = torch.tanh(u).view(-1, self.action_dim)  # shape checking

        if return_log_prob:
            log_pi_a_given_s = mu_given_s.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=1)
            return a, log_pi_a_given_s.view(-1, 1)  # add another dim to match Q values
        else:
            return a

    def act(self, state: np.array, deterministic: bool) -> np.array:
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).float().to(get_device())
            embedding = self.cnn(state)
            action = self.sample_action_from_distribution(embedding, deterministic=deterministic, return_log_prob=False)
            return action.view(-1).cpu().numpy()  # view as 1d -> to cpu -> to numpy

    def get_current_alpha(self):
        if self.autotune_alpha:
            return np.exp(float(self.log_alpha))
        else:
            return self.alpha

    def update_networks(self, b: Batch) -> dict:

        # update learning rate

        if self.lr_schedule is not None:
            self.lr_scheduler.get_new_lr()

        bs = len(b.ns)  # for shape checking

        # By def, "an embedding is a relatively lower-dimensional space into which you can translate high-dimensional
        # vectors" (from google).

        embedding = self.cnn(b.s)
        n_embedding = self.cnn_targ(b.ns)

        # compute predictions

        Q1_predictions = self.Q1(embedding, b.a)
        Q2_predictions = self.Q2(embedding, b.a)

        assert Q1_predictions.shape == (bs, 1)
        assert Q2_predictions.shape == (bs, 1)

        # compute targets

        with torch.no_grad():

            na, log_pi_na_given_ns = self.sample_action_from_distribution(n_embedding, deterministic=False,
                                                                          return_log_prob=True)

            n_min_Q_targ = torch.min(self.Q1_targ(n_embedding, na), self.Q2_targ(n_embedding, na))
            n_sample_entropy = - log_pi_na_given_ns

            targets = b.r + self.gamma * (1 - b.d) * (n_min_Q_targ + self.get_current_alpha() * n_sample_entropy)

            assert na.shape == (bs, self.action_dim)
            assert log_pi_na_given_ns.shape == (bs, 1)
            assert n_min_Q_targ.shape == (bs, 1)
            assert targets.shape == (bs, 1)

        # compute td error

        Q1_loss = torch.mean((Q1_predictions - targets) ** 2)
        Q2_loss = torch.mean((Q2_predictions - targets) ** 2)

        assert Q1_loss.shape == ()
        assert Q2_loss.shape == ()

        # reduce td error

        self.cnn_optimizer.zero_grad()

        self.Q1_optimizer.zero_grad()
        Q1_loss.backward(retain_graph=True)
        self.Q1_optimizer.step()

        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.Q2_optimizer.step()

        self.cnn_optimizer.step()

        # compute policy loss

        a, log_pi_a_given_s = self.sample_action_from_distribution(embedding.detach(),  # actor cannot update cnn
                                                                   deterministic=False, return_log_prob=True)

        min_Q = torch.min(self.Q1(embedding.detach(), a), self.Q2(embedding.detach(), a))
        sample_entropy = - log_pi_a_given_s

        policy_loss = - torch.mean(min_Q + self.get_current_alpha() * sample_entropy)

        assert a.shape == (bs, self.action_dim)
        assert log_pi_a_given_s.shape == (bs, 1)
        assert min_Q.shape == (bs, 1)
        assert policy_loss.shape == ()

        # reduce policy loss

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.autotune_alpha:

            # compute log alpha loss

            # derivation to make things more intuitive
            #
            # alpha_loss = - self.log_alpha * (log_pi_a_given_s.detach() + self.target_entropy)  (eq used in SB3)
            #            = self.log_alpha * (- log_pi_a_given_s.detach() - self.target_entropy)
            #            = self.log_alpha * (sample_entropy - self.target_entropy)
            #            = self.log_alpha * excess_entropy (in expectation) (eq used in this codebase)
            #
            # If excess_entropy > 0, minimum is achieved by settings log_alpha to negative (ideally negative infinity),
            # which corresponds to an alpha value between (0, 1).
            #
            # If excess_entropy < 0, minimum is achieved by setting log_alpha to positive (ideally positive infinity),
            # which corresponds to an alpha value greater than 1.
            #
            # Intuitively, an alpha value in (0, 1) means that entropy maximization is less important than other
            # objectives, while an alpha value greater than 1 means that it is more important.
            #
            # Q: Where does SB3 compute alpha loss?
            # https://github.com/DLR-RM/stable-baselines3/blob/78e8d405d7bf6186c8529ed26967cb17ccbe420c/stable_baselines3/sac/sac.py#L184
            #
            # Q: Why use log_alpha instead of alpha directly?
            # - https://github.com/DLR-RM/stable-baselines3/issues/36
            # - https://github.com/rail-berkeley/softlearning/issues/37

            excess_entropy = sample_entropy.detach() - self.target_entropy
            log_alpha_loss = self.log_alpha * torch.mean(excess_entropy)

            # reduce log alpha loss

            self.log_alpha_optimizer.zero_grad()
            log_alpha_loss.backward()
            self.log_alpha_optimizer.step()

        else:

            log_alpha_loss = 0  # needed only for logging purposes

        # update target networks

        polyak_update(targ_net=self.cnn_targ, pred_net=self.cnn, polyak=self.polyak)
        polyak_update(targ_net=self.Q1_targ, pred_net=self.Q1, polyak=self.polyak)
        polyak_update(targ_net=self.Q2_targ, pred_net=self.Q2, polyak=self.polyak)

        return {
            # for learning the q functions
            '(qfunc) Q1 pred': float(Q1_predictions.mean()),
            '(qfunc) Q2 pred': float(Q2_predictions.mean()),
            '(qfunc) Q1 loss': float(Q1_loss),
            '(qfunc) Q2 loss': float(Q2_loss),
            # for learning the actor
            '(actor) min Q value': float(min_Q.mean()),
            '(actor) entropy (sample)': float(sample_entropy.mean()),
            # for learning the entropy coefficient (alpha)
            '(alpha) alpha': self.get_current_alpha(),
            '(alpha) log alpha loss': float(log_alpha_loss.mean())
        }

    def save_actor(self, save_dir: str) -> None:
        save_net(net=self.cnn, save_dir=save_dir, save_name="cnn.pth")
        save_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")

    def load_actor(self, save_dir: str) -> None:
        load_net(net=self.cnn, save_dir=save_dir, save_name="cnn.pth")
        load_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")
