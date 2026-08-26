"""Actor-critic with a residual terrain encoder for perceptive locomotion."""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation


class PerceptiveActor(nn.Module):
    """Fuse terrain features without changing the warm-started proprioceptive path."""

    def __init__(
        self,
        proprio_obs_dim,
        terrain_obs_dim,
        num_actions,
        actor_hidden_dims,
        terrain_encoder_dims,
        activation,
    ):
        super().__init__()
        if len(actor_hidden_dims) != 2:
            raise ValueError("PerceptiveActor currently expects two actor hidden layers")
        if len(terrain_encoder_dims) == 0:
            raise ValueError("terrain_encoder_dims must contain at least one dimension")

        hidden_0, hidden_1 = actor_hidden_dims
        self.proprio_obs_dim = proprio_obs_dim
        self.terrain_obs_dim = terrain_obs_dim
        self.proprio_input = nn.Linear(proprio_obs_dim, hidden_0)

        encoder_layers = []
        encoder_input = terrain_obs_dim
        for encoder_output in terrain_encoder_dims:
            encoder_layers.append(nn.Linear(encoder_input, encoder_output))
            encoder_layers.append(get_activation(activation))
            encoder_input = encoder_output
        self.terrain_encoder = nn.Sequential(*encoder_layers)
        self.terrain_adapter = nn.Linear(encoder_input, hidden_0)

        self.activation_0 = get_activation(activation)
        self.proprio_hidden = nn.Linear(hidden_0, hidden_1)
        self.activation_1 = get_activation(activation)
        self.action_head = nn.Linear(hidden_1, num_actions)

        # A transferred rough policy must initially behave exactly like the
        # source policy. Terrain influence is learned gradually from zero.
        nn.init.zeros_(self.terrain_adapter.weight)
        nn.init.zeros_(self.terrain_adapter.bias)

    def split_observation(self, observations):
        expected = self.proprio_obs_dim + self.terrain_obs_dim
        if observations.shape[-1] != expected:
            raise RuntimeError(
                "Perceptive actor received {}D, expected {}D".format(
                    observations.shape[-1], expected
                )
            )
        return torch.split(
            observations,
            [self.proprio_obs_dim, self.terrain_obs_dim],
            dim=-1,
        )

    def forward(self, observations):
        proprioception, terrain = self.split_observation(observations)
        terrain_latent = self.terrain_encoder(terrain)
        fused = self.proprio_input(proprioception) + self.terrain_adapter(terrain_latent)
        hidden = self.activation_0(fused)
        hidden = self.activation_1(self.proprio_hidden(hidden))
        return self.action_head(hidden)


class ActorCriticPerceptive(nn.Module):
    """PPO-compatible asymmetric actor-critic for height-map perception."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        obs_context_len=1,
        proprio_obs_dim=141,
        terrain_obs_dim=63,
        terrain_encoder_dims=(128, 32),
        actor_hidden_dims=(512, 128),
        critic_hidden_dims=(512, 128),
        activation="elu",
        init_noise_std=1.0,
        **kwargs
    ):
        super().__init__()
        ignored = [key for key in kwargs if key not in ("device", "args")]
        if ignored:
            print("ActorCriticPerceptive ignored arguments: {}".format(ignored))
        if num_actor_obs != proprio_obs_dim + terrain_obs_dim:
            raise ValueError(
                "num_actor_obs={} but proprio_obs_dim + terrain_obs_dim={}".format(
                    num_actor_obs, proprio_obs_dim + terrain_obs_dim
                )
            )
        self.obs_context_len = obs_context_len
        self.actor = PerceptiveActor(
            proprio_obs_dim=proprio_obs_dim,
            terrain_obs_dim=terrain_obs_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            terrain_encoder_dims=terrain_encoder_dims,
            activation=activation,
        )

        critic_layers = []
        critic_input = num_critic_obs
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(critic_input, hidden_dim))
            critic_layers.append(get_activation(activation))
            critic_input = hidden_dim
        critic_layers.append(nn.Linear(critic_input, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        print("Perceptive Actor Network: {}".format(self.actor))
        print("Perceptive Critic Network: {}".format(self.critic))

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        if self.obs_context_len != 1:
            observations = observations[..., -1, :]
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        if self.obs_context_len != 1:
            observations = observations[..., -1, :]
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        if self.obs_context_len != 1:
            critic_observations = critic_observations[..., -1, :]
        return self.critic(critic_observations)

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

