"""ANP (Kim et al. 2019) and the released SMACNP core (Bao et al. 2024)."""
import math

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F


def features(s, x):
    return torch.cat((s, x), dim=-1)


def gaussian_mlp(input_dim, width, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, output_dim),
    )


class SelfAttention(nn.Module):
    def __init__(self, input_dim, width):
        super().__init__()
        self.embed = gaussian_mlp(input_dim, width, width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, observations):
        values = self.embed(observations)
        update, _ = self.attention(values, values, values, need_weights=False)
        return self.norm(values + update)


class AttentiveNeuralProcess(nn.Module):
    """Global Gaussian latent path plus query-conditioned deterministic path."""

    def __init__(self, input_dim, width=64):
        super().__init__()
        self.width = width
        self.latent_encoder = SelfAttention(input_dim + 1, width)
        self.latent_parameters = gaussian_mlp(width, width, 2 * width)
        self.context_encoder = SelfAttention(input_dim + 1, width)
        self.position = gaussian_mlp(input_dim, width, width)
        self.cross_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.decoder = gaussian_mlp(2 * width + input_dim, width, 2)

    def latent(self, observations):
        mean, raw_scale = self.latent_parameters(
            self.latent_encoder(observations).mean(1)
        ).chunk(2, -1)
        return Normal(mean, 0.1 + 0.9 * torch.sigmoid(raw_scale))

    def context(self, cx, observations):
        return self.position(cx), self.context_encoder(observations)

    def decode(self, context, tx, z):
        keys, values = context
        query = self.position(tx)
        attended, _ = self.cross_attention(query, keys, values, need_weights=False)
        attended = self.cross_norm(query + attended)
        batch, draws, width = z.shape
        count = tx.shape[1]
        z_query = z[:, :, None, :].expand(batch, draws, count, width)
        deterministic = torch.cat((tx, attended), dim=-1)[:, None].expand(
            -1, draws, -1, -1
        )
        output = self.decoder(torch.cat((deterministic, z_query), dim=-1))
        mu = output[..., 0].transpose(1, 2)
        sigma = (0.001 + F.softplus(output[..., 1])).transpose(1, 2)
        return mu, sigma

    def training_loss(self, cs, cx, cy, ts, tx, ty):
        cx, tx = features(cs, cx), features(ts, tx)
        observations = torch.cat((cx, cy), dim=-1)
        targets = torch.cat((tx, ty[..., None]), dim=-1)
        prior = self.latent(observations)
        posterior = self.latent(torch.cat((observations, targets), dim=1))
        mu, sigma = self.decode(
            self.context(cx, observations), tx, posterior.rsample()[:, None]
        )
        log_likelihood = Normal(mu[..., 0], sigma[..., 0]).log_prob(ty).sum(dim=1)
        kl = kl_divergence(posterior, prior).sum(dim=-1)
        return ((kl - log_likelihood) / ty.shape[1]).mean()

    def predictions(self, cs, cx, cy, ts, tx, latent_noise):
        cx, tx = features(cs, cx), features(ts, tx)
        observations = torch.cat((cx, cy), dim=-1)
        prior = self.latent(observations)
        z = prior.loc[:, None, :] + prior.scale[:, None, :] * latent_noise[None]
        context = self.context(cx, observations)
        parts = [self.decode(context, chunk, z) for chunk in tx.split(256, dim=1)]
        return tuple(torch.cat(group, dim=1) for group in zip(*parts))


class SMACNP(nn.Module):
    """Original mean/variance pathways with the released variance semantics."""

    def __init__(self, attr_dim, width=128):
        super().__init__()
        from baselines._vendor.smacnp_upstream.model_smanp import (
            LocationEncoder,
            SpatialNeuralProcess,
        )

        class MemoryBoundLocationEncoder(LocationEncoder):
            """Compute the same projected Laplace weights in bounded query blocks."""

            def forward(self, context_x, context_y, target_x):
                values = self.input_projection(
                    torch.cat((context_x[..., :2], context_y), dim=-1)
                )
                key = self.context_projection(context_x[..., :2])
                query = self.context_projection(target_x[..., :2])
                # The original implementation materializes an Nq by Nc by width
                # tensor. cdist evaluates the same L1 scores and retains their gradients.
                weights = torch.softmax(-0.5 * torch.cdist(query, key, p=1), dim=-1)
                return self.target_projection(torch.bmm(weights, values))

        self.core = SpatialNeuralProcess(attr_dim, 1, width)
        location = MemoryBoundLocationEncoder(width)
        location.load_state_dict(self.core.location.state_dict())
        self.core.location = location

    def forward(self, cx, cy, tx):
        mu, variance = self.core(cx, cy, tx)
        return mu[..., None], variance.sqrt()[..., None]

    def training_loss(self, cs, cx, cy, ts, tx, ty):
        mu, sigma = self(features(cs, cx), cy, features(ts, tx))
        return -Normal(mu[..., 0], sigma[..., 0]).log_prob(ty).mean()

    def predictions(self, cs, cx, cy, ts, tx, latent_noise):
        cx, tx = features(cs, cx), features(ts, tx)
        parts = [self(cx, cy, chunk) for chunk in tx.split(256, dim=1)]
        return tuple(torch.cat(group, dim=1) for group in zip(*parts))


def marginal_nll(mu, sigma, y):
    log_prob = Normal(mu, sigma).log_prob(y[..., None])
    return (math.log(mu.shape[-1]) - torch.logsumexp(log_prob, dim=-1)).mean()
