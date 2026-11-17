"""GECNP and its architecture-matched controls."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def net(n, h, out):
    return nn.Sequential(
        nn.Linear(n, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, out)
    )


class SpecializedGECNP(nn.Module):
    def __init__(
        self,
        coord_dim,
        attr_dim,
        hidden=64,
        components=2,
        shared_base=False,
        gate_mode="learned",
        fusion_mode="predictive",
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        if fusion_mode == "latent" and components != 2:
            raise ValueError(
                "The common-plus-gated-local model uses exactly two Gaussian components"
            )
        self.components = components
        self.gate_mode = gate_mode
        self.adapters = nn.ModuleList(
            [
                nn.Linear(coord_dim + 1, hidden),
                nn.Linear(attr_dim + 1, hidden),
                nn.Linear(coord_dim + attr_dim, hidden),
            ]
        )
        self.shared_base = shared_base
        if shared_base:
            self.base = net(hidden, hidden, hidden)
            self.bases = None
        else:
            self.base = None
            self.bases = nn.ModuleList([net(hidden, hidden, hidden) for _ in range(3)])
        self.transforms = nn.ModuleList([net(hidden, hidden, hidden) for _ in range(3)])
        self.queries = nn.ModuleList(
            [
                net(coord_dim, hidden, hidden),
                net(attr_dim, hidden, hidden),
                net(coord_dim + attr_dim, hidden, hidden),
            ]
        )
        self.fusions = nn.ModuleList(
            [
                net(2 * hidden, hidden, hidden)
                for _ in range(2 if fusion_mode == "latent" else 3)
            ]
        )
        if fusion_mode == "latent":
            self.information_fusion = net(3 * hidden, hidden, hidden)
            self.prediction_head = net(hidden, hidden, 3 * components)
        else:
            self.heads = nn.ModuleList(
                [net(hidden, hidden, 3 * components) for _ in range(3)]
            )
        self.attr_key = net(attr_dim, hidden, hidden)
        self.attr_query = net(attr_dim, hidden, hidden)
        self.log_distance_temperature = nn.Parameter(torch.tensor(math.log(0.15)))
        self.log_similarity_temperature = nn.Parameter(torch.tensor(math.log(0.2)))
        self.gate = (
            net(4 * hidden, hidden, 2)
            if fusion_mode == "latent"
            else net(3 * hidden + 6, hidden, 3)
        )

    def attention(self, context_s, context_x, target_s, target_x):
        d2 = torch.cdist(target_s, context_s).square()
        wd = torch.softmax(
            -d2 / self.log_distance_temperature.exp().clamp_min(1e-4), -1
        )
        keys = F.normalize(self.attr_key(context_x), dim=-1, eps=1e-8)
        queries = F.normalize(self.attr_query(target_x), dim=-1, eps=1e-8)
        wa = torch.softmax(
            torch.bmm(queries, keys.transpose(1, 2))
            / self.log_similarity_temperature.exp().clamp_min(1e-4),
            -1,
        )
        return wd, wa

    def forward(self, context_s, context_x, context_y, target_s, target_x):
        # No query target y is accepted. Each candidate density has its own
        # restricted forward inputs; the router can compare all candidates.
        inputs = [
            torch.cat((context_s, context_y), -1),
            torch.cat((context_x, context_y), -1),
            torch.cat((context_s, context_x), -1),
        ]
        query_inputs = [target_s, target_x, torch.cat((target_s, target_x), -1)]
        values = []
        for i, v in enumerate(inputs):
            v = F.gelu(self.adapters[i](v))
            v = self.base(v) if self.shared_base else self.bases[i](v)
            values.append(self.transforms[i](v))
        wd, wa = self.attention(context_s, context_x, target_s, target_x)
        messages = [
            torch.bmm(wd, values[0]),
            torch.bmm(wa, values[1]),
            values[2].mean(1, keepdim=True).expand(-1, target_s.shape[1], -1),
        ]
        if self.fusion_mode == "latent":
            qd = self.queries[0](target_s)
            qa = self.queries[1](target_x)
            rd = self.fusions[0](torch.cat((messages[0], qd), -1))
            ra = self.fusions[1](torch.cat((messages[1], qa), -1))
            if self.gate_mode == "uniform":
                gate = torch.full_like(target_s[..., :1].expand(-1, -1, 2), 0.5)
            else:
                gate = torch.softmax(self.gate(torch.cat((rd, ra, qd, qa), -1)), -1)
            local = gate[..., 0:1] * rd + gate[..., 1:2] * ra
            qcommon = self.queries[2](query_inputs[2])
            representation = self.information_fusion(
                torch.cat((messages[2], local, qcommon), -1)
            )
            mu, raw_sigma, logits = self.prediction_head(representation).chunk(3, -1)
            sigma = F.softplus(raw_sigma) + 1e-3
            lp = F.log_softmax(logits, -1)
            pi = lp.exp()
            mean = (pi * mu).sum(-1)
            # Central form of total variance avoids subtracting large squares.
            variance = (
                (pi * (sigma.square() + (mu - mean.unsqueeze(-1)).square()))
                .sum(-1)
                .clamp_min(1e-8)
            )
            return dict(
                mu=mu,
                sigma=sigma,
                log_pi=lp,
                mean=mean,
                variance=variance,
                std=variance.sqrt(),
                gate_weights=gate,
                representation=representation,
                spatial_weights=wd,
                attribute_weights=wa,
                spatial_representation=rd,
                attribute_representation=ra,
                common_representation=messages[2],
                local_representation=local,
            )
        reps = []
        mus = []
        sigmas = []
        logs = []
        means = []
        stds = []
        for i in range(3):
            q = self.queries[i](query_inputs[i])
            r = self.fusions[i](torch.cat((messages[i], q), -1))
            reps.append(r)
            mu, raw_sigma, logits = self.heads[i](r).chunk(3, -1)
            sigma = F.softplus(raw_sigma) + 1e-3
            lp = F.log_softmax(logits, -1)
            pi = lp.exp()
            mean = (pi * mu).sum(-1, keepdim=True)
            var = (pi * (sigma.square() + mu.square())).sum(
                -1, keepdim=True
            ) - mean.square()
            mus.append(mu)
            sigmas.append(sigma)
            logs.append(lp)
            means.append(mean)
            stds.append(var.clamp_min(1e-8).sqrt())
        if self.gate_mode == "uniform":
            gate = torch.full_like(torch.cat(means, -1), 1 / 3)
        else:
            gate = torch.softmax(self.gate(torch.cat(reps + means + stds, -1)), -1)
        mu = torch.cat(mus, -1)
        sigma = torch.cat(sigmas, -1)
        lp = torch.cat(
            [logs[i] + gate[..., i : i + 1].clamp_min(1e-12).log() for i in range(3)],
            -1,
        )
        pi = lp.exp()
        mean = (pi * mu).sum(-1)
        variance = (
            (pi * (sigma.square() + mu.square())).sum(-1) - mean.square()
        ).clamp_min(1e-8)
        return dict(
            mu=mu,
            sigma=sigma,
            log_pi=lp,
            mean=mean,
            variance=variance,
            std=variance.sqrt(),
            gate_weights=gate,
            representation=sum(gate[..., i : i + 1] * reps[i] for i in range(3)),
            spatial_weights=wd,
            attribute_weights=wa,
            expert_mu=torch.stack(mus, -2),
            expert_sigma=torch.stack(sigmas, -2),
            expert_log_pi=torch.stack(logs, -2),
        )


EARTH_RADIUS_KM = 6371.0088


def haversine_km(a, b):
    # Both inputs: batch x points x (latitude, longitude), in degrees.
    ar = torch.deg2rad(a.to(torch.float64))
    br = torch.deg2rad(b.to(torch.float64))
    dlat = ar[:, :, None, 0] - br[:, None, :, 0]
    dlon = ar[:, :, None, 1] - br[:, None, :, 1]
    h = (
        torch.sin(dlat / 2).square()
        + torch.cos(ar[:, :, None, 0])
        * torch.cos(br[:, None, :, 0])
        * torch.sin(dlon / 2).square()
    ).clamp(0, 1)
    return EARTH_RADIUS_KM * 2 * torch.atan2(h.sqrt(), (1 - h).clamp_min(0).sqrt())


class GeographicKernelMDN(SpecializedGECNP):
    def __init__(
        self,
        coord_dim,
        attr_dim,
        coordinate_mean,
        coordinate_std,
        geometry_scale_km,
        geometry="standardized",
        spatial_kernel="gaussian",
        attribute_kernel="cosine",
        hidden=64,
        gate_mode="learned",
    ):
        # Construct common modules in the exact same RNG order for every variant.
        super().__init__(
            coord_dim,
            attr_dim,
            hidden,
            components=2,
            shared_base=True,
            gate_mode=gate_mode,
            fusion_mode="latent",
        )
        self.geometry = geometry
        self.spatial_kernel = spatial_kernel
        self.attribute_kernel = attribute_kernel
        self.register_buffer(
            "coordinate_mean",
            torch.as_tensor(coordinate_mean, dtype=torch.float64).reshape(1, 1, 2),
        )
        self.register_buffer(
            "coordinate_std",
            torch.as_tensor(coordinate_std, dtype=torch.float64).reshape(1, 1, 2),
        )
        self.register_buffer(
            "geometry_scale_km",
            torch.as_tensor(float(geometry_scale_km), dtype=torch.float64),
        )
        if attribute_kernel == "weighted_l1":
            # Remove attention networks, not their random draws: common weights
            # remain paired but the simpler attribute metric has fewer parameters.
            del self.attr_key
            del self.attr_query
            self.attribute_logits = nn.Parameter(torch.zeros(attr_dim))

    def coordinate_distance(self, target_s, context_s):
        if self.geometry == "standardized":
            return torch.cdist(target_s, context_s)
        if self.geometry != "haversine":
            raise ValueError(self.geometry)
        target = target_s.to(torch.float64) * self.coordinate_std + self.coordinate_mean
        context = (
            context_s.to(torch.float64) * self.coordinate_std + self.coordinate_mean
        )
        return (haversine_km(target, context) / self.geometry_scale_km).to(
            target_s.dtype
        )

    def attribute_distance(self, target_x, context_x):
        weights = torch.softmax(self.attribute_logits, dim=0)
        return torch.cdist(target_x * weights, context_x * weights, p=1)

    def attention(self, context_s, context_x, target_s, target_x):
        distance = self.coordinate_distance(target_s, context_s)
        bandwidth_sq = self.log_distance_temperature.exp().clamp_min(1e-4)
        if self.spatial_kernel == "gaussian":
            log_spatial = -distance.square() / bandwidth_sq
        elif self.spatial_kernel == "exponential":
            log_spatial = -distance / bandwidth_sq.sqrt()
        else:
            raise ValueError(self.spatial_kernel)
        wd = torch.softmax(log_spatial, dim=-1)
        temperature = self.log_similarity_temperature.exp().clamp_min(1e-4)
        if self.attribute_kernel == "cosine":
            key = F.normalize(self.attr_key(context_x), dim=-1, eps=1e-8)
            query = F.normalize(self.attr_query(target_x), dim=-1, eps=1e-8)
            score = torch.bmm(query, key.transpose(1, 2)) / temperature
        elif self.attribute_kernel == "weighted_l1":
            score = -self.attribute_distance(target_x, context_x) / temperature
        else:
            raise ValueError(self.attribute_kernel)
        return wd, torch.softmax(score, dim=-1)


FINAL_CONFIG = dict(
    geometry="standardized",
    spatial_kernel="gaussian",
    attribute_kernel="weighted_l1",
    components=2,
    shared_base=True,
    hidden=64,
)


class FinalGECNP(GeographicKernelMDN):
    """Final model with nonnegative weighted attribute-distance attention.

    Global feature weights softmax(theta) define the attribute distance.
    Context attention is softmax(-distance / temperature), evaluated separately
    for every query and the currently available context set. Feature weights
    themselves are not query-conditioned in this frozen main specification.
    """

    ATTRIBUTE_ATTENTION_NAME = "nonnegative weighted attribute-distance attention"

    def __init__(
        self,
        coord_dim,
        attr_dim,
        hidden=64,
        gate_mode="learned",
        coordinate_mean=None,
        coordinate_std=None,
    ):
        if gate_mode not in ("learned", "uniform"):
            raise ValueError(
                "Use learned routing or the explicitly named uniform ablation"
            )
        super().__init__(
            coord_dim,
            attr_dim,
            [0.0, 0.0] if coordinate_mean is None else coordinate_mean,
            [1.0, 1.0] if coordinate_std is None else coordinate_std,
            geometry_scale_km=1.0,
            geometry="standardized",
            spatial_kernel="gaussian",
            attribute_kernel="weighted_l1",
            hidden=hidden,
            gate_mode=gate_mode,
        )


class SingleGaussianControl(FinalGECNP):
    """Same encoder, fusion and shared head trunk; one Gaussian output ablation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.prediction_head[-1]
        new = nn.Linear(old.in_features, 3)
        with torch.no_grad():
            # Initialize from component 1 of the identical two-component trunk.
            rows = torch.tensor([0, 2, 4])
            new.weight.copy_(old.weight[rows])
            new.bias.copy_(old.bias[rows])
        self.prediction_head[-1] = new
        self.components = 1
