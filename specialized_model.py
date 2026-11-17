"""Input-isolated spatial, attribute and feature-common predictive experts."""
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
