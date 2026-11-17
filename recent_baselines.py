"""Covariate-aware adapters around pinned, unmodified numerical author cores.

SFConvCNP/SFVConvCNP: Mohseni et al., ICML 2026, arXiv:2606.01172.
TE-ISTNP: Ashman et al., ICML 2024; the 2026 authors' reference implementation
adapted from cambridge-mlg/tetnp. These are explicitly scaled, covariate-aware
adaptations, not reproductions of the papers' original benchmark numbers.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

COMPAT = Path(__file__).resolve().parent / "external/fourier_volterra_py39"
sys.path.insert(0, str(COMPAT))
from nps.core.mlp import MLP
from nps.core.cnns.sfcnn import SetFourierConvNet
from nps.core.convolution_blocks import SetFourierConvBlock, SetFourierVolterraConvBlock
from nps.core.attentions import MultiHeadTEAttention
from nps.core.attention_layers import MultiHeadTEAttentionLayer
from nps.core.transformers import TEISTransformerEncoder, PseudoTokenInitialiser

SOURCE_COMMIT = "25c3b0bd4b27a3eda0cea099effc2ee450a87c49"
BASELINE_SPEC = {
    "sfconvcnp": dict(
        year=2026, embed=32, blocks=2, max_freq=2.0, freq_resolution=0.5, groups=4
    ),
    "sfvconvcnp": dict(
        year=2026,
        embed=32,
        blocks=2,
        max_freq=2.0,
        freq_resolution=0.5,
        groups=4,
        volterra_rank=2,
    ),
    "te_istnp": dict(
        year=2024, embed=64, blocks=2, num_heads=4, head_dim=16, num_pseudo_tokens=32
    ),
}


class RecentCovariateNP(nn.Module):
    def __init__(self, coord_dim, attr_dim, variant):
        super().__init__()
        if coord_dim != 2:
            raise ValueError("This study uses two spatial coordinate axes")
        self.variant = variant
        config = BASELINE_SPEC[variant]
        h = config["embed"]
        # At query locations, only y is masked; observed covariates stay available.
        self.value_encoder = MLP(attr_dim + 2, h, num_layers=2, layer_dim=h)
        self.head = MLP(h, 2, num_layers=2, layer_dim=h)
        if variant == "te_istnp":
            kernel = MLP(
                config["num_heads"] + coord_dim,
                config["num_heads"],
                num_layers=2,
                layer_dim=h,
            )
            attention = MultiHeadTEAttention(
                q_dim=h, k_dim=h, v_dim=h, num_heads=4, head_dim=16, kernel=kernel
            )
            layer = MultiHeadTEAttentionLayer(
                attention=attention, embed_dim=h, feedforward_dim=4 * h, norm_first=True
            )
            self.operator = TEISTransformerEncoder(
                layer=layer,
                num_layers=2,
                num_pseudo_tokens=32,
                x_dim=coord_dim,
                pseudo_token_initialiser=PseudoTokenInitialiser(
                    embed_dim=h, num_heads=4, head_dim=16
                ),
            )
        else:
            kind = (
                SetFourierConvBlock
                if variant == "sfconvcnp"
                else SetFourierVolterraConvBlock
            )
            kw = dict(
                spatial_dim=coord_dim,
                embed_channels=h,
                feedforward_channels=4 * h,
                max_freq=2.0,
                freq_resolution=0.5,
                groups=4,
                norm_first=True,
                init_lengthscale=0.05,
                learnable_lengthscale=True,
                share_params=False,
            )
            kw.pop("share_params")
            if variant == "sfvconvcnp":
                kw.update(volterra_rank=2, activation=nn.Identity())
            self.operator = SetFourierConvNet(
                block=kind(**kw), num_blocks=2, share_params=False
            )

    def forward(self, context_s, context_x, context_y, target_s, target_x):
        observed = torch.cat((context_y, context_x, torch.zeros_like(context_y)), -1)
        query = torch.cat(
            (
                torch.zeros_like(target_x[..., :1]),
                target_x,
                torch.ones_like(target_x[..., :1]),
            ),
            -1,
        )
        zc = self.value_encoder(observed)
        zq = self.value_encoder(query)
        representation = self.operator(zc=zc, zq=zq, xc=context_s, xq=target_s)
        mu, raw = self.head(representation).chunk(2, -1)
        sigma = F.softplus(raw) + 0.001
        variance = sigma.square().squeeze(-1)
        mean = mu.squeeze(-1)
        return dict(
            mu=mu,
            sigma=sigma,
            log_pi=torch.zeros_like(mu),
            mean=mean,
            variance=variance,
            std=sigma.squeeze(-1),
        )


def real_parameter_count(model):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
