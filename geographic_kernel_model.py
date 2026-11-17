"""Factorial attention-kernel comparison within the fixed common/local MDN."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from specialized_model import SpecializedGECNP

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
