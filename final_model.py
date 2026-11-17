"""Frozen main architecture selected by the author; historical GECNP stays unchanged."""
import torch
from torch import nn
from geographic_kernel_model import GeographicKernelMDN

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
