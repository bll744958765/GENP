"""Training-context scale for the California Haversine ablation."""
import numpy as np
import pandas as pd
from data import DATA_DIR, DATASETS
from geographic_kernel_model import EARTH_RADIUS_KM


def california_geometry_metadata(data, dataset):
    frame = pd.read_csv(DATA_DIR / DATASETS[dataset])
    if list(frame.columns[1:3]) != ["Latitude", "Longitude"]:
        raise ValueError("Latitude/Longitude degree columns required")
    context = (
        frame.set_index(frame.columns[0])
        .loc[data.context_ids]
        .iloc[:, :2]
        .to_numpy(dtype=np.float64)
    )
    mean = context.mean(0)
    sd = context.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    reconstructed = data.context_s[0].cpu().numpy() * sd + mean
    assert np.max(np.abs(reconstructed - context)) < 1e-4
    # A single train-context scalar keeps distances proportional to kilometers.
    # It does not independently rescale latitude and longitude in the metric.
    latitude_scale = EARTH_RADIUS_KM * np.deg2rad(sd[0])
    longitude_scale = EARTH_RADIUS_KM * np.cos(np.deg2rad(mean[0])) * np.deg2rad(sd[1])
    scale = float(np.sqrt((latitude_scale**2 + longitude_scale**2) / 2))
    return mean, sd, scale
