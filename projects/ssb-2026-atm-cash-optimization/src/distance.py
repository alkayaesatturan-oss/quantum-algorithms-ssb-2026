"""
Section 1.1: "The travel time between two ATMs is computed using the
Haversine distance metric and an average vehicle speed of 35 km/h."
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def travel_minutes_matrix(lats: np.ndarray, lons: np.ndarray, speed_kmh: float) -> np.ndarray:
    """N x N travel time matrix (minutes). lats/lons must be given in the
    same order and should be called with index 0 as VEZNE (the depot); this
    is not strictly required, but the upper layers of this module rely on
    that ordering."""
    n = len(lats)
    lat_i = lats.reshape(n, 1)
    lon_i = lons.reshape(n, 1)
    lat_j = lats.reshape(1, n)
    lon_j = lons.reshape(1, n)
    km = haversine_km(lat_i, lon_i, lat_j, lon_j)
    return km / speed_kmh * 60.0
