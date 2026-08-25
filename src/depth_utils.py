import numpy as np


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """
    Normalize a raw depth map to the range 0.0 - 1.0.

    The minimum depth becomes 0.0 and the maximum depth
    becomes 1.0.
    """
    depth = depth.astype(np.float32)

    min_depth = np.min(depth)
    max_depth = np.max(depth)

    depth_range = max_depth - min_depth

    if depth_range < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)

    normalized = (depth - min_depth) / depth_range

    return np.clip(normalized, 0.0, 1.0)