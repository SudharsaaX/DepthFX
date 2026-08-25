import cv2
import numpy as np


def _prepare_depth(
    depth: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    """
    Prepare a normalized depth map so that it matches
    the frame dimensions.

    Expected depth range:
        0.0 -> 1.0
    """

    if depth is None or depth.size == 0:
        raise ValueError("Invalid depth map.")

    if depth.shape[:2] != frame.shape[:2]:
        depth = cv2.resize(
            depth,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    depth = depth.astype(np.float32)

    depth = np.clip(
        depth,
        0.0,
        1.0,
    )

    return depth


def apply_depth_fog(
    frame: np.ndarray,
    depth: np.ndarray,
    fog_strength: float = 0.65,
    fog_start: float = 0.20,
) -> np.ndarray:
    """
    Apply depth-aware fog.

    The normalized Depth Anything V2 output is used
    to determine how much fog is applied to each pixel.

    Lower depth values are treated as farther regions,
    so farther regions receive more fog.

    Args:
        frame:
            BGR image from OpenCV.

        depth:
            Normalized depth map in the range [0, 1].

        fog_strength:
            Maximum fog intensity.

        fog_start:
            Distance threshold at which fog begins.

    Returns:
        BGR image with depth-aware fog.
    """

    if frame is None or frame.size == 0:
        raise ValueError("Invalid input frame.")

    if not 0.0 <= fog_strength <= 1.0:
        raise ValueError(
            "fog_strength must be between 0 and 1."
        )

    if not 0.0 <= fog_start <= 1.0:
        raise ValueError(
            "fog_start must be between 0 and 1."
        )

    depth = _prepare_depth(
        depth,
        frame,
    )

    # Lower depth values represent farther regions.
    far_depth = 1.0 - depth

    # Start fog after the selected threshold.
    fog_range = max(
        1.0 - fog_start,
        1e-6,
    )

    fog_amount = np.clip(
        (far_depth - fog_start) / fog_range,
        0.0,
        1.0,
    )

    # Smooth the transition.
    fog_amount = fog_amount * fog_amount

    fog_amount *= fog_strength

    # H x W -> H x W x 1
    fog_amount = fog_amount[..., np.newaxis]

    # Neutral white/gray fog.
    fog_color = np.full_like(
        frame,
        220,
        dtype=np.uint8,
    )

    result = (
        frame.astype(np.float32)
        * (1.0 - fog_amount)
        + fog_color.astype(np.float32)
        * fog_amount
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def apply_depth_blur(
    frame: np.ndarray,
    depth: np.ndarray,
    blur_strength: float = 0.85,
) -> np.ndarray:
    """
    Apply depth-aware blur.

    Farther regions receive stronger blur while
    closer regions remain relatively sharp.

    Args:
        frame:
            BGR image from OpenCV.

        depth:
            Normalized depth map in the range [0, 1].

        blur_strength:
            Maximum blur intensity.

    Returns:
        BGR image with depth-dependent blur.
    """

    if frame is None or frame.size == 0:
        raise ValueError("Invalid input frame.")

    if not 0.0 <= blur_strength <= 1.0:
        raise ValueError(
            "blur_strength must be between 0 and 1."
        )

    depth = _prepare_depth(
        depth,
        frame,
    )

    # Lower depth values represent farther regions.
    far_depth = 1.0 - depth

    # Smooth the depth mask so the blur transition
    # does not produce harsh edges.
    far_depth = cv2.GaussianBlur(
        far_depth,
        (0, 0),
        sigmaX=5.0,
    )

    # Create a blurred version of the original image.
    blurred = cv2.GaussianBlur(
        frame,
        (0, 0),
        sigmaX=12.0,
        sigmaY=12.0,
    )

    # Determine how much blur each pixel receives.
    blend_amount = (
        far_depth * blur_strength
    )

    # H x W -> H x W x 1
    blend_amount = blend_amount[
        ...,
        np.newaxis,
    ]

    result = (
        frame.astype(np.float32)
        * (1.0 - blend_amount)
        + blurred.astype(np.float32)
        * blend_amount
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def apply_depth_effects(
    frame: np.ndarray,
    depth: np.ndarray,
    fog_strength: float = 0.65,
    blur_strength: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply both depth-aware fog and depth-aware blur.

    This helper is useful when both effects are needed
    for the same frame.

    Returns:
        fogged_frame, blurred_frame
    """

    fogged_frame = apply_depth_fog(
        frame,
        depth,
        fog_strength=fog_strength,
    )

    blurred_frame = apply_depth_blur(
        frame,
        depth,
        blur_strength=blur_strength,
    )

    return (
        fogged_frame,
        blurred_frame,
    )