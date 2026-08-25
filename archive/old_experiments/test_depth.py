from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "depth_anything_v2_vits.pth"

MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}


def main():
    print("=" * 60)
    print("DepthFX - Single Frame Depth Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading model...")

    model = DepthAnythingV2(**MODEL_CONFIG)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Capturing webcam frame...")

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError("Could not capture webcam frame.")

    print(f"Input frame shape: {frame.shape}")

    print("Running depth inference...")

    depth = model.infer_image(frame)

    print(f"Depth map shape: {depth.shape}")
    print(f"Depth data type: {depth.dtype}")
    print(f"Depth minimum: {depth.min():.4f}")
    print(f"Depth maximum: {depth.max():.4f}")

    depth_normalized = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    depth_colored = cv2.applyColorMap(
        depth_normalized,
        cv2.COLORMAP_INFERNO,
    )

    cv2.imwrite(
        str(BASE_DIR / "assets" / "images" / "depth_test.png"),
        depth_colored,
    )

    print("Depth visualization saved:")
    print(
        BASE_DIR
        / "assets"
        / "images"
        / "depth_test.png"
    )

    cv2.imshow("DepthFX - Depth Test", depth_colored)

    print("Press any key in the image window to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Test completed successfully.")


if __name__ == "__main__":
    main()