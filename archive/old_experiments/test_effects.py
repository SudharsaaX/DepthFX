import cv2
import numpy as np

from effects import apply_depth_fog


def main():
    frame = cv2.imread(
        "assets/images/depth_test.png"
    )

    if frame is None:
        raise RuntimeError(
            "Could not load assets/images/depth_test.png"
        )

    height, width = frame.shape[:2]

    depth = np.tile(
        np.linspace(
            0.0,
            1.0,
            width,
            dtype=np.float32,
        ),
        (height, 1),
    )

    fogged = apply_depth_fog(
        frame,
        depth,
        fog_strength=0.8,
    )

    cv2.imshow("Original", frame)
    cv2.imshow("Depth Fog Test", fogged)

    print("Fog effect test successful.")
    print("Press any key to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()