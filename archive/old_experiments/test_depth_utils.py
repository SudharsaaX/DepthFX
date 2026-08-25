import numpy as np

from depth_utils import normalize_depth


def main():
    depth = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )

    normalized = normalize_depth(depth)

    print("Original depth:")
    print(depth)

    print("\nNormalized depth:")
    print(normalized)

    print("\nMinimum:", normalized.min())
    print("Maximum:", normalized.max())


if __name__ == "__main__":
    main()