import time
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2
from depth_utils import normalize_depth

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "depth_anything_v2_vits.pth"

MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}

INFERENCE_SIZE = 320
USE_FP16 = True
WARMUP_FRAMES = 10
BENCHMARK_FRAMES = 100


class DepthEstimator:
    def __init__(self, input_size: int = INFERENCE_SIZE):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size = input_size

        print("Loading Depth Anything V2 Small...")
        print(f"Device: {self.device}")
        print(f"AI inference size: {self.input_size}")

        if self.device == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
            print("CUDA:", torch.version.cuda)
            print("Compute capability:", torch.cuda.get_device_capability(0))

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Depth Anything V2 checkpoint not found:\n{CHECKPOINT_PATH}"
            )

        self.model = DepthAnythingV2(**MODEL_CONFIG)

        print("Loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

        if not isinstance(checkpoint, dict):
            raise RuntimeError("Unexpected checkpoint format.")

        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)

        self.use_fp16 = self.device == "cuda" and USE_FP16

        # Note: Do NOT call self.model.half().
        # Depth Anything V2's infer_image() preprocessing produces FP32 tensors.
        # CUDA autocast handles FP16 computation safely without input type mismatch.
        if self.use_fp16:
            print("Using FP16 autocast inference.")
        else:
            print("Using FP32 inference.")

        self.model.eval()

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        print("Depth model loaded successfully.")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate normalized relative depth.

        Parameters
        ----------
        frame : np.ndarray
            BGR OpenCV image.

        Returns
        -------
        np.ndarray
            Float32 depth map normalized to [0, 1].
        """
        if frame is None:
            raise ValueError("Input frame is None.")
        if frame.size == 0:
            raise ValueError("Input frame is empty.")

        with torch.inference_mode():
            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    raw_depth = self.model.infer_image(frame, input_size=self.input_size)
            else:
                raw_depth = self.model.infer_image(frame, input_size=self.input_size)

        depth = normalize_depth(raw_depth)
        depth = np.asarray(depth, dtype=np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
        depth = np.clip(depth, 0.0, 1.0)

        return depth

    def warmup(self, frame: np.ndarray, iterations: int = WARMUP_FRAMES):
        print("Warming up GPU...")
        for _ in range(iterations):
            self.estimate(frame)

        if self.device == "cuda":
            torch.cuda.synchronize()

        print("GPU warm-up complete.")

    def benchmark(self, frame: np.ndarray, iterations: int = BENCHMARK_FRAMES):
        print()
        print("=" * 60)
        print("Depth Anything V2 Benchmark")
        print("=" * 60)
        print(f"Inference size: {self.input_size}")

        self.warmup(frame, WARMUP_FRAMES)

        times = []
        print("Running benchmark...")

        for _ in range(iterations):
            if self.device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            self.estimate(frame)

            if self.device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000.0)

        times = np.asarray(times, dtype=np.float64)
        average_ms = float(np.mean(times))
        fastest_ms = float(np.min(times))
        slowest_ms = float(np.max(times))
        estimated_fps = 1000.0 / average_ms if average_ms > 0 else 0.0

        print()
        print(f"Average AI time: {average_ms:.2f} ms")
        print(f"Estimated AI FPS: {estimated_fps:.2f}")
        print(f"Fastest frame: {fastest_ms:.2f} ms")
        print(f"Slowest frame: {slowest_ms:.2f} ms")

        if self.device == "cuda":
            allocated_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
            print(f"VRAM allocated: {allocated_gb:.2f} GB")
            print(f"VRAM reserved: {reserved_gb:.2f} GB")

        print("=" * 60)

        return {
            "average_ms": average_ms,
            "fps": estimated_fps,
            "fastest_ms": fastest_ms,
            "slowest_ms": slowest_ms,
        }


def main():
    print("=" * 60)
    print("DepthFX - Optimized Depth Estimator")
    print("=" * 60)

    estimator = DepthEstimator(input_size=INFERENCE_SIZE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    success, frame = cap.read()
    if not success:
        cap.release()
        raise RuntimeError("Could not capture webcam frame.")

    print()
    print("Webcam opened.")
    print(f"Camera resolution: {frame.shape[1]}x{frame.shape[0]}")
    print()
    print("Starting benchmark...")

    estimator.benchmark(frame, BENCHMARK_FRAMES)

    cap.release()
    if estimator.device == "cuda":
        torch.cuda.empty_cache()

    print()
    print("Depth benchmark completed.")


if __name__ == "__main__":
    main()