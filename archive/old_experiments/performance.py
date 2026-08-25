import time
from dataclasses import dataclass


@dataclass
class PerformanceStats:
    ai_ms: float = 0.0
    upload_ms: float = 0.0
    render_ms: float = 0.0
    total_ms: float = 0.0
    fps: float = 0.0


class PerformanceProfiler:
    def __init__(self):
        self.ai_times = []
        self.upload_times = []
        self.render_times = []
        self.total_times = []

        self.window_start = time.perf_counter()
        self.frame_count = 0

        self.stats = PerformanceStats()

    def start_frame(self):
        return time.perf_counter()

    def start_stage(self):
        return time.perf_counter()

    def record_ai(self, start):
        self.ai_times.append(
            (time.perf_counter() - start) * 1000.0
        )

    def record_upload(self, start):
        self.upload_times.append(
            (time.perf_counter() - start) * 1000.0
        )

    def record_render(self, start):
        self.render_times.append(
            (time.perf_counter() - start) * 1000.0
        )

    def end_frame(self, start):
        self.total_times.append(
            (time.perf_counter() - start) * 1000.0
        )

        self.frame_count += 1

        elapsed = (
            time.perf_counter()
            - self.window_start
        )

        if elapsed >= 1.0:
            self._update_stats(elapsed)

    def _average(self, values):
        if not values:
            return 0.0

        return sum(values) / len(values)

    def _update_stats(self, elapsed):
        self.stats.ai_ms = self._average(
            self.ai_times
        )

        self.stats.upload_ms = self._average(
            self.upload_times
        )

        self.stats.render_ms = self._average(
            self.render_times
        )

        self.stats.total_ms = self._average(
            self.total_times
        )

        self.stats.fps = (
            self.frame_count / elapsed
        )

        self.ai_times.clear()
        self.upload_times.clear()
        self.render_times.clear()
        self.total_times.clear()

        self.frame_count = 0
        self.window_start = time.perf_counter()

    def get_stats(self):
        return self.stats


def format_stats(stats):
    return (
        f"FPS: {stats.fps:5.1f} | "
        f"AI: {stats.ai_ms:6.1f} ms | "
        f"Upload: {stats.upload_ms:5.1f} ms | "
        f"GPU: {stats.render_ms:5.1f} ms | "
        f"Frame: {stats.total_ms:6.1f} ms"
    )