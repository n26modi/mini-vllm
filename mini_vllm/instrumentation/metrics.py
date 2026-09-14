import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class RequestMetrics:
    request_id: str
    prompt: str
    max_new_tokens: int
    ttft_ms: float
    itl_ms_per_token: list[float]
    total_tokens: int
    wall_time_s: float
    gpu_util_pct: float
    peak_mem_gb: float

    @property
    def mean_itl_ms(self) -> float:
        if not self.itl_ms_per_token:
            return 0.0
        return sum(self.itl_ms_per_token) / len(self.itl_ms_per_token)

    @property
    def throughput_tok_s(self) -> float:
        if self.wall_time_s == 0:
            return 0.0
        return self.total_tokens / self.wall_time_s


class MetricsLogger:
    def __init__(self, output_path: str | Path):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a")

    def log(self, metrics: RequestMetrics) -> None:
        row = asdict(metrics)
        row["mean_itl_ms"] = metrics.mean_itl_ms
        row["throughput_tok_s"] = metrics.throughput_tok_s
        row["timestamp"] = time.time()
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
