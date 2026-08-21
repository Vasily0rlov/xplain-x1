"""Box etiquette (project CLAUDE.md): wait for the box before batteries, then use all of it."""
from __future__ import annotations

import os
import time


def load_avg_1min() -> float:
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def wait_until_free(load_threshold: float = 8.0, check_interval_s: int = 60,
                    max_wait_s: int = 24 * 3600, verbose: bool = True) -> None:
    """Block until 1-min loadavg < threshold (sister projects may be running)."""
    start = time.time()
    while True:
        load = load_avg_1min()
        if load < load_threshold:
            if verbose:
                print(f"[box] load {load:.2f} < {load_threshold} — proceeding")
            return
        if time.time() - start > max_wait_s:
            raise TimeoutError(f"box busy for >{max_wait_s}s (load {load:.2f})")
        if verbose:
            print(f"[box] load {load:.2f} >= {load_threshold} — waiting {check_interval_s}s")
        time.sleep(check_interval_s)


def pin_threads(n_workers: int, total_threads: int | None = None) -> int:
    """Threads per worker so n_workers x threads never oversubscribes the box."""
    total = total_threads or os.cpu_count() or 1
    per_worker = max(1, total // max(1, n_workers))
    os.environ["OMP_NUM_THREADS"] = str(per_worker)
    os.environ["MKL_NUM_THREADS"] = str(per_worker)
    try:
        import torch

        torch.set_num_threads(per_worker)
    except ImportError:
        pass
    return per_worker
