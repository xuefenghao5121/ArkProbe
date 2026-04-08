"""Python fallback workloads when C binaries are not available.

WARNING: These workloads run in a Python process. PMU data collected
against them will include Python interpreter overhead (higher cache miss,
more branch mispredictions). Use C binaries for accurate micro-architectural
characterization.
"""

from __future__ import annotations

import sys
import time
import threading

import click
import numpy as np


def _run_workers(target, threads: int, duration: int) -> list:
    """Run target function in N threads for duration seconds."""
    results = [None] * threads
    stop_event = threading.Event()

    def wrapper(idx):
        results[idx] = target(stop_event)

    workers = []
    for i in range(threads):
        t = threading.Thread(target=wrapper, args=(i,), daemon=True)
        workers.append(t)
        t.start()

    time.sleep(duration)
    stop_event.set()

    for t in workers:
        t.join(timeout=5)

    return results


def compute_worker(stop_event: threading.Event) -> dict:
    """Matrix multiply using NumPy (BLAS-backed)."""
    n = 256
    count = 0
    while not stop_event.is_set():
        a = np.random.randn(n, n)
        np.dot(a, a)
        count += 1
    return {"ops": count, "n": n}


def memory_worker(stop_event: threading.Event) -> dict:
    """Streaming array copy."""
    buf_size = 64 * 1024 * 1024  # 64 MB
    src = np.ones(buf_size // 8, dtype=np.float64)
    total_bytes = 0
    while not stop_event.is_set():
        dst = src.copy()
        total_bytes += buf_size
        # Prevent optimization
        if dst[0] < 0:
            break
    return {"bytes": total_bytes}


def mixed_worker(stop_event: threading.Event) -> dict:
    """Alternating compute + memory phases."""
    n = 128
    buf_size = 32 * 1024 * 1024
    src = np.ones(buf_size // 8, dtype=np.float64)
    count = 0
    while not stop_event.is_set():
        # Compute phase
        a = np.random.randn(n, n)
        np.dot(a, a)
        # Memory phase
        dst = src.copy()
        count += 1
        if dst[0] < 0:
            break
    return {"ops": count}


WORKLOADS = {
    "compute": compute_worker,
    "memory": memory_worker,
    "mixed": mixed_worker,
}


@click.command()
@click.option("--workload", "-w", type=click.Choice(list(WORKLOADS.keys())), required=True)
@click.option("--threads", "-t", type=int, default=1)
@click.option("--duration", "-d", type=int, default=60)
def main(workload: str, threads: int, duration: int):
    """Run a Python fallback workload."""
    print(f"WARNING: Using Python fallback for '{workload}'. "
          "PMU data will include interpreter overhead.", file=sys.stderr)

    target = WORKLOADS[workload]
    results = _run_workers(target, threads, duration)

    if workload == "compute":
        total_ops = sum(r["ops"] for r in results if r)
        n = results[0]["n"] if results[0] else 256
        flops = total_ops * 2.0 * n * n * n
        print(f"{flops / (duration * 1e6):.2f} Mflops")
        print(f"{total_ops / duration:.2f} ops/sec")
    elif workload == "memory":
        total_bytes = sum(r["bytes"] for r in results if r)
        print(f"{total_bytes / (duration * 1024 * 1024):.2f} MB/s")
        print(f"{total_bytes / duration:.2f} ops/sec")
    elif workload == "mixed":
        total_ops = sum(r["ops"] for r in results if r)
        print(f"{total_ops / duration:.2f} ops/sec")


if __name__ == "__main__":
    main()
