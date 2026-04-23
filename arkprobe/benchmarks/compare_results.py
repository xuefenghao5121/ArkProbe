#!/usr/bin/env python3
"""Compare baseline (Java-only) and accelerated (Java+C++) benchmark results."""
import json
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    text = Path(path).read_text()
    # strip info lines (stderr mixed in)
    lines = [l for l in text.strip().splitlines() if not l.startswith("[INFO]")]
    return json.loads("\n".join(lines))


def compare(baseline: dict, accelerated: dict) -> None:
    # extract method names from baseline
    methods = set()
    for key in baseline:
        if key.endswith("_java_ms"):
            methods.add(key.replace("_java_ms", ""))

    results = []
    for m in sorted(methods):
        java_key = f"{m}_java_ms"
        cpp_key = f"{m}_cpp_ms"
        java_ms = baseline.get(java_key) or accelerated.get(java_key)
        cpp_ms = accelerated.get(cpp_key)
        if java_ms is None:
            continue
        speedup = java_ms / cpp_ms if cpp_ms and cpp_ms > 0 else None
        results.append((m, java_ms, cpp_ms, speedup))

    # print report
    print(f"{'Method':<20} {'Java (ms)':>12} {'C++ (ms)':>12} {'Speedup':>10}")
    print("-" * 58)
    for m, java_ms, cpp_ms, speedup in results:
        java_str = f"{java_ms:.3f}"
        cpp_str = f"{cpp_ms:.3f}" if cpp_ms else "N/A"
        spd_str = f"{speedup:.2f}x" if speedup else "N/A"
        print(f"{m:<20} {java_str:>12} {cpp_str:>12} {spd_str:>10}")

    # summary
    print()
    accelerated_methods = [(m, s) for m, _, _, s in results if s is not None]
    if accelerated_methods:
        avg_speedup = sum(s for _, s in accelerated_methods) / len(accelerated_methods)
        wins = sum(1 for _, s in accelerated_methods if s > 1.0)
        losses = sum(1 for _, s in accelerated_methods if s < 1.0)
        print(f"Summary: {wins} wins, {losses} losses, avg speedup {avg_speedup:.2f}x")
        best = max(accelerated_methods, key=lambda x: x[1])
        worst = min(accelerated_methods, key=lambda x: x[1])
        print(f"  Best:  {best[0]} ({best[1]:.2f}x)")
        print(f"  Worst: {worst[0]} ({worst[1]:.2f}x)")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} baseline.json accelerated.json")
        sys.exit(1)
    baseline = load_results(sys.argv[1])
    accelerated = load_results(sys.argv[2])
    compare(baseline, accelerated)


if __name__ == "__main__":
    main()
