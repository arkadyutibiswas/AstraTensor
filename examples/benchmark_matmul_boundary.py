"""Microbenchmark Python/native boundary costs for OddlyTyped matmul.

Run after building the extension:
    python examples/benchmark_matmul_boundary.py --batch-size 128 --repeats 30

This is a diagnostic, not a substitute for end-to-end training benchmarks.
"""
import argparse
import statistics
import time

import numpy as np

from oddlytyped.backend import matmul


def measure(function, repeats, warmup=5):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.repeats <= 0:
        parser.error("batch-size and repeats must be positive")

    rng = np.random.default_rng(args.seed)
    batch = args.batch_size
    cases = [
        ("forward", (batch, 784), (784, 128)),
        ("input-gradient", (batch, 128), (128, 784)),
        ("weight-gradient", (128, batch), (batch, 10)),
    ]

    print("OddlyTyped native matmul boundary microbenchmark")
    print(f"Batch size: {batch} | Repeats: {args.repeats} | Median latency")
    print("All paths call the same compiled C kernel; input/output conversion differs.")
    print()

    for name, left_shape, right_shape in cases:
        left = rng.normal(size=left_shape)
        right = rng.normal(size=right_shape)
        left_list, right_list = left.tolist(), right.tolist()

        expected = matmul(left, right)
        list_result = matmul(left_list, right_list)
        if not np.allclose(expected, list_result, rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"Numerical mismatch in {name} case")

        array_ms = measure(lambda: matmul(left, right), args.repeats)
        list_ms = measure(lambda: matmul(left_list, right_list), args.repeats)
        engine_path_ms = measure(
            lambda: matmul(left_list, right_list).tolist(), args.repeats
        )
        print(
            f"{name:16s} {left_shape} x {right_shape}\n"
            f"  NumPy input/output array : {array_ms:9.3f} ms\n"
            f"  Python-list input        : {list_ms:9.3f} ms\n"
            f"  Engine-style list path   : {engine_path_ms:9.3f} ms\n"
            f"  List-path overhead       : "
            f"{(engine_path_ms / array_ms - 1.0) * 100.0:8.1f}% vs array path"
        )


if __name__ == "__main__":
    main()
