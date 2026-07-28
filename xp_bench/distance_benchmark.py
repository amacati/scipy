import os

os.environ["SCIPY_ARRAY_API"] = "1"

from pathlib import Path
import json
from typing import Callable, List, Optional
import fire
import numpy as np
import timeit
import torch
import jax
import jax.numpy as jp
import cupy
from functools import partial
from numpy.typing import NDArray

from scipy.spatial import distance


# Pair metrics operate on two 1-D vectors and return a scalar. We benchmark them by
# scaling the vector length, which is the elementwise work an array-API backend
# parallelizes. The call shape is identical on main and on any converted branch, so the
# timings are directly comparable. Each entry is (callable, dtype, jittable).
PAIR_SPECS = {
    "euclidean": (distance.euclidean, "float64", True),
    "sqeuclidean": (distance.sqeuclidean, "float64", True),
    "cityblock": (distance.cityblock, "float64", True),
    "chebyshev": (distance.chebyshev, "float64", True),
    "minkowski": (partial(distance.minkowski, p=3), "float64", True),
    "cosine": (distance.cosine, "float64", True),
    "correlation": (distance.correlation, "float64", True),
    "braycurtis": (distance.braycurtis, "float64", True),
    "canberra": (distance.canberra, "float64", True),
    "hamming": (distance.hamming, "float64", True),
    "jensenshannon": (distance.jensenshannon, "float64", True),
    "jaccard": (distance.jaccard, "bool", True),
    "yule": (distance.yule, "bool", True),
    "dice": (distance.dice, "bool", True),
    "rogerstanimoto": (distance.rogerstanimoto, "bool", True),
    "russellrao": (distance.russellrao, "bool", True),
    "sokalsneath": (distance.sokalsneath, "bool", False),  # data-dependent guard, no jit
}

# Everything else needs bespoke data (auxiliary matrices, observation matrices, distance
# matrices, condensed vectors).
OTHER_FUNCTIONS = [
    "mahalanobis",
    "seuclidean",
    "pdist",
    "cdist",
    "squareform",
    "is_valid_dm",
    "is_valid_y",
    "num_obs_dm",
    "num_obs_y",
]

DISTANCE_FUNCTIONS = list(PAIR_SPECS) + OTHER_FUNCTIONS

FRAMEWORKS = ["numpy", "torch", "jax", "cupy"]
TIMEOUT = 60 * 5  # 5 minutes
N_FEATURES = 10  # feature dimension for observation matrices (pdist/cdist)


def to_xp_array(xp: str, array: NDArray, device: str = "cpu") -> NDArray:
    if xp == "numpy":
        return array
    elif xp == "torch":
        device = "cuda" if device == "gpu" else device
        return torch.from_numpy(array).to(device)
    elif xp == "jax":
        dev = jax.devices(device)[0]
        return jax.numpy.asarray(array, device=dev)
    elif xp == "cupy":
        assert device == "gpu", "cupy only supports gpu"
        return cupy.asarray(array)
    raise ValueError(f"Invalid xp_str: {xp}")


def device_of(array: NDArray) -> str:
    dev = str(getattr(array, "device", "cpu")).lower()
    return "gpu" if "cuda" in dev else "cpu"


def create_vector(xp: str, device: str, n: int, dtype: str = "float64") -> NDArray:
    if dtype == "bool":
        arr = np.random.rand(n) > 0.5
        arr[0] = True  # never entirely false (sokalsneath is undefined there)
        if n > 1:
            arr[1] = False
    else:
        arr = np.random.rand(n).astype(dtype)
    return to_xp_array(xp, arr, device)


def create_matrix(xp: str, device: str, m: int, f: int = N_FEATURES) -> NDArray:
    return to_xp_array(xp, np.random.rand(m, f), device)


def create_distance_matrix(xp: str, device: str, k: int) -> NDArray:
    """Symmetric matrix with a zero diagonal, a valid input to is_valid_dm/num_obs_dm."""
    a = np.random.rand(k, k)
    d = a + a.T
    np.fill_diagonal(d, 0.0)
    return to_xp_array(xp, d, device)


def create_condensed(xp: str, device: str, n: int) -> NDArray:
    """Condensed vector whose length is a valid binomial coefficient near n."""
    m = int((1 + (1 + 8 * n) ** 0.5) // 2)
    length = max(m * (m - 1) // 2, 1)
    return to_xp_array(xp, np.random.rand(length), device)


def benchmark_function(
    setup_code: Callable, test_code: Callable, R: int, N: int
) -> NDArray:
    """Run benchmark with timeout."""

    # Run setup once to ensure everything is initialized
    setup_code()

    # First test run to check if it exceeds timeout
    start_time = timeit.default_timer()
    test_code()
    elapsed = timeit.default_timer() - start_time

    # If a single run takes more than timeout/R seconds, abort
    if elapsed > TIMEOUT / (R * N):
        print(
            f"Aborting: Single run took {elapsed:.2f}s, which would exceed timeout of {TIMEOUT}s for {R} runs"
        )
        return np.array([])

    # Proceed with full benchmark if within time limit
    timer = timeit.Timer(stmt=test_code, setup=setup_code)
    return np.array(timer.repeat(repeat=R, number=N)) / N


def benchmark_pair(
    call: Callable,
    xp: str,
    device: str,
    n: int,
    repeat: int,
    number: int,
    dtype: str = "float64",
    jittable: bool = True,
) -> NDArray:
    """Benchmark a two-vector scalar metric, scaling the vector length."""
    u, v, jfn = None, None, None

    def setup():
        nonlocal u, v, jfn
        u = create_vector(xp, device, n, dtype)
        v = create_vector(xp, device, n, dtype)
        assert device_of(u) == device, f"setup device mismatch: {device_of(u)} != {device}"
        if xp == "jax" and jittable:
            jfn = jax.jit(call)
            jax.block_until_ready(jfn(u, v))

    def test():
        nonlocal u, v
        return call(u, v)

    def jax_test():
        nonlocal u, v, jfn
        jax.block_until_ready((jfn if jittable else call)(u, v))

    return benchmark_function(
        setup, jax_test if xp == "jax" else test, repeat, number
    )


def benchmark_mahalanobis(
    xp: str, device: str, n: int, repeat: int, number: int
) -> NDArray:
    print(f"Benchmarking mahalanobis with {xp} and {device}")
    u, v, VI = None, None, None

    def setup():
        nonlocal u, v, VI
        u = create_vector(xp, device, n)
        v = create_vector(xp, device, n)
        VI = to_xp_array(xp, np.random.rand(n, n), device)
        assert device_of(u) == device, f"setup device mismatch: {device_of(u)} != {device}"

    def test():
        nonlocal u, v, VI
        return distance.mahalanobis(u, v, VI)

    def jax_test():
        nonlocal u, v, VI
        jax.block_until_ready(distance.mahalanobis(u, v, VI))

    return benchmark_function(
        setup, jax_test if xp == "jax" else test, repeat, number
    )


def benchmark_seuclidean(
    xp: str, device: str, n: int, repeat: int, number: int
) -> NDArray:
    print(f"Benchmarking seuclidean with {xp} and {device}")
    u, v, V = None, None, None

    def setup():
        nonlocal u, v, V
        u = create_vector(xp, device, n)
        v = create_vector(xp, device, n)
        V = create_vector(xp, device, n)  # positive variances (uniform in [0, 1))
        assert device_of(u) == device, f"setup device mismatch: {device_of(u)} != {device}"

    def test():
        nonlocal u, v, V
        return distance.seuclidean(u, v, V)

    def jax_test():
        nonlocal u, v, V
        jax.block_until_ready(distance.seuclidean(u, v, V))

    return benchmark_function(
        setup, jax_test if xp == "jax" else test, repeat, number
    )


def benchmark_pdist(xp: str, device: str, n: int, repeat: int, number: int) -> NDArray:
    print(f"Benchmarking pdist with {xp} and {device}")
    X = None

    def setup():
        nonlocal X
        X = create_matrix(xp, device, n)
        assert device_of(X) == device, f"setup device mismatch: {device_of(X)} != {device}"

    def test():
        nonlocal X
        return distance.pdist(X)

    def jax_test():
        nonlocal X
        jax.block_until_ready(distance.pdist(X))

    return benchmark_function(
        setup, jax_test if xp == "jax" else test, repeat, number
    )


def benchmark_cdist(xp: str, device: str, n: int, repeat: int, number: int) -> NDArray:
    print(f"Benchmarking cdist with {xp} and {device}")
    XA, XB = None, None

    def setup():
        nonlocal XA, XB
        XA = create_matrix(xp, device, n)
        XB = create_matrix(xp, device, n)
        assert device_of(XA) == device, f"setup device mismatch: {device_of(XA)} != {device}"

    def test():
        nonlocal XA, XB
        return distance.cdist(XA, XB)

    def jax_test():
        nonlocal XA, XB
        jax.block_until_ready(distance.cdist(XA, XB))

    return benchmark_function(
        setup, jax_test if xp == "jax" else test, repeat, number
    )


def benchmark_squareform(
    xp: str, device: str, n: int, repeat: int, number: int
) -> NDArray:
    print(f"Benchmarking squareform with {xp} and {device}")
    y = None

    def setup():
        nonlocal y
        y = create_condensed(xp, device, n)
        assert device_of(y) == device, f"setup device mismatch: {device_of(y)} != {device}"

    def test():
        nonlocal y
        return distance.squareform(y)

    def jax_test():
        nonlocal y
        jax.block_until_ready(distance.squareform(y))

    return benchmark_function(
        setup, jax_test if xp == "jax" else test, repeat, number
    )


def _benchmark_validator(
    call: Callable, kind: str, xp: str, device: str, n: int, repeat: int, number: int
) -> NDArray:
    data = None

    def setup():
        nonlocal data
        data = (
            create_distance_matrix(xp, device, n)
            if kind == "dm"
            else create_condensed(xp, device, n)
        )
        assert device_of(data) == device, f"setup device mismatch: {device_of(data)} != {device}"

    def test():
        nonlocal data
        return call(data)

    def jax_test():
        nonlocal data
        jax.block_until_ready(call(data))

    return benchmark_function(
        setup, jax_test if xp == "jax" else test, repeat, number
    )


def save_results(xp: str, device: str, func: str, results: List[float], n_samples: int):
    """Save benchmark results to JSON file."""
    save_dir = Path(__file__).parent / "distance_results" / xp / device
    save_dir.mkdir(parents=True, exist_ok=True)

    result_file = save_dir / f"{func}.json"
    existing_results = {}
    if result_file.exists():
        with open(result_file, "r") as f:
            existing_results = json.load(f)

    n_samples = int(n_samples)
    existing_results[str(n_samples)] = results
    with open(result_file, "w") as f:
        json.dump(existing_results, f, indent=2)


def _benchmark(
    fn: str,
    xp: str,
    device: str,
    n_samples: int = 10000,
    repeat: int = 5,
    number: int = 100,
) -> NDArray:
    if fn in PAIR_SPECS:
        print(f"Benchmarking {fn} with {xp} and {device}")
        call, dtype, jittable = PAIR_SPECS[fn]
        results = benchmark_pair(
            call, xp, device, n_samples, repeat, number, dtype, jittable
        )
    elif fn == "mahalanobis":
        results = benchmark_mahalanobis(xp, device, n_samples, repeat, number)
    elif fn == "seuclidean":
        results = benchmark_seuclidean(xp, device, n_samples, repeat, number)
    elif fn == "pdist":
        results = benchmark_pdist(xp, device, n_samples, repeat, number)
    elif fn == "cdist":
        results = benchmark_cdist(xp, device, n_samples, repeat, number)
    elif fn == "squareform":
        results = benchmark_squareform(xp, device, n_samples, repeat, number)
    elif fn == "is_valid_dm":
        print(f"Benchmarking is_valid_dm with {xp} and {device}")
        results = _benchmark_validator(
            distance.is_valid_dm, "dm", xp, device, n_samples, repeat, number
        )
    elif fn == "num_obs_dm":
        print(f"Benchmarking num_obs_dm with {xp} and {device}")
        results = _benchmark_validator(
            distance.num_obs_dm, "dm", xp, device, n_samples, repeat, number
        )
    elif fn == "is_valid_y":
        print(f"Benchmarking is_valid_y with {xp} and {device}")
        results = _benchmark_validator(
            distance.is_valid_y, "y", xp, device, n_samples, repeat, number
        )
    elif fn == "num_obs_y":
        print(f"Benchmarking num_obs_y with {xp} and {device}")
        results = _benchmark_validator(
            distance.num_obs_y, "y", xp, device, n_samples, repeat, number
        )
    else:
        raise ValueError(f"Invalid function: {fn}")

    if len(results) > 0:
        save_results(xp, device, fn, results.tolist(), n_samples)
    return results


def run_benchmarks(
    fn: Optional[List[str]] = None,
    xp: str | None = None,
    device: str | None = None,
    low: int = 0,
    high: int = 7,
    repeat: int = 5,
    number: int = 100,
):
    """Run benchmarks with specified configurations."""
    sample_sizes = np.logspace(low, high, high - low + 1).astype(int)
    sample_sizes = np.sort(np.array(list(set(sample_sizes))))

    fns = [fn] if fn is not None else DISTANCE_FUNCTIONS
    frameworks = [xp] if xp is not None else FRAMEWORKS
    devices = [device] if device is not None else ["cpu", "gpu"]
    SKIP_XP_DEVICES = [("numpy", "gpu"), ("cupy", "cpu")]

    for xp in frameworks:
        for fn in fns:
            for device in devices:
                if (xp, device) in SKIP_XP_DEVICES:
                    print(f"Skipping {xp} on {device}")
                    continue
                for n_samples in sample_sizes:
                    print(f"Running {fn} benchmark for {n_samples} samples")
                    try:
                        results = _benchmark(fn, xp, device, n_samples, repeat, number)
                    except (MemoryError, np.core._exceptions._ArrayMemoryError):
                        print(f"Skipping {fn} with {xp} on {device} - Out of Memory")
                        break
                    except ValueError as e:
                        print(f"Skipping {fn} with {xp} on {device} - {e}")
                        break
                    except torch.OutOfMemoryError:
                        print(f"Skipping {fn} with {xp} on {device} - Out of Memory")
                        break
                    except cupy.cuda.memory.OutOfMemoryError:
                        print(f"Skipping {fn} with {xp} on {device} - Out of Memory")
                        break
                    if len(results) == 0:
                        print(
                            f"Skipping remaining sample sizes for {fn} with {xp} on {device}"
                        )
                        break  # Skip to next function if timeout occurred


if __name__ == "__main__":
    fire.Fire(run_benchmarks)
