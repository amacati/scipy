import os

os.environ["SCIPY_ARRAY_API"] = "1"

from pathlib import Path
import json
from typing import Callable, Dict, List, Optional
import fire
import numpy as np
import timeit
import torch
import jax
import jax.numpy as jp
from functools import partial
from numpy.typing import NDArray

from scipy.spatial.transform import Rotation as R


ROTATION_FUNCTIONS = [
    "from_quat",
    "from_matrix",
    "from_rotvec",
    "from_mrp",
    "from_euler",
    "from_davenport",
    "as_quat",
    "as_matrix",
    "as_rotvec",
    "as_mrp",
    "as_euler",
    "as_davenport",
    "apply",
    "magnitude",
    "approx_equal",
    "mean",
    "reduce",
]
FRAMEWORKS = ["numpy", "torch", "jax"]
TIMEOUT = 60 * 5  # 5 minutes


def create_random_data(
    n_samples: int = 10000, xp_str: str = "numpy", device: str = "cpu"
):
    """Create random test data in numpy format."""
    if xp_str == "numpy":
        return np.random.randn(n_samples, 4), np.random.rand(n_samples, 3)
    elif xp_str == "torch":
        device = "cuda" if device == "gpu" else device
        return torch.randn(n_samples, 4, device=device), torch.rand(
            n_samples, 3, device=device
        )
    elif xp_str == "jax":
        return jax_qp(n_samples, device)
    raise ValueError(f"Invalid xp_str: {xp_str}")


@partial(jax.jit, static_argnums=[0, 1])
def jax_qp(n_samples: int = 10000, device: str = "cpu"):
    dev = jax.devices(device)[0]
    q = jp.array(jax.random.normal(jax.random.PRNGKey(0), (n_samples, 4)), device=dev)
    p = jp.array(jax.random.uniform(jax.random.PRNGKey(0), (n_samples, 3)), device=dev)
    return q, p


def to_xp_array(xp: str, array: NDArray, device: str = "cpu") -> NDArray:
    if xp == "numpy":
        return array
    elif xp == "torch":
        device = "cuda" if device == "gpu" else device
        return torch.from_numpy(array).to(device)
    elif xp == "jax":
        dev = jax.devices(device)[0]
        return jax.numpy.asarray(array, device=dev)


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


def benchmark_from_quat(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    print(f"Benchmarking from_quat with {xp} and {device}")
    q, p, r, from_quat = None, None, None, None

    def setup() -> str:
        nonlocal q, p, r, from_quat
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        if xp == "jax":
            from_quat = jax.jit(R.from_quat)
            jax.block_until_ready(from_quat(q))
        r = R.from_quat(q)

    def test():
        nonlocal q
        return R.from_quat(q)

    def jax_test():
        nonlocal q, from_quat
        jax.block_until_ready(from_quat(q))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_as_quat(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark as_quat with different array types."""
    print(f"Benchmarking as_quat with {xp} and {device}")
    q, p, r, as_quat = None, None, None, None

    def setup() -> str:
        nonlocal q, p, r, as_quat
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        if xp == "jax":
            as_quat = jax.jit(R.as_quat)
            jax.block_until_ready(as_quat(r))

    def test():
        nonlocal r
        return r.as_quat()

    def jax_test():
        nonlocal r, as_quat
        jax.block_until_ready(as_quat(r))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_as_matrix(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark as_matrix with different array types."""
    print(f"Benchmarking as_matrix with {xp} and {device}")
    q, p, r, as_matrix = None, None, None, None

    def setup() -> str:
        nonlocal q, p, r, as_matrix
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        if xp == "jax":
            as_matrix = jax.jit(R.as_matrix)
            jax.block_until_ready(as_matrix(r))

    def test():
        nonlocal r
        return r.as_matrix()

    def jax_test():
        nonlocal r, as_matrix
        jax.block_until_ready(as_matrix(r))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )

    return timing


def benchmark_apply(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark apply with different array types."""

    print(f"Benchmarking apply with {xp} and {device}")
    q, p, r, apply = None, None, None, None

    def setup() -> str:
        nonlocal q, p, r, apply
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        if xp == "jax":
            apply = jax.jit(R.apply)
            jax.block_until_ready(apply(r, p))

    def test():
        nonlocal r, p
        return r.apply(p)

    def jax_test():
        nonlocal r, p, apply
        jax.block_until_ready(apply(r, p))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )

    return timing


def benchmark_from_matrix(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark from_matrix with different array types."""
    print(f"Benchmarking from_matrix with {xp} and {device}")
    matrices, p, r, from_matrix = None, None, None, None

    def setup() -> str:
        nonlocal matrices, p, r, from_matrix
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        matrices = r.as_matrix()
        if xp == "jax":
            from_matrix = jax.jit(R.from_matrix)
            jax.block_until_ready(from_matrix(matrices))
        r = R.from_matrix(matrices)

    def test():
        nonlocal matrices
        return R.from_matrix(matrices)

    def jax_test():
        nonlocal matrices, from_matrix
        jax.block_until_ready(from_matrix(matrices))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_from_rotvec(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark from_rotvec with different array types."""
    print(f"Benchmarking from_rotvec with {xp} and {device}")
    rotvecs, p, r, from_rotvec = None, None, None, None

    def setup() -> str:
        nonlocal rotvecs, p, r, from_rotvec
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        rotvecs = r.as_rotvec()
        if xp == "jax":
            from_rotvec = jax.jit(R.from_rotvec)
            jax.block_until_ready(from_rotvec(rotvecs))
        r = R.from_rotvec(rotvecs)

    def test():
        nonlocal rotvecs
        return R.from_rotvec(rotvecs)

    def jax_test():
        nonlocal rotvecs, from_rotvec
        jax.block_until_ready(from_rotvec(rotvecs))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_from_mrp(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark from_mrp with different array types."""
    print(f"Benchmarking from_mrp with {xp} and {device}")
    mrps, p, r, from_mrp = None, None, None, None

    def setup() -> str:
        nonlocal mrps, p, r, from_mrp
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        mrps = r.as_mrp()
        if xp == "jax":
            from_mrp = jax.jit(R.from_mrp)
            jax.block_until_ready(from_mrp(mrps))
        r = R.from_mrp(mrps)

    def test():
        nonlocal mrps
        return R.from_mrp(mrps)

    def jax_test():
        nonlocal mrps, from_mrp
        jax.block_until_ready(from_mrp(mrps))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_from_euler(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark from_euler with different array types."""
    print(f"Benchmarking from_euler with {xp} and {device}")
    angles, p, r, from_euler = None, None, None, None

    def setup() -> str:
        nonlocal angles, p, r, from_euler
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        angles = r.as_euler("xyz")
        if xp == "jax":
            from_euler = jax.jit(lambda x: R.from_euler("xyz", x))
            jax.block_until_ready(from_euler(angles))
        r = R.from_euler("xyz", angles)

    def test():
        nonlocal angles
        return R.from_euler("xyz", angles)

    def jax_test():
        nonlocal angles, from_euler
        jax.block_until_ready(from_euler(angles))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_magnitude(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark magnitude with different array types."""
    print(f"Benchmarking magnitude with {xp} and {device}")
    q, p, r, magnitude = None, None, None, None

    def setup() -> str:
        nonlocal q, p, r, magnitude
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        if xp == "jax":
            magnitude = jax.jit(R.magnitude)
            jax.block_until_ready(magnitude(r))

    def test():
        nonlocal r
        return r.magnitude()

    def jax_test():
        nonlocal r, magnitude
        jax.block_until_ready(magnitude(r))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_approx_equal(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark approx_equal with different array types."""
    print(f"Benchmarking approx_equal with {xp} and {device}")
    q, p, r1, r2, approx_equal = None, None, None, None, None

    def setup() -> str:
        nonlocal q, p, r1, r2, approx_equal
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r1 = R.from_quat(q)
        r2 = R.from_quat(q)  # Same rotation for testing
        if xp == "jax":
            approx_equal = jax.jit(R.approx_equal)
            jax.block_until_ready(approx_equal(r1, r2))

    def test():
        nonlocal r1, r2
        return r1.approx_equal(r2)

    def jax_test():
        nonlocal r1, r2, approx_equal
        jax.block_until_ready(approx_equal(r1, r2))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_mean(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark mean with different array types."""
    print(f"Benchmarking mean with {xp} and {device}")
    q, p, r, mean = None, None, None, None

    def setup() -> str:
        nonlocal q, p, r, mean
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        if xp == "jax":
            mean = jax.jit(R.mean)
            jax.block_until_ready(mean(r))

    def test():
        nonlocal r
        return r.mean()

    def jax_test():
        nonlocal r, mean
        jax.block_until_ready(mean(r))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_reduce(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark reduce with different array types."""
    print(f"Benchmarking reduce with {xp} and {device}")
    r, left, right, reduce = None, None, None, None

    def setup() -> str:
        nonlocal r, left, right, reduce
        q, _ = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        q, _ = create_random_data(n_samples, xp, device)
        left = R.from_quat(q)
        q, _ = create_random_data(n_samples, xp, device)
        right = R.from_quat(q)
        if xp == "jax":
            reduce = jax.jit(R.reduce)
            jax.block_until_ready(reduce(r, left, right))

    def test():
        nonlocal r, left, right
        return R.reduce(r, left, right)

    def jax_test():
        nonlocal r, left, right, reduce
        jax.block_until_ready(reduce(r, left, right))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_from_davenport(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark from_davenport with different array types."""
    print(f"Benchmarking from_davenport with {xp} and {device}")
    p, from_davenport = None, None

    def setup() -> str:
        nonlocal p, from_davenport
        p = create_random_data(n_samples, xp, device)[1]
        dev = "gpu" if "cuda" in str(p.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        # Create two sets of vectors for Davenport's q-method
        if xp == "jax":
            from_davenport = jax.jit(partial(R.from_davenport, order="e"))
            jax.block_until_ready(from_davenport(p[0, :], angles=p[:, 0]))

    def test():
        nonlocal p
        return R.from_davenport(p[0, :], "e", p[:, 0])

    def jax_test():
        nonlocal p, from_davenport
        jax.block_until_ready(from_davenport(p[0, :], angles=p[:, 0]))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_as_rotvec(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark as_rotvec with different array types."""
    print(f"Benchmarking as_rotvec with {xp} and {device}")
    q, p, r, as_rotvec = None, None, None, None

    def setup() -> str:
        nonlocal q, p, r, as_rotvec
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        if xp == "jax":
            as_rotvec = jax.jit(R.as_rotvec)
            jax.block_until_ready(as_rotvec(r))

    def test():
        nonlocal r
        return r.as_rotvec()

    def jax_test():
        nonlocal r, as_rotvec
        jax.block_until_ready(as_rotvec(r))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_as_mrp(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark as_mrp with different array types."""
    print(f"Benchmarking as_mrp with {xp} and {device}")
    q, p, r, as_mrp = None, None, None, None

    def setup() -> str:
        nonlocal q, p, r, as_mrp
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        if xp == "jax":
            as_mrp = jax.jit(R.as_mrp)
            jax.block_until_ready(as_mrp(r))

    def test():
        nonlocal r
        return r.as_mrp()

    def jax_test():
        nonlocal r, as_mrp
        jax.block_until_ready(as_mrp(r))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_as_euler(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark as_euler with different array types."""
    print(f"Benchmarking as_euler with {xp} and {device}")
    q, p, r, as_euler = None, None, None, None

    def setup() -> str:
        nonlocal q, p, r, as_euler
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        if xp == "jax":
            as_euler = jax.jit(R.as_euler, static_argnames=["seq"])
            jax.block_until_ready(as_euler(r, seq="xyz"))

    def test():
        nonlocal r
        return r.as_euler("xyz")

    def jax_test():
        nonlocal r, as_euler
        jax.block_until_ready(as_euler(r, seq="xyz"))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_as_davenport(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> Dict[str, float]:
    """Benchmark as_davenport with different array types."""
    print(f"Benchmarking as_davenport with {xp} and {device}")
    axes, r, as_davenport = None, None, None

    def setup() -> str:
        nonlocal axes, r, as_davenport
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device) else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        axes = to_xp_array(xp, np.eye(3), device)
        if xp == "jax":
            as_davenport = jax.jit(R.as_davenport, static_argnames=["order"])
            jax.block_until_ready(as_davenport(r, axes, order="e"))

    def test():
        nonlocal axes, r
        return r.as_davenport(axes, "e")

    def jax_test():
        nonlocal r, as_davenport
        jax.block_until_ready(as_davenport(r, axes, order="e"))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def save_results(
    xp: str,
    device: str,
    func: str,
    results: Dict[str, List[float]],
    n_samples: int,
):
    """Save benchmark results to JSON file."""
    save_dir = Path(__file__).parent / "results" / xp / device
    save_dir.mkdir(parents=True, exist_ok=True)

    result_file = save_dir / f"{func}.json"
    existing_results = {}
    if result_file.exists():
        with open(result_file, "r") as f:
            existing_results = json.load(f)

    n_samples = int(n_samples)
    assert isinstance(n_samples, int)
    existing_results[n_samples] = results
    with open(result_file, "w") as f:
        json.dump(existing_results, f, indent=2)


def _benchmark(
    fn: str,
    xp: str,
    device: str,
    n_samples: int = 10000,
    repeat: int = 5,
    number: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Run benchmarks for specified frameworks and devices."""
    match fn:
        case "from_quat":
            results = benchmark_from_quat(xp, device, n_samples, repeat, number)
        case "from_matrix":
            results = benchmark_from_matrix(xp, device, n_samples, repeat, number)
        case "from_rotvec":
            results = benchmark_from_rotvec(xp, device, n_samples, repeat, number)
        case "from_mrp":
            results = benchmark_from_mrp(xp, device, n_samples, repeat, number)
        case "from_euler":
            results = benchmark_from_euler(xp, device, n_samples, repeat, number)
        case "from_davenport":
            results = benchmark_from_davenport(xp, device, n_samples, repeat, number)
        case "as_quat":
            results = benchmark_as_quat(xp, device, n_samples, repeat, number)
        case "as_matrix":
            results = benchmark_as_matrix(xp, device, n_samples, repeat, number)
        case "as_rotvec":
            results = benchmark_as_rotvec(xp, device, n_samples, repeat, number)
        case "as_mrp":
            results = benchmark_as_mrp(xp, device, n_samples, repeat, number)
        case "as_euler":
            results = benchmark_as_euler(xp, device, n_samples, repeat, number)
        case "as_davenport":
            results = benchmark_as_davenport(xp, device, n_samples, repeat, number)
        case "apply":
            results = benchmark_apply(xp, device, n_samples, repeat, number)
        case "magnitude":
            results = benchmark_magnitude(xp, device, n_samples, repeat, number)
        case "approx_equal":
            results = benchmark_approx_equal(xp, device, n_samples, repeat, number)
        case "mean":
            results = benchmark_mean(xp, device, n_samples, repeat, number)
        case "reduce":
            results = benchmark_reduce(xp, device, n_samples, repeat, number)
        case _:
            raise ValueError(f"Invalid function: {fn}")
    # Save results for each framework/device combination
    if len(results) > 0:
        save_results(xp, device, fn, results.tolist(), n_samples)
    return results


def run_benchmarks(
    fn: Optional[List[str]] = None,
    xp: str | None = None,
    device: str = "cpu",
    exp_low: int = 0,
    exp_high: int = 7,
    repeat: int = 5,
    number: int = 100,
):
    """Run benchmarks with specified configurations."""
    sample_sizes = np.logspace(exp_low, exp_high, exp_high - exp_low + 1).astype(int)
    sample_sizes = np.sort(np.array(list(set(sample_sizes))))

    fns = [fn] if fn is not None else ROTATION_FUNCTIONS
    frameworks = [xp] if xp is not None else FRAMEWORKS

    for xp in frameworks:
        for fn in fns:
            for n_samples in sample_sizes:
                print(f"Running {fn} benchmark for {n_samples} samples")
                results = _benchmark(fn, xp, device, n_samples, repeat, number)
                if len(results) == 0:
                    print(
                        f"Skipping remaining sample sizes for {fn} with {xp} on {device}"
                    )
                    break  # Skip to next function if timeout occurred


if __name__ == "__main__":
    fire.Fire(run_benchmarks)
