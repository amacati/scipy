import json
import numpy as np
import os

os.environ["SCIPY_ARRAY_API"] = "1"
import jax
from pathlib import Path
from scipy.spatial.transform import Rotation as R, RigidTransform
from rotation_benchmark import (
    benchmark_function,
    create_random_data,
    FRAMEWORKS,
)
from array_api_compat import array_namespace
import fire

TRANSFORM_FUNCTIONS = [
    "from_matrix",
    "from_rotation",
    "from_translation",
    "from_components",
    "from_exp_coords",
    "from_dual_quat",
    "as_matrix",
    "as_components",
    "as_exp_coords",
    "as_dual_quat",
    "apply",
    "concatenate",
    "inv",
    "mul",
    "pow",
]


def benchmark_from_rotation(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark from_rotation with different array types."""
    print(f"Benchmarking from_rotation with {xp} and {device}")
    r, from_rotation = None, None

    def setup() -> str:
        nonlocal r, from_rotation
        q, _ = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        if xp == "jax":
            from_rotation = jax.jit(RigidTransform.from_rotation)
            jax.block_until_ready(from_rotation(r))
        RigidTransform.from_rotation(r)

    def test():
        nonlocal r
        return RigidTransform.from_rotation(r)

    def jax_test():
        nonlocal r, from_rotation
        jax.block_until_ready(from_rotation(r))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_from_components(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark from_components with different array types."""
    print(f"Benchmarking from_components with {xp} and {device}")
    p, r, from_components = None, None, None

    def setup() -> str:
        nonlocal p, r, from_components
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        if xp == "jax":
            from_components = jax.jit(RigidTransform.from_components)
            jax.block_until_ready(from_components(p, r))
        RigidTransform.from_components(p, r)

    def test():
        nonlocal p, r
        return RigidTransform.from_components(p, r)

    def jax_test():
        nonlocal p, r, from_components
        jax.block_until_ready(from_components(p, r))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_as_matrix(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark as_matrix with different array types."""
    print(f"Benchmarking as_matrix with {xp} and {device}")
    tf, as_matrix = None, None

    def setup() -> str:
        nonlocal tf, as_matrix
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        tf = RigidTransform.from_components(p, r)
        if xp == "jax":
            as_matrix = jax.jit(RigidTransform.as_matrix)
            jax.block_until_ready(as_matrix(tf))

    def test():
        nonlocal tf
        return tf.as_matrix()

    def jax_test():
        nonlocal tf, as_matrix
        jax.block_until_ready(as_matrix(tf))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_as_components(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark as_components with different array types."""
    print(f"Benchmarking as_components with {xp} and {device}")
    tf, as_components = None, None

    def setup() -> str:
        nonlocal tf, as_components
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        tf = RigidTransform.from_components(p, r)
        if xp == "jax":
            as_components = jax.jit(RigidTransform.as_components)
            jax.block_until_ready(as_components(tf))

    def test():
        nonlocal tf
        return tf.as_components()

    def jax_test():
        nonlocal tf, as_components
        jax.block_until_ready(as_components(tf))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_mul(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark multiplication of transforms with different array types."""
    print(f"Benchmarking mul with {xp} and {device}")
    tf1, tf2, mul = None, None, None

    def setup() -> str:
        nonlocal tf1, tf2, mul
        q1, p1 = create_random_data(n_samples, xp, device)
        q2, p2 = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q1.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        tf1 = RigidTransform.from_components(p1, r1)
        tf2 = RigidTransform.from_components(p2, r2)
        if xp == "jax":
            mul = jax.jit(RigidTransform.__mul__)
            jax.block_until_ready(mul(tf1, tf2))

    def test():
        nonlocal tf1, tf2
        return tf1 * tf2

    def jax_test():
        nonlocal tf1, tf2, mul
        jax.block_until_ready(mul(tf1, tf2))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_pow(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark power operation with different array types."""
    print(f"Benchmarking pow with {xp} and {device}")
    tf, pow_fn = None, None

    def setup() -> str:
        nonlocal tf, pow_fn
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        tf = RigidTransform.from_components(p, r)
        if xp == "jax":
            pow_fn = jax.jit(RigidTransform.__pow__)
            jax.block_until_ready(pow_fn(tf, 2.0))

    def test():
        nonlocal tf
        return tf**2.0

    def jax_test():
        nonlocal tf, pow_fn
        jax.block_until_ready(pow_fn(tf, 2.0))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_from_matrix(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark from_matrix with different array types."""
    print(f"Benchmarking from_matrix with {xp} and {device}")
    matrices, from_matrix = None, None

    def setup() -> str:
        nonlocal matrices, from_matrix
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        tf = RigidTransform.from_components(p, r)
        matrices = tf.as_matrix()
        if xp == "jax":
            from_matrix = jax.jit(RigidTransform.from_matrix)
            jax.block_until_ready(from_matrix(matrices))

    def test():
        nonlocal matrices
        return RigidTransform.from_matrix(matrices)

    def jax_test():
        nonlocal matrices, from_matrix
        jax.block_until_ready(from_matrix(matrices))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_from_translation(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark from_translation with different array types."""
    print(f"Benchmarking from_translation with {xp} and {device}")
    p, from_translation = None, None

    def setup() -> str:
        nonlocal p, from_translation
        _, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(p.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        if xp == "jax":
            from_translation = jax.jit(RigidTransform.from_translation)
            jax.block_until_ready(from_translation(p))

    def test():
        nonlocal p
        return RigidTransform.from_translation(p)

    def jax_test():
        nonlocal p, from_translation
        jax.block_until_ready(from_translation(p))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_apply(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark apply with different array types."""
    print(f"Benchmarking apply with {xp} and {device}")
    tf, vectors, apply = None, None, None

    def setup() -> str:
        nonlocal tf, vectors, apply
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        tf = RigidTransform.from_components(p, r)
        vectors = p  # Use translation vectors as test vectors
        if xp == "jax":
            apply = jax.jit(RigidTransform.apply)
            jax.block_until_ready(apply(tf, vectors))

    def test():
        nonlocal tf, vectors
        return tf.apply(vectors)

    def jax_test():
        nonlocal tf, vectors, apply
        jax.block_until_ready(apply(tf, vectors))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_inv(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark inv with different array types."""
    print(f"Benchmarking inv with {xp} and {device}")
    tf, inv = None, None

    def setup() -> str:
        nonlocal tf, inv
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        tf = RigidTransform.from_components(p, r)
        if xp == "jax":
            inv = jax.jit(RigidTransform.inv)
            jax.block_until_ready(inv(tf))

    def test():
        nonlocal tf
        return tf.inv()

    def jax_test():
        nonlocal tf, inv
        jax.block_until_ready(inv(tf))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_from_exp_coords(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark from_exp_coords with different array types."""
    print(f"Benchmarking from_exp_coords with {xp} and {device}")
    exp_coords, from_exp_coords = None, None

    def setup() -> str:
        nonlocal exp_coords, from_exp_coords
        _, p1 = create_random_data(n_samples, xp, device)
        _, p2 = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(p1.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        # Create 6D exponential coordinates (rotation vector + translation)
        _xp = array_namespace(p1)
        exp_coords = _xp.concat([p1, p2], axis=-1)
        if xp == "jax":
            from_exp_coords = jax.jit(RigidTransform.from_exp_coords)
            jax.block_until_ready(from_exp_coords(exp_coords))

    def test():
        nonlocal exp_coords
        return RigidTransform.from_exp_coords(exp_coords)

    def jax_test():
        nonlocal exp_coords, from_exp_coords
        jax.block_until_ready(from_exp_coords(exp_coords))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_from_dual_quat(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark from_dual_quat with different array types."""
    print(f"Benchmarking from_dual_quat with {xp} and {device}")
    dual_quat, from_dual_quat = None, None

    def setup() -> str:
        nonlocal dual_quat, from_dual_quat
        q1, _ = create_random_data(n_samples, xp, device)
        q2, _ = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q1.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        # Create dual quaternion from rotation quaternion and translation
        _xp = array_namespace(q1)
        dual_quat = _xp.concat([q1, q2], axis=-1)
        if xp == "jax":
            from_dual_quat = jax.jit(RigidTransform.from_dual_quat)
            jax.block_until_ready(from_dual_quat(dual_quat))

    def test():
        nonlocal dual_quat
        return RigidTransform.from_dual_quat(dual_quat)

    def jax_test():
        nonlocal dual_quat, from_dual_quat
        jax.block_until_ready(from_dual_quat(dual_quat))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_as_exp_coords(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark as_exp_coords with different array types."""
    print(f"Benchmarking as_exp_coords with {xp} and {device}")
    tf, as_exp_coords = None, None

    def setup() -> str:
        nonlocal tf, as_exp_coords
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        tf = RigidTransform.from_components(p, r)
        if xp == "jax":
            as_exp_coords = jax.jit(RigidTransform.as_exp_coords)
            jax.block_until_ready(as_exp_coords(tf))

    def test():
        nonlocal tf
        return tf.as_exp_coords()

    def jax_test():
        nonlocal tf, as_exp_coords
        jax.block_until_ready(as_exp_coords(tf))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_as_dual_quat(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark as_dual_quat with different array types."""
    print(f"Benchmarking as_dual_quat with {xp} and {device}")
    tf, as_dual_quat = None, None

    def setup() -> str:
        nonlocal tf, as_dual_quat
        q, p = create_random_data(n_samples, xp, device)
        dev = "gpu" if "cuda" in str(q.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r = R.from_quat(q)
        tf = RigidTransform.from_components(p, r)
        if xp == "jax":
            as_dual_quat = jax.jit(RigidTransform.as_dual_quat)
            jax.block_until_ready(as_dual_quat(tf))

    def test():
        nonlocal tf
        return tf.as_dual_quat()

    def jax_test():
        nonlocal tf, as_dual_quat
        jax.block_until_ready(as_dual_quat(tf))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def benchmark_concatenate(
    xp: str, device: str, n_samples: int, repeat: int, number: int
) -> dict[str, float]:
    """Benchmark concatenate with different array types."""
    print(f"Benchmarking concatenate with {xp} and {device}")
    tf1, tf2, concatenate = None, None, None

    def setup() -> str:
        nonlocal tf1, tf2, concatenate
        q1, p1 = create_random_data(n_samples // 2, xp, device)
        q2, p2 = create_random_data(n_samples // 2, xp, device)
        dev = "gpu" if "cuda" in str(q1.device).lower() else "cpu"
        assert dev == device, f"setup device mismatch: {dev} != {device}"
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        tf1 = RigidTransform.from_components(p1, r1)
        tf2 = RigidTransform.from_components(p2, r2)
        if xp == "jax":
            concatenate = jax.jit(RigidTransform.concatenate)
            jax.block_until_ready(concatenate([tf1, tf2]))

    def test():
        nonlocal tf1, tf2
        return RigidTransform.concatenate([tf1, tf2])

    def jax_test():
        nonlocal tf1, tf2, concatenate
        jax.block_until_ready(concatenate([tf1, tf2]))

    timing = benchmark_function(
        setup, test if xp != "jax" else jax_test, repeat, number
    )
    return timing


def _benchmark_transform(
    fn: str,
    xp: str,
    device: str,
    n_samples: int = 10000,
    repeat: int = 5,
    number: int = 100,
) -> dict[str, dict[str, float]]:
    """Run benchmarks for specified frameworks and devices."""
    match fn:
        case "from_rotation":
            results = benchmark_from_rotation(xp, device, n_samples, repeat, number)
        case "from_components":
            results = benchmark_from_components(xp, device, n_samples, repeat, number)
        case "as_matrix":
            results = benchmark_as_matrix(xp, device, n_samples, repeat, number)
        case "as_components":
            results = benchmark_as_components(xp, device, n_samples, repeat, number)
        case "mul":
            results = benchmark_mul(xp, device, n_samples, repeat, number)
        case "pow":
            results = benchmark_pow(xp, device, n_samples, repeat, number)
        case "from_matrix":
            results = benchmark_from_matrix(xp, device, n_samples, repeat, number)
        case "from_translation":
            results = benchmark_from_translation(xp, device, n_samples, repeat, number)
        case "apply":
            results = benchmark_apply(xp, device, n_samples, repeat, number)
        case "inv":
            results = benchmark_inv(xp, device, n_samples, repeat, number)
        case "from_exp_coords":
            results = benchmark_from_exp_coords(xp, device, n_samples, repeat, number)
        case "from_dual_quat":
            results = benchmark_from_dual_quat(xp, device, n_samples, repeat, number)
        case "as_exp_coords":
            results = benchmark_as_exp_coords(xp, device, n_samples, repeat, number)
        case "as_dual_quat":
            results = benchmark_as_dual_quat(xp, device, n_samples, repeat, number)
        case "concatenate":
            results = benchmark_concatenate(xp, device, n_samples, repeat, number)
        case _:
            raise ValueError(f"Invalid function: {fn}")

    # Save results for each framework/device combination
    if len(results) > 0:
        save_dir = Path(__file__).parent / "tf_results" / xp / device
        save_dir.mkdir(parents=True, exist_ok=True)
        result_file = save_dir / f"{fn}.json"

        existing_results = {}
        if result_file.exists():
            with open(result_file, "r") as f:
                existing_results = json.load(f)

        n_samples = int(n_samples)
        assert isinstance(n_samples, int)
        existing_results[str(n_samples)] = results.tolist()

        with open(result_file, "w") as f:
            json.dump(existing_results, f, indent=2)

    return results


def run_transform_benchmarks(
    fn: list[str] | None = None,
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

    fns = [fn] if fn is not None else TRANSFORM_FUNCTIONS
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
                    results = []
                    try:
                        results = _benchmark_transform(
                            fn, xp, device, n_samples, repeat, number
                        )
                    except AttributeError as e:
                        if str(e).startswith("'ndarray' object has no attribute 'mT'"):
                            print(
                                f"Skipping {fn} with {xp} on {device} - mT not supported"
                            )
                            break
                        raise e
                    except (RuntimeError, MemoryError) as e:
                        if "out of memory" in str(e).lower():
                            print(
                                f"Out of memory for {fn} with {xp} on {device} at {n_samples} samples"
                            )
                            break
                        raise e
                    if len(results) == 0:
                        print(
                            f"Skipping remaining sample sizes for {fn} with {xp} on {device}"
                        )
                        break


if __name__ == "__main__":
    fire.Fire(run_transform_benchmarks)
