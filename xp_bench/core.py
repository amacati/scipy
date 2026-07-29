"""Timing engine, backend helpers and result storage shared by every suite.

Read xp_bench/README.md before changing anything in here. The timing contract is
subtle and the numbers on disk are only comparable while it holds.
"""

import json
import timeit
from pathlib import Path

import cupy
import jax
import numpy as np
import torch

FRAMEWORKS = ["numpy", "torch", "jax", "cupy"]
DEVICES = ["cpu", "gpu"]
SKIP_XP_DEVICES = [("numpy", "gpu"), ("cupy", "cpu")]
TIMEOUT = 60 * 5
ROOT = Path(__file__).parent


def to_xp(xp, array, device):
    """Move a numpy array to `xp` on `device`.

    Args:
        xp: Framework name, one of FRAMEWORKS.
        array: Source numpy array.
        device: "cpu" or "gpu".

    Returns:
        The array in the framework's own type, resident on `device`.
    """
    if xp == "numpy":
        return array
    if xp == "torch":
        return torch.from_numpy(array).to("cuda" if device == "gpu" else "cpu")
    if xp == "jax":
        return jax.numpy.asarray(array, device=jax.devices(device)[0])
    if xp == "cupy":
        assert device == "gpu", "cupy runs on gpu only"
        return cupy.asarray(array)
    raise ValueError(f"Unknown framework {xp}")


def to_numpy(array, xp):
    """Bring an array from `xp` back to numpy."""
    if xp == "numpy":
        return array
    if xp == "torch":
        return array.detach().cpu().numpy()
    if xp == "cupy":
        return cupy.asnumpy(array)
    return np.asarray(array)


def device_of(array):
    """Report which device an array lives on, as "cpu" or "gpu"."""
    return "gpu" if "cuda" in str(getattr(array, "device", "cpu")).lower() else "cpu"


def timed_call(xp, device, test, jax_test):
    """Pick the timed callable, synchronizing GPU work.

    torch and cupy launch kernels asynchronously, so an unsynchronized timer measures
    only the launch, not the computation. We force a device synchronization after the
    call on GPU. jax uses its own block_until_ready path in `jax_test`.
    """
    if xp == "jax":
        return jax_test
    if xp == "torch" and device == "gpu":
        return lambda: (test(), torch.cuda.synchronize())
    if xp == "cupy":
        return lambda: (test(), cupy.cuda.Device().synchronize())
    return test


def time_case(setup, test, repeat, number):
    """Time `test`, returning one seconds-per-call sample per repeat.

    `setup` runs once up front and then once per repeat, always outside the timed
    region. Cases rebind their input state from `setup` via `nonlocal`, so every
    repeat measures freshly built data.

    Args:
        setup: Callable building the inputs and any jitted wrapper.
        test: The timed callable, already wrapped for device synchronization.
        repeat: Number of samples to collect.
        number: Calls per sample. Timings are divided by it.

    Returns:
        Array of `repeat` per-call times in seconds, empty if the case is too slow.
    """
    setup()
    start = timeit.default_timer()
    test()
    elapsed = timeit.default_timer() - start
    if elapsed > TIMEOUT / (repeat * number):
        print(f"  aborting: one call took {elapsed:.2f}s, over the {TIMEOUT}s budget")
        return np.array([])
    timer = timeit.Timer(stmt=test, setup=setup)
    return np.array(timer.repeat(repeat=repeat, number=number)) / number


def sample_sizes(low, high):
    """Sample sizes for a sweep, one per decade from 10**low to 10**high."""
    sizes = np.logspace(low, high, high - low + 1).astype(int)
    return np.sort(np.array(sorted(set(sizes.tolist()))))


def result_path(mirror, variant, xp, device, fn):
    """Path of one result file, mirroring the scipy module tree."""
    return ROOT / "results" / mirror / variant / xp / device / f"{fn}.json"


def load_result(mirror, variant, xp, device, fn):
    """Load one result file as {sample size: [seconds per call]}, None if absent."""
    path = result_path(mirror, variant, xp, device, fn)
    return json.loads(path.read_text()) if path.exists() else None


def save_result(mirror, variant, xp, device, fn, n_samples, timings, append=False):
    """Merge one sample size into a result file, keeping the other sizes.

    With `append`, the timings extend what is already stored for that size instead of
    replacing it, so repeated invocations accumulate samples across processes. Process
    to process variation dominates on some backends, and only samples drawn from
    separate processes capture it.
    """
    path = result_path(mirror, variant, xp, device, fn)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(path.read_text()) if path.exists() else {}
    key = str(int(n_samples))
    previous = results[key] if append and key in results else []
    results[key] = previous + timings
    path.write_text(json.dumps(results, indent=2))


def _skip_reason(exc):
    """Why a case cannot run on this backend, or None if the error is a real bug."""
    if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
        return "out of memory"
    if isinstance(exc, AttributeError | ValueError | TypeError):
        # A function that is not array API converted yet raises here, for instance by
        # calling np.asarray on a traced jax array.
        return f"unsupported: {exc}"
    return None


def sweep(
    mirror,
    cases,
    fns,
    frameworks,
    devices,
    low,
    high,
    repeat,
    number,
    variant,
    append=False,
):
    """Run cases across frameworks, devices and sample sizes, saving as we go.

    A case that runs out of memory or hits an unsupported backend operation skips the
    remaining, larger sample sizes for that framework and device.

    Args:
        mirror: Scipy module path of the suite, e.g. "spatial/distance".
        cases: Mapping of case name to builder, from the registry.
        fns: Case names to run.
        frameworks: Framework names to run.
        devices: Device names to run.
        low: Log10 of the smallest sample size.
        high: Log10 of the largest sample size.
        repeat: Samples per sample size.
        number: Calls per sample.
        variant: Result tree to write into, "current" or "baseline".
        append: Add to the stored samples instead of replacing them, so repeated
            invocations accumulate across processes.
    """
    for xp in frameworks:
        for fn in fns:
            for device in devices:
                if (xp, device) in SKIP_XP_DEVICES:
                    continue
                for n_samples in sample_sizes(low, high):
                    print(f"{mirror}/{fn}: {xp} {device} n={n_samples}")
                    try:
                        setup, test, jax_test = cases[fn](xp, device, int(n_samples))
                        timings = time_case(
                            setup,
                            timed_call(xp, device, test, jax_test),
                            repeat,
                            number,
                        )
                    except Exception as exc:
                        reason = _skip_reason(exc)
                        if reason is None:
                            raise
                        print(f"  SKIP {fn} on {xp} {device} - {reason}")
                        break
                    if len(timings) == 0:
                        break
                    save_result(
                        mirror,
                        variant,
                        xp,
                        device,
                        fn,
                        n_samples,
                        timings.tolist(),
                        append,
                    )
