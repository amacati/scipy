"""Benchmarks for scipy.spatial.distance."""

from functools import partial

import jax
import numpy as np
from scipy.spatial import distance

from xp_bench import device_of, register, to_xp

N_FEATURES = 10  # feature dimension of the observation matrices of pdist and cdist

# Pair metrics take two 1-D vectors and return a scalar. Scaling the vector length
# measures the elementwise work an array API backend parallelizes. Each entry is
# (callable, dtype, jittable).
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
    "sokalsneath": (
        distance.sokalsneath,
        "bool",
        False,
    ),  # data-dependent guard, no jit
}

# Validators take either a square distance matrix ("dm") or a condensed vector ("y").
VALIDATOR_SPECS = {
    "is_valid_dm": (distance.is_valid_dm, "dm"),
    "num_obs_dm": (distance.num_obs_dm, "dm"),
    "is_valid_y": (distance.is_valid_y, "y"),
    "num_obs_y": (distance.num_obs_y, "y"),
}


def create_vector(xp, device, n, dtype="float64"):
    """Random vector of length n.

    Args:
        xp: Framework name.
        device: "cpu" or "gpu".
        n: Vector length.
        dtype: "bool" for the boolean metrics, any float dtype otherwise.

    Returns:
        The vector on `device`. Boolean vectors always contain a True and, from n = 2
        on, also a False, so sokalsneath, yule, dice and russellrao stay defined.
    """
    if dtype == "bool":
        arr = np.random.rand(n) > 0.5
        arr[0] = True
        if n > 1:
            arr[1] = False
    else:
        arr = np.random.rand(n).astype(dtype)
    return to_xp(xp, arr, device)


def create_matrix(xp, device, m):
    """Observation matrix of m observations with N_FEATURES features each."""
    return to_xp(xp, np.random.rand(m, N_FEATURES), device)


def create_distance_matrix(xp, device, k):
    """Symmetric k by k matrix with an exact zero diagonal.

    Both properties are required for is_valid_dm to run all of its checks instead of
    short-circuiting on the first failing one.
    """
    a = np.random.rand(k, k)
    d = a + a.T
    np.fill_diagonal(d, 0.0)
    return to_xp(xp, d, device)


def create_condensed(xp, device, n):
    """Condensed distance vector whose length is the valid binomial size nearest n."""
    m = int((1 + (1 + 8 * n) ** 0.5) // 2)
    return to_xp(xp, np.random.rand(max(m * (m - 1) // 2, 1)), device)


def _pair_case(call, dtype, jittable):
    """Case builder for a two-vector scalar metric, scaling the vector length."""

    def build(xp, device, n_samples):
        u, v, jfn = None, None, None

        def setup():
            nonlocal u, v, jfn
            u = create_vector(xp, device, n_samples, dtype)
            v = create_vector(xp, device, n_samples, dtype)
            assert device_of(u) == device, f"setup on {device_of(u)}, want {device}"
            if xp == "jax" and jittable:
                jfn = jax.jit(call)
                jax.block_until_ready(jfn(u, v))

        def test():
            return call(u, v)

        def jax_test():
            jax.block_until_ready((jfn if jittable else call)(u, v))

        return setup, test, jax_test

    return build


def _validator_case(call, kind):
    """Case builder for a validator, fed a distance matrix or a condensed vector."""
    create = create_distance_matrix if kind == "dm" else create_condensed

    def build(xp, device, n_samples):
        data = None

        def setup():
            nonlocal data
            data = create(xp, device, n_samples)
            assert device_of(data) == device, (
                f"setup on {device_of(data)}, not {device}"
            )

        def test():
            return call(data)

        def jax_test():
            jax.block_until_ready(call(data))

        return setup, test, jax_test

    return build


for _name, (_call, _dtype, _jittable) in PAIR_SPECS.items():
    register(_name, _pair_case(_call, _dtype, _jittable))

for _name, (_call, _kind) in VALIDATOR_SPECS.items():
    register(_name, _validator_case(_call, _kind))


@register
def mahalanobis(xp, device, n_samples):
    u, v, VI = None, None, None

    def setup():
        nonlocal u, v, VI
        u = create_vector(xp, device, n_samples)
        v = create_vector(xp, device, n_samples)
        VI = to_xp(xp, np.random.rand(n_samples, n_samples), device)
        assert device_of(u) == device, f"setup on {device_of(u)}, want {device}"

    def test():
        return distance.mahalanobis(u, v, VI)

    def jax_test():
        jax.block_until_ready(distance.mahalanobis(u, v, VI))

    return setup, test, jax_test


@register
def seuclidean(xp, device, n_samples):
    u, v, V = None, None, None

    def setup():
        nonlocal u, v, V
        u = create_vector(xp, device, n_samples)
        v = create_vector(xp, device, n_samples)
        V = create_vector(xp, device, n_samples)  # positive variances in [0, 1)
        assert device_of(u) == device, f"setup on {device_of(u)}, want {device}"

    def test():
        return distance.seuclidean(u, v, V)

    def jax_test():
        jax.block_until_ready(distance.seuclidean(u, v, V))

    return setup, test, jax_test


@register
def pdist(xp, device, n_samples):
    X = None

    def setup():
        nonlocal X
        X = create_matrix(xp, device, n_samples)
        assert device_of(X) == device, f"setup on {device_of(X)}, want {device}"

    def test():
        return distance.pdist(X)

    def jax_test():
        jax.block_until_ready(distance.pdist(X))

    return setup, test, jax_test


@register
def cdist(xp, device, n_samples):
    XA, XB = None, None

    def setup():
        nonlocal XA, XB
        XA = create_matrix(xp, device, n_samples)
        XB = create_matrix(xp, device, n_samples)
        assert device_of(XA) == device, f"setup on {device_of(XA)}, want {device}"

    def test():
        return distance.cdist(XA, XB)

    def jax_test():
        jax.block_until_ready(distance.cdist(XA, XB))

    return setup, test, jax_test


@register
def squareform(xp, device, n_samples):
    y = None

    def setup():
        nonlocal y
        y = create_condensed(xp, device, n_samples)
        assert device_of(y) == device, f"setup on {device_of(y)}, want {device}"

    def test():
        return distance.squareform(y)

    def jax_test():
        jax.block_until_ready(distance.squareform(y))

    return setup, test, jax_test
