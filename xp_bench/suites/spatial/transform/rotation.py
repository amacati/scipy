"""Benchmarks for scipy.spatial.transform.Rotation."""

from functools import partial

import cupy
import jax
import jax.numpy as jp
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from xp_bench import device_of, register, to_xp

from . import _pytree  # noqa: F401  registers Rotation and RigidTransform with jax


@partial(jax.jit, static_argnums=[0, 1])
def jax_qp(n_samples, device):
    """Quaternions and translations drawn on a jax device."""
    dev = jax.devices(device)[0]
    q = jp.array(jax.random.normal(jax.random.PRNGKey(0), (n_samples, 4)), device=dev)
    p = jp.array(jax.random.uniform(jax.random.PRNGKey(0), (n_samples, 3)), device=dev)
    return q, p


def random_data(xp, device, n_samples):
    """Random quaternions and translation vectors, drawn natively per framework.

    Args:
        xp: Framework name.
        device: "cpu" or "gpu".
        n_samples: Batch size.

    Returns:
        An (n_samples, 4) standard normal quaternion batch and an (n_samples, 3) uniform
        translation batch, both resident on `device`. The quaternions are unnormalized.
    """
    if xp == "numpy":
        return np.random.randn(n_samples, 4), np.random.rand(n_samples, 3)
    if xp == "torch":
        dev = "cuda" if device == "gpu" else "cpu"
        return (
            torch.randn(n_samples, 4, device=dev),
            torch.rand(n_samples, 3, device=dev),
        )
    if xp == "jax":
        return jax_qp(n_samples, device)
    if xp == "cupy":
        return cupy.random.randn(n_samples, 4), cupy.random.rand(n_samples, 3)
    raise ValueError(f"Unknown framework {xp}")


def method_case(xp, device, n_samples, *, call, args=()):
    """Instance method of a rotation batch.

    The timed call is the bound method, as a caller would write it. jax jits the unbound
    method instead and passes the rotation as its first argument.
    """
    r, jfn = None, None

    def setup():
        nonlocal r, jfn
        q, _ = random_data(xp, device, n_samples)
        assert device_of(q) == device, f"setup on {device_of(q)}, want {device}"
        r = R.from_quat(q)
        if xp == "jax":
            jfn = jax.jit(call)
            jax.block_until_ready(jfn(r, *args))

    def test():
        return getattr(r, call.__name__)(*args)

    def jax_test():
        jax.block_until_ready(jfn(r, *args))

    return setup, test, jax_test


def constructor_case(xp, device, n_samples, *, source, call):
    """Constructor, fed the array `source` builds from a quaternion batch."""
    data, jfn = None, None

    def setup():
        nonlocal data, jfn
        q, _ = random_data(xp, device, n_samples)
        assert device_of(q) == device, f"setup on {device_of(q)}, want {device}"
        data = source(q)
        if xp == "jax":
            jfn = jax.jit(call)
            jax.block_until_ready(jfn(data))

    def test():
        return call(data)

    def jax_test():
        jax.block_until_ready(jfn(data))

    return setup, test, jax_test


register("as_quat", partial(method_case, call=R.as_quat))
register("as_matrix", partial(method_case, call=R.as_matrix))
register("as_rotvec", partial(method_case, call=R.as_rotvec))
register("as_mrp", partial(method_case, call=R.as_mrp))
register("magnitude", partial(method_case, call=R.magnitude))
register("mean", partial(method_case, call=R.mean))
register("inv", partial(method_case, call=R.inv))
register("pow", partial(method_case, call=R.__pow__, args=(2.0,)))

register("from_quat", partial(constructor_case, source=lambda q: q, call=R.from_quat))
register(
    "from_matrix",
    partial(
        constructor_case,
        source=lambda q: R.from_quat(q).as_matrix(),
        call=R.from_matrix,
    ),
)
register(
    "from_matrix_assume_valid",
    partial(
        constructor_case,
        source=lambda q: R.from_quat(q).as_matrix(),
        call=partial(R.from_matrix, assume_valid=True),
    ),
)
register(
    "from_rotvec",
    partial(
        constructor_case,
        source=lambda q: R.from_quat(q).as_rotvec(),
        call=R.from_rotvec,
    ),
)
register(
    "from_mrp",
    partial(
        constructor_case, source=lambda q: R.from_quat(q).as_mrp(), call=R.from_mrp
    ),
)
register(
    "from_euler",
    partial(
        constructor_case,
        source=lambda q: R.from_quat(q).as_euler("xyz"),
        call=lambda angles: R.from_euler("xyz", angles),
    ),
)


@register
def apply(xp, device, n_samples):
    r, p, jfn = None, None, None

    def setup():
        nonlocal r, p, jfn
        q, p = random_data(xp, device, n_samples)
        assert device_of(q) == device, f"setup on {device_of(q)}, want {device}"
        r = R.from_quat(q)
        if xp == "jax":
            jfn = jax.jit(R.apply)
            jax.block_until_ready(jfn(r, p))

    def test():
        return r.apply(p)

    def jax_test():
        jax.block_until_ready(jfn(r, p))

    return setup, test, jax_test


@register
def approx_equal(xp, device, n_samples):
    r1, r2, jfn = None, None, None

    def setup():
        nonlocal r1, r2, jfn
        q, _ = random_data(xp, device, n_samples)
        assert device_of(q) == device, f"setup on {device_of(q)}, want {device}"
        r1 = R.from_quat(q)
        r2 = R.from_quat(q)  # the same rotation, so every comparison matches
        if xp == "jax":
            jfn = jax.jit(R.approx_equal)
            jax.block_until_ready(jfn(r1, r2))

    def test():
        return r1.approx_equal(r2)

    def jax_test():
        jax.block_until_ready(jfn(r1, r2))

    return setup, test, jax_test


@register
def mul(xp, device, n_samples):
    r1, r2, jfn = None, None, None

    def setup():
        nonlocal r1, r2, jfn
        q1, _ = random_data(xp, device, n_samples)
        q2, _ = random_data(xp, device, n_samples)
        assert device_of(q1) == device, f"setup on {device_of(q1)}, want {device}"
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        if xp == "jax":
            jfn = jax.jit(R.__mul__)
            jax.block_until_ready(jfn(r1, r2))

    def test():
        return r1 * r2

    def jax_test():
        jax.block_until_ready(jfn(r1, r2))

    return setup, test, jax_test


@register
def reduce(xp, device, n_samples):
    r, left, right, jfn = None, None, None, None

    def setup():
        nonlocal r, left, right, jfn
        q, _ = random_data(xp, device, n_samples)
        assert device_of(q) == device, f"setup on {device_of(q)}, want {device}"
        r = R.from_quat(q)
        left = R.from_quat(random_data(xp, device, n_samples)[0])
        right = R.from_quat(random_data(xp, device, n_samples)[0])
        if xp == "jax":
            jfn = jax.jit(R.reduce)
            jax.block_until_ready(jfn(r, left, right))

    def test():
        return R.reduce(r, left, right)

    def jax_test():
        jax.block_until_ready(jfn(r, left, right))

    return setup, test, jax_test


@register
def align_vectors(xp, device, n_samples):
    v1, v2, jfn = None, None, None

    def setup():
        nonlocal v1, v2, jfn
        v1 = random_data(xp, device, n_samples)[1]
        v2 = random_data(xp, device, n_samples)[1]
        assert device_of(v1) == device, f"setup on {device_of(v1)}, want {device}"
        if xp == "jax":
            jfn = jax.jit(R.align_vectors)
            jax.block_until_ready(jfn(v1, v2))

    def test():
        return R.align_vectors(v1, v2)

    def jax_test():
        jax.block_until_ready(jfn(v1, v2))

    return setup, test, jax_test


@register
def as_euler(xp, device, n_samples):
    r, jfn = None, None

    def setup():
        nonlocal r, jfn
        q, _ = random_data(xp, device, n_samples)
        assert device_of(q) == device, f"setup on {device_of(q)}, want {device}"
        r = R.from_quat(q)
        if xp == "jax":
            jfn = jax.jit(R.as_euler, static_argnames=["seq"])
            jax.block_until_ready(jfn(r, seq="xyz"))

    def test():
        return r.as_euler("xyz")

    def jax_test():
        jax.block_until_ready(jfn(r, seq="xyz"))

    return setup, test, jax_test


@register
def as_davenport(xp, device, n_samples):
    r, axes, jfn = None, None, None

    def setup():
        nonlocal r, axes, jfn
        q, _ = random_data(xp, device, n_samples)
        assert device_of(q) == device, f"setup on {device_of(q)}, want {device}"
        r = R.from_quat(q)
        axes = to_xp(xp, np.eye(3), device)
        if xp == "jax":
            jfn = jax.jit(R.as_davenport, static_argnames=["order"])
            jax.block_until_ready(jfn(r, axes, order="e"))

    def test():
        return r.as_davenport(axes, "e")

    def jax_test():
        jax.block_until_ready(jfn(r, axes, order="e"))

    return setup, test, jax_test


@register
def from_davenport(xp, device, n_samples):
    p, jfn = None, None

    def setup():
        nonlocal p, jfn
        p = random_data(xp, device, n_samples)[1]
        assert device_of(p) == device, f"setup on {device_of(p)}, want {device}"
        if xp == "jax":
            jfn = jax.jit(partial(R.from_davenport, order="e"))
            jax.block_until_ready(jfn(p[0, :], angles=p[:, 0:1]))

    def test():
        # The axis and angle slicing is part of the timed call.
        return R.from_davenport(p[0, :], "e", p[:, 0:1])

    def jax_test():
        jax.block_until_ready(jfn(p[0, :], angles=p[:, 0:1]))

    return setup, test, jax_test
