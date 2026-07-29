"""Benchmarks for scipy.spatial.transform.RigidTransform."""

from functools import partial

import jax
from scipy._lib._array_api import array_namespace
from scipy.spatial.transform import RigidTransform
from scipy.spatial.transform import Rotation as R

from xp_bench import device_of, register

from . import _pytree  # noqa: F401  registers Rotation and RigidTransform with jax
from .rotation import random_data


def method_case(xp, device, n_samples, *, call, args=()):
    """Instance method of a transform batch.

    The timed call is the bound method, as a caller would write it. jax jits the unbound
    method instead and passes the transform as its first argument.
    """
    tf, jfn = None, None

    def setup():
        nonlocal tf, jfn
        q, p = random_data(xp, device, n_samples)
        assert device_of(q) == device, f"setup on {device_of(q)}, want {device}"
        tf = RigidTransform.from_components(p, R.from_quat(q))
        if xp == "jax":
            jfn = jax.jit(call)
            jax.block_until_ready(jfn(tf, *args))

    def test():
        return getattr(tf, call.__name__)(*args)

    def jax_test():
        jax.block_until_ready(jfn(tf, *args))

    return setup, test, jax_test


def constructor_case(xp, device, n_samples, *, source, call):
    """Constructor, fed the argument tuple `source` builds from quaternions and translations."""
    args, jfn = None, None

    def setup():
        nonlocal args, jfn
        q, p = random_data(xp, device, n_samples)
        assert device_of(q) == device, f"setup on {device_of(q)}, want {device}"
        args = source(q, p)
        if xp == "jax":
            jfn = jax.jit(call)
            jax.block_until_ready(jfn(*args))

    def test():
        return call(*args)

    def jax_test():
        jax.block_until_ready(jfn(*args))

    return setup, test, jax_test


register("as_matrix", partial(method_case, call=RigidTransform.as_matrix))
register("as_components", partial(method_case, call=RigidTransform.as_components))
register("as_exp_coords", partial(method_case, call=RigidTransform.as_exp_coords))
register("as_dual_quat", partial(method_case, call=RigidTransform.as_dual_quat))
register("inv", partial(method_case, call=RigidTransform.inv))
register("pow", partial(method_case, call=RigidTransform.__pow__, args=(2.0,)))

register(
    "from_rotation",
    partial(
        constructor_case,
        source=lambda q, p: (R.from_quat(q),),
        call=RigidTransform.from_rotation,
    ),
)
register(
    "from_components",
    partial(
        constructor_case,
        source=lambda q, p: (p, R.from_quat(q)),
        call=RigidTransform.from_components,
    ),
)
register(
    "from_translation",
    partial(
        constructor_case,
        source=lambda q, p: (p,),
        call=RigidTransform.from_translation,
    ),
)
register(
    "from_matrix",
    partial(
        constructor_case,
        source=lambda q, p: (
            RigidTransform.from_components(p, R.from_quat(q)).as_matrix(),
        ),
        call=RigidTransform.from_matrix,
    ),
)


@register
def apply(xp, device, n_samples):
    tf, vectors, jfn = None, None, None

    def setup():
        nonlocal tf, vectors, jfn
        q, p = random_data(xp, device, n_samples)
        assert device_of(q) == device, f"setup on {device_of(q)}, want {device}"
        tf = RigidTransform.from_components(p, R.from_quat(q))
        vectors = p  # the translations double as the vectors to transform
        if xp == "jax":
            jfn = jax.jit(RigidTransform.apply)
            jax.block_until_ready(jfn(tf, vectors))

    def test():
        return tf.apply(vectors)

    def jax_test():
        jax.block_until_ready(jfn(tf, vectors))

    return setup, test, jax_test


@register
def mul(xp, device, n_samples):
    tf1, tf2, jfn = None, None, None

    def setup():
        nonlocal tf1, tf2, jfn
        q1, p1 = random_data(xp, device, n_samples)
        q2, p2 = random_data(xp, device, n_samples)
        assert device_of(q1) == device, f"setup on {device_of(q1)}, want {device}"
        tf1 = RigidTransform.from_components(p1, R.from_quat(q1))
        tf2 = RigidTransform.from_components(p2, R.from_quat(q2))
        if xp == "jax":
            jfn = jax.jit(RigidTransform.__mul__)
            jax.block_until_ready(jfn(tf1, tf2))

    def test():
        return tf1 * tf2

    def jax_test():
        jax.block_until_ready(jfn(tf1, tf2))

    return setup, test, jax_test


@register
def concatenate(xp, device, n_samples):
    tf1, tf2, jfn = None, None, None

    def setup():
        nonlocal tf1, tf2, jfn
        # Two halves, so the concatenated batch holds n_samples transforms.
        q1, p1 = random_data(xp, device, n_samples // 2)
        q2, p2 = random_data(xp, device, n_samples // 2)
        assert device_of(q1) == device, f"setup on {device_of(q1)}, want {device}"
        tf1 = RigidTransform.from_components(p1, R.from_quat(q1))
        tf2 = RigidTransform.from_components(p2, R.from_quat(q2))
        if xp == "jax":
            jfn = jax.jit(RigidTransform.concatenate)
            jax.block_until_ready(jfn([tf1, tf2]))

    def test():
        return RigidTransform.concatenate([tf1, tf2])

    def jax_test():
        jax.block_until_ready(jfn([tf1, tf2]))

    return setup, test, jax_test


@register
def from_exp_coords(xp, device, n_samples):
    exp_coords, jfn = None, None

    def setup():
        nonlocal exp_coords, jfn
        p1 = random_data(xp, device, n_samples)[1]
        p2 = random_data(xp, device, n_samples)[1]
        assert device_of(p1) == device, f"setup on {device_of(p1)}, want {device}"
        # 6D exponential coordinates, a rotation vector followed by a translation
        exp_coords = array_namespace(p1).concat([p1, p2], axis=-1)
        if xp == "jax":
            jfn = jax.jit(RigidTransform.from_exp_coords)
            jax.block_until_ready(jfn(exp_coords))

    def test():
        return RigidTransform.from_exp_coords(exp_coords)

    def jax_test():
        jax.block_until_ready(jfn(exp_coords))

    return setup, test, jax_test


@register
def from_dual_quat(xp, device, n_samples):
    dual_quat, jfn = None, None

    def setup():
        nonlocal dual_quat, jfn
        q1 = random_data(xp, device, n_samples)[0]
        q2 = random_data(xp, device, n_samples)[0]
        assert device_of(q1) == device, f"setup on {device_of(q1)}, want {device}"
        # Two independent Gaussian quaternions, which is not a valid dual quaternion
        dual_quat = array_namespace(q1).concat([q1, q2], axis=-1)
        if xp == "jax":
            jfn = jax.jit(RigidTransform.from_dual_quat)
            jax.block_until_ready(jfn(dual_quat))

    def test():
        return RigidTransform.from_dual_quat(dual_quat)

    def jax_test():
        jax.block_until_ready(jfn(dual_quat))

    return setup, test, jax_test
