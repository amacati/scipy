"""jax pytree registration for Rotation and RigidTransform.

Importing this module registers both classes, so jax can trace and jit functions that
take or return them. Both transform suites import it and the module cache keeps the
registration to a single call per process.
"""

from jax.tree_util import register_pytree_node
from scipy._lib._array_api import array_namespace
from scipy.spatial.transform import RigidTransform, Rotation
from scipy.spatial.transform import _rigid_transform, _rotation


def rotation_unflatten(_, children):
    """Rebuild a Rotation from its quaternion, bypassing __init__.

    __init__ normalizes a second time and dispatches through the non-jitted Array API
    backend, which costs far more than the attribute assignments here.
    """
    rotation = Rotation.__new__(Rotation)
    # A different backend may be registered for jax, so we look up the current one and
    # fall back to the Array API backend.
    xp = array_namespace(children[0])
    rotation._xp = xp
    rotation._backend = _rotation.backend_registry.get(xp, _rotation.xp_backend)
    rotation._quat = children[0]
    # _single is False because the Array API backend broadcasts by default and therefore
    # returns the correct shape without the _single workaround.
    rotation._single = False
    return rotation


def rigid_transform_unflatten(_, children):
    """Rebuild a RigidTransform from its matrix, bypassing __init__.

    __init__ normalizes a second time and dispatches through the non-jitted Array API
    backend, which costs far more than the attribute assignments here.
    """
    transform = RigidTransform.__new__(RigidTransform)
    # A different backend may be registered for jax, so we look up the current one and
    # fall back to the Array API backend.
    xp = array_namespace(children[0])
    transform._xp = xp
    transform._backend = _rigid_transform.backend_registry.get(
        xp, _rigid_transform.xp_backend
    )
    transform._matrix = children[0]
    # _single is False because the Array API backend broadcasts by default and therefore
    # returns the correct shape without the _single workaround.
    transform._single = False
    return transform


register_pytree_node(Rotation, lambda r: ((r._quat,), None), rotation_unflatten)
register_pytree_node(
    RigidTransform, lambda tf: ((tf._matrix,), None), rigid_transform_unflatten
)
