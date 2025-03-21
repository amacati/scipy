# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.spatial` namespace for importing the functions
# included below.

from __future__ import annotations

from typing import Iterable

import numpy as np

from scipy._lib.deprecation import _sub_module_deprecation
import scipy.spatial.transform._cython_backend as cython_backend
import scipy.spatial.transform._array_api_backend as array_api_backend
from scipy._lib._array_api import array_namespace, Array, is_numpy, ArrayLike
from scipy._lib.array_api_compat import device
import scipy._lib.array_api_extra as xpx
from scipy._lib._util import _transition_to_rng


__all__ = ["Rotation", "Slerp"]  # noqa: F822

# Fast path for numpy arrays: If quat is a numpy object, we call the Cython backend
backend_registry = {array_namespace(np.empty(0)): cython_backend}


class Rotation:
    def __init__(
        self,
        quat: ArrayLike,
        normalize: bool = True,
        copy: bool = True,
        scalar_first: bool = False,
    ):
        quat = self._to_array(quat)
        xp = array_namespace(quat)
        # Legacy behavior for cython backend: Differentiate between single quat and batched quats
        # We only use this for the cython backend. The Array API backend uses broadcasting by
        # default and hence returns the correct shape without additional logic
        self._single = quat.ndim == 1 and is_numpy(xp)
        if self._single:
            quat = xpx.atleast_nd(quat, ndim=2, xp=xp)
        self._backend = backend_registry.get(xp, array_api_backend)
        self._quat: Array = self._backend.from_quat(
            quat, normalize=normalize, copy=copy, scalar_first=scalar_first
        )

    @classmethod
    def from_quat(cls, quat: ArrayLike, *, scalar_first: bool = False) -> Rotation:
        return cls(quat, normalize=True, scalar_first=scalar_first)

    @classmethod
    def from_matrix(cls, matrix: ArrayLike) -> Rotation:
        backend = backend_registry.get(array_namespace(matrix), array_api_backend)
        quat = backend.from_matrix(matrix)
        return cls(quat, normalize=False, copy=False)

    @classmethod
    def from_rotvec(cls, rotvec: ArrayLike, degrees: bool = False) -> Rotation:
        backend = backend_registry.get(array_namespace(rotvec), array_api_backend)
        quat = backend.from_rotvec(rotvec, degrees=degrees)
        return cls(quat, normalize=False, copy=False)

    @classmethod
    def from_mrp(cls, mrp: ArrayLike) -> Rotation:
        backend = backend_registry.get(array_namespace(mrp), array_api_backend)
        quat = backend.from_mrp(mrp)
        return cls(quat, normalize=False, copy=False)

    @classmethod
    def from_euler(cls, seq: str, angles: ArrayLike, degrees: bool = False) -> Rotation:
        backend = backend_registry.get(array_namespace(angles), array_api_backend)
        quat = backend.from_euler(seq, angles, degrees=degrees)
        return cls(quat, normalize=False, copy=False)

    def as_quat(self, canonical=False, *, scalar_first=False):
        quat = self._backend.as_quat(
            self._quat, canonical=canonical, scalar_first=scalar_first
        )
        if self._single:
            return quat[0, ...]
        return quat

    def as_matrix(self) -> Array:
        matrix = self._backend.as_matrix(self._quat)
        if self._single:
            return matrix[0, ...]
        return matrix

    def as_rotvec(self, degrees: bool = False) -> Array:
        rotvec = self._backend.as_rotvec(self._quat, degrees=degrees)
        if self._single:
            return rotvec[0, ...]
        return rotvec

    def as_mrp(self) -> Array:
        mrp = self._backend.as_mrp(self._quat)
        if self._single:
            return mrp[0, ...]
        return mrp

    def as_euler(self, seq: str, degrees: bool = False) -> Array:
        euler = self._backend.as_euler(self._quat, seq, degrees=degrees)
        if self._single:
            return euler[0, ...]
        return euler

    @classmethod
    @_transition_to_rng("random_state", position_num=2)
    def random(cls, num: int | None = None, rng: np.random.Generator | None = None):
        # DECISION: How do we handle random numbers in other frameworks?
        # TODO: The array API does not have a unified random interface. This method only creates
        # numpy arrays. If we do want to support other frameworks, we need a way to handle other rng
        # implementations.
        sample = cython_backend.random(num, rng)
        return cls(sample, normalize=True, copy=False)

    @classmethod
    def identity(cls, num: int | None = None) -> Rotation:
        return cls(cython_backend.identity(num), normalize=False, copy=False)

    def inv(self):
        """Invert this rotation.

        Composition of a rotation with its inverse results in an identity
        transformation.

        Returns
        -------
        inverse : `Rotation` instance
            Object containing inverse of the rotations in the current instance.

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        Inverting a single rotation:

        >>> p = R.from_euler('z', 45, degrees=True)
        >>> q = p.inv()
        >>> q.as_euler('zyx', degrees=True)
        array([-45.,   0.,   0.])

        Inverting multiple rotations:

        >>> p = R.from_rotvec([[0, 0, np.pi/3], [-np.pi/4, 0, 0]])
        >>> q = p.inv()
        >>> q.as_rotvec()
        array([[-0.        , -0.        , -1.04719755],
               [ 0.78539816, -0.        , -0.        ]])

        """
        q_inv = self._backend.inv(self._quat)
        if self._single:
            q_inv = q_inv[0, ...]
        return Rotation(q_inv, normalize=False, copy=False)

    def magnitude(self):
        magnitude = self._backend.magnitude(self._quat)
        if self._single:
            return magnitude[0, ...]
        return magnitude

    def approx_equal(
        self, other: Rotation, atol: float | None = None, degrees: bool = False
    ):
        return self._backend.approx_equal(
            self._quat, other._quat, atol=atol, degrees=degrees
        )

    def mean(self, weights: ArrayLike | None = None) -> Rotation:
        return Rotation(
            self._backend.mean(self._quat, weights=weights), normalize=False
        )

    def reduce(
        self,
        left: Rotation | None = None,
        right: Rotation | None = None,
        return_indices: bool = False,
    ) -> Rotation:
        left = left.as_quat() if left is not None else None
        right = right.as_quat() if right is not None else None
        reduced, left_idx, right_idx = self._backend.reduce(
            self._quat, left=left, right=right
        )
        rot = Rotation(reduced, normalize=False, copy=False)
        if return_indices:
            left_idx = left_idx if left is not None else None
            right_idx = right_idx if right is not None else None
            return rot, left_idx, right_idx
        return rot

    def apply(self, points: Array, inverse: bool = False) -> Array:
        points = array_namespace(self._quat).asarray(
            points,
            device=device(self._quat),
            dtype=array_api_backend.atleast_f32(self._quat),
        )
        result = self._backend.apply(self._quat, points, inverse=inverse)
        if self._single and points.ndim == 1:
            return result[0, ...]
        return result

    @classmethod
    def align_vectors(
        cls,
        a: Array,
        b: Array,
        weights: Array | None = None,
        return_sensitivity: bool = False,
    ) -> tuple[Rotation, float] | tuple[Rotation, float, Array]:
        backend = backend_registry.get(array_namespace(a), array_api_backend)
        q, rssd, sensitivity = backend.align_vectors(a, b, weights, return_sensitivity)
        if return_sensitivity:
            return cls(q, normalize=False, copy=False), rssd, sensitivity
        return cls(q, normalize=False, copy=False), rssd

    @classmethod
    def concatenate(cls, rotations: Rotation | Iterable[Rotation]) -> Rotation:
        """Concatenate a sequence of `Rotation` objects into a single object.

        This is useful if you want to, for example, take the mean of a set of
        rotations and need to pack them into a single object to do so.

        Parameters
        ----------
        rotations : sequence of `Rotation` objects
            The rotations to concatenate. If a single `Rotation` object is
            passed in, a copy is returned.

        Returns
        -------
        concatenated : `Rotation` instance
            The concatenated rotations.

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> r1 = R.from_rotvec([0, 0, 1])
        >>> r2 = R.from_rotvec([0, 0, 2])
        >>> rc = R.concatenate([r1, r2])
        >>> rc.as_rotvec()
        array([[0., 0., 1.],
                [0., 0., 2.]])
        >>> rc.mean().as_rotvec()
        array([0., 0., 1.5])

        Concatenation of a split rotation recovers the original object.

        >>> rs = [r for r in rc]
        >>> R.concatenate(rs).as_rotvec()
        array([[0., 0., 1.],
                [0., 0., 2.]])

        Note that it may be simpler to create the desired rotations by passing
        in a single list of the data during initialization, rather then by
        concatenating:

        >>> R.from_rotvec([[0, 0, 1], [0, 0, 2]]).as_rotvec()
        array([[0., 0., 1.],
                [0., 0., 2.]])

        Notes
        -----
        .. versionadded:: 1.8.0
        """
        if isinstance(rotations, Rotation):
            return cls(rotations.as_quat(), normalize=False, copy=True)
        if not all(isinstance(x, Rotation) for x in rotations):
            raise TypeError("input must contain Rotation objects only")

        xp = array_namespace(rotations[0].as_quat())
        quats = xp.concat(
            [xpx.atleast_nd(x.as_quat(), ndim=2, xp=xp) for x in rotations]
        )
        return cls(quats, normalize=False)

    def __getitem__(self, indexer) -> Rotation:
        if self._single or self._quat.ndim == 1:
            raise TypeError("Single rotation is not subscriptable.")
        return Rotation(self._quat[indexer, ...], normalize=False)

    def __setitem__(self, indexer, value: Rotation):
        if self._single or self._quat.ndim == 1:
            raise TypeError("Single rotation is not subscriptable.")

        if not isinstance(value, Rotation):
            raise TypeError("value must be a Rotation object")

        self._quat = self._backend.setitem(self._quat, value.as_quat(), indexer)

    def __getstate__(self):
        return (self._quat, self._single)

    def __setstate__(self, state):
        quat, single = state
        xp = array_namespace(quat)
        self._backend = backend_registry.get(xp, array_api_backend)
        self._quat = xp.asarray(quat, copy=True)
        self._single = single

    @property
    def single(self):
        """Whether this instance represents a single rotation."""
        # TODO: Remove this once we properly support broadcasting with arbitrary
        # number of rotations
        return self._single

    def __bool__(self):
        """Comply with Python convention for objects to be True.

        Required because `Rotation.__len__()` is defined and not always truthy.
        """
        return True

    def __len__(self):
        """Number of rotations contained in this object.

        Multiple rotations can be stored in a single instance.

        Returns
        -------
        length : int
            Number of rotations stored in object.

        Raises
        ------
        TypeError if the instance was created as a single rotation.
        """
        # TODO: Should we replace this with a shape instead?
        if self._single or self._quat.ndim == 1:
            raise TypeError("Single rotation has no len().")
        return self._quat.shape[0]

    def __mul__(self, other):
        """Compose this rotation with the other.

        If `p` and `q` are two rotations, then the composition of 'q followed
        by p' is equivalent to `p * q`. In terms of rotation matrices,
        the composition can be expressed as
        ``p.as_matrix() @ q.as_matrix()``.

        Parameters
        ----------
        other : `Rotation` instance
            Object containing the rotations to be composed with this one. Note
            that rotation compositions are not commutative, so ``p * q`` is
            generally different from ``q * p``.

        Returns
        -------
        composition : `Rotation` instance
            This function supports composition of multiple rotations at a time.
            The following cases are possible:

            - Either ``p`` or ``q`` contains a single rotation. In this case
              `composition` contains the result of composing each rotation in
              the other object with the single rotation.
            - Both ``p`` and ``q`` contain ``N`` rotations. In this case each
              rotation ``p[i]`` is composed with the corresponding rotation
              ``q[i]`` and `output` contains ``N`` rotations.

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        Composition of two single rotations:

        >>> p = R.from_quat([0, 0, 1, 1])
        >>> q = R.from_quat([1, 0, 0, 1])
        >>> p.as_matrix()
        array([[ 0., -1.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])
        >>> q.as_matrix()
        array([[ 1.,  0.,  0.],
               [ 0.,  0., -1.],
               [ 0.,  1.,  0.]])
        >>> r = p * q
        >>> r.as_matrix()
        array([[0., 0., 1.],
               [1., 0., 0.],
               [0., 1., 0.]])

        Composition of two objects containing equal number of rotations:

        >>> p = R.from_quat([[0, 0, 1, 1], [1, 0, 0, 1]])
        >>> q = R.from_rotvec([[np.pi/4, 0, 0], [-np.pi/4, 0, np.pi/4]])
        >>> p.as_quat()
        array([[0.        , 0.        , 0.70710678, 0.70710678],
               [0.70710678, 0.        , 0.        , 0.70710678]])
        >>> q.as_quat()
        array([[ 0.38268343,  0.        ,  0.        ,  0.92387953],
               [-0.37282173,  0.        ,  0.37282173,  0.84971049]])
        >>> r = p * q
        >>> r.as_quat()
        array([[ 0.27059805,  0.27059805,  0.65328148,  0.65328148],
               [ 0.33721128, -0.26362477,  0.26362477,  0.86446082]])

        """
        if not _broadcastable(self._quat.shape, other._quat.shape):
            raise ValueError(
                "Expected equal number of rotations in both or a single "
                f"rotation in either object, got {self._quat.shape[:-1]} rotations in "
                f"first and {other._quat.shape[:-1]} rotations in second object."
            )
        quat = self._backend.compose_quat(self._quat, other._quat)
        if self._single:
            quat = quat[0]
        return Rotation(quat, normalize=True, copy=False)

    def __pow__(self, n: float, modulus: None = None) -> Rotation:
        """Compose this rotation with itself `n` times.

        Composition of a rotation ``p`` with itself can be extended to
        non-integer ``n`` by considering the power ``n`` to be a scale factor
        applied to the angle of rotation about the rotation's fixed axis. The
        expression ``q = p ** n`` can also be expressed as
        ``q = Rotation.from_rotvec(n * p.as_rotvec())``.

        If ``n`` is negative, then the rotation is inverted before the power
        is applied. In other words, ``p ** -abs(n) == p.inv() ** abs(n)``.

        Parameters
        ----------
        n : float
            The number of times to compose the rotation with itself.
        modulus : None
            This overridden argument is not applicable to Rotations and must be
            ``None``.

        Returns
        -------
        power : `Rotation` instance
            If the input Rotation ``p`` contains ``N`` multiple rotations, then
            the output will contain ``N`` rotations where the ``i`` th rotation
            is equal to ``p[i] ** n``

        Notes
        -----
        For example, a power of 2 will double the angle of rotation, and a
        power of 0.5 will halve the angle. There are three notable cases: if
        ``n == 1`` then the original rotation is returned, if ``n == 0``
        then the identity rotation is returned, and if ``n == -1`` then
        ``p.inv()`` is returned.

        Note that fractional powers ``n`` which effectively take a root of
        rotation, do so using the shortest path smallest representation of that
        angle (the principal root). This means that powers of ``n`` and ``1/n``
        are not necessarily inverses of each other. For example, a 0.5 power of
        a +240 degree rotation will be calculated as the 0.5 power of a -120
        degree rotation, with the result being a rotation of -60 rather than
        +120 degrees.

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R

        Raising a rotation to a power:

        >>> p = R.from_rotvec([1, 0, 0])
        >>> q = p ** 2
        >>> q.as_rotvec()
        array([2., 0., 0.])
        >>> r = p ** 0.5
        >>> r.as_rotvec()
        array([0.5, 0., 0.])

        Inverse powers do not necessarily cancel out:

        >>> p = R.from_rotvec([0, 0, 120], degrees=True)
        >>> ((p ** 2) ** 0.5).as_rotvec(degrees=True)
        array([  -0.,   -0., -60.])

        """
        if modulus is not None:
            raise NotImplementedError("modulus not supported")
        quat = self._backend.pow(self._quat, n)
        if self._single:
            quat = quat[0]
        return Rotation(quat, normalize=False, copy=False)

    def _to_array(self, quat: ArrayLike) -> Array:
        """Convert the quaternion to an array.

        The return array dtype follows the following rules:
        - If quat is an ArrayLike or NumPy array, we always promote to float64
        - If quat is an Array from frameworks other than NumPy, we preserve the dtype if it is
          float32. Otherwise, we promote to the result type of combining float32 and float64

        The first rule is required by the cython backend signatures that expect cython.double views.
        The second rule is necessary to promote non-floating arrays to the correct type in
        frameworks that may not support double precision (e.g. jax by default).
        """
        xp = array_namespace(quat)
        quat = xp.asarray(quat)
        # TODO: Remove this once we properly support broadcasting
        if quat.ndim not in (1, 2) or quat.shape[-1] != 4 or quat.shape[0] == 0:
            raise ValueError(f"Expected `quat` to have shape (N, 4), got {quat.shape}.")

        # TODO: Do we always want to promote to float64 for NumPy? This is consistent with the old
        # implementation, but it might make more sense to preserve float32 if passed in by the user.
        # This would make the behavior more consistent with the Array API backend, but requires
        # changes in the cython backend.
        if is_numpy(xp):
            dtype = xp.float64
        else:
            dtype = array_api_backend.atleast_f32(quat)
        return xp.asarray(quat, dtype=dtype)

    def __repr__(self):
        m = f"{np.asarray(self.as_matrix())!r}".splitlines()
        # bump indent (+21 characters)
        m[1:] = [" " * 21 + m[i] for i in range(1, len(m))]
        return "Rotation.from_matrix(" + "\n".join(m) + ")"


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(
        sub_package="spatial.transform",
        module="rotation",
        private_modules=["_rotation"],
        all=__all__,
        attribute=name,
    )


def _broadcastable(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> bool:
    """Check if two shapes are broadcastable."""
    return all(
        (m == n) or (m == 1) or (n == 1) for m, n in zip(shape_a[::-1], shape_b[::-1])
    )


# Register as pytree node for JAX if available to make Rotation compatible as input argument and
# return type for jit-compiled functions
try:
    from jax.tree_util import register_pytree_node

    def rot_unflatten(_, c):
        # Optimization: We do not want to call __init__ here because it would perform normalizations
        # twice. More importantly, it would call the non-jitted Array API backend and therefore
        # incur a significant performance hit
        r = Rotation.__new__(Rotation)
        # Someone could have registered a different backend for jax, so we attempt to fetch the
        # updated backend here. If not, we fall back to the Array API backend.
        r._backend = backend_registry.get(array_namespace(c[0]), array_api_backend)
        r._quat = c[0]
        # We set _single to False for jax because the Array API backend supports broadcasting by
        # default and hence returns the correct shape without the _single workaround
        r._single = False
        return r

    register_pytree_node(Rotation, lambda v: ((v._quat,), None), rot_unflatten)
except ImportError:
    pass
