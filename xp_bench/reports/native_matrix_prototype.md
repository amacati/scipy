# Native array-API matrix backend: prototype and gains vs converting to numpy

`pdist`/`cdist` run compiled C/C++ kernels, so array-API inputs are copied to NumPy and back
(`pdist` via `xpx.lazy_apply(as_numpy=True)`; `cdist` via a bare `np.asarray`, which is numpy-only
and **crashes on a CUDA tensor**). This prototype implements a native, device-resident euclidean
`cdist` by broadcasting and measures it against that round-trip on every backend.

```python
def cdist_native(XA, XB):
    xp = array_namespace(XA, XB)
    diff = XA[:, None, :] - XB[None, :, :]      # (mA, mB, n)
    return xp.sqrt(xp.sum(diff * diff, axis=-1)) # (mA, mB)
```

Method: min-of-10, features = 10, euclidean. "convert to numpy" = copy both inputs to NumPy,
run `distance.cdist` (C), copy the result back to the device.

## Native speedup over convert-to-numpy (min-of-10)

| config | m=100 | m=300 | m=1000 | m=3000 |
|---|--:|--:|--:|--:|
| numpy cpu | 0.12× | 0.03× | 0.03× | 0.06× |
| torch cpu | 0.60× | 2.82× | 0.11× | 0.22× |
| torch **gpu** | 3.5× | 13.4× | 12.7× | **18.6×** |
| jax cpu | 0.79× | 2.19× | 0.56× | 1.22× |
| jax **gpu** | 0.42× | 1.65× | 11.1× | **34.8×** |
| cupy **gpu** | 1.6× | 1.8× | 1.6× | 2.6× |

![native vs bridge](native_cdist.png)

## Reading

- **GPU backends: native is a decisive win** (torch 3.5–18.6×, jax up to 34.8×, cupy 1.6–2.6×),
  growing with m. Two effects compound: the bridge pays a device→host→device copy on every call,
  and the pairwise reduction itself parallelizes on the GPU. This is the case for a native backend.
- **NumPy: native loses 8–30×.** The C `cdist` is a tight blocked kernel; the broadcasting form
  materializes the full `(m, m, n)` difference and is memory-bound. Keep the C path on NumPy.
- **CPU non-numpy: roughly break-even** (native ahead at small m, behind at large m where the
  `(m, m, n)` intermediate dominates over C).

## Recommendation

A `select_backend`-style split, mirroring the rotation module: **NumPy → the compiled C kernels;
torch/jax/cupy → a native array-API implementation.** The native path also removes the current
hard failure of `cdist` on GPU tensors. Caveats for a production version:

- The naive `(m, m, n)` intermediate is O(m²·n) memory — fine on GPU up to a few thousand points,
  but a blocked/tiled form (or the `‖a‖² + ‖b‖² − 2a·b` expansion for euclidean) is needed for
  large m to bound memory.
- Only euclidean is prototyped here; each metric needs its own vectorized kernel, but they all
  follow the same broadcast-and-reduce shape already used by the T2 scalar metrics.
- `pdist` is the same computation restricted to the upper triangle; the condensed extraction is a
  cheap post-step (a boolean mask) relative to the O(m²·n) distance computation.
