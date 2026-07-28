# T2 distance functions: array API port of the scalar metrics

Branch `xp-spatial-t2` (stacked on `xp-spatial-t1`). T2 ports the 17 pairwise scalar metrics
plus `mahalanobis`, `seuclidean`, and `jensenshannon`, and the shared helpers
`_validate_vector`, `_validate_weights`, `_nbool_correspond_all/_ft_tf`.

## Design

- **Return type.** Metrics return the raw array-API expression. On NumPy that is an
  `np.float64` scalar (so `isinstance(x, float)` holds and doctests still print a scalar); on
  torch/jax/cupy it is a 0-d array of that backend. No unconditional `float()` — that would
  force a host sync and break `jax.jit`. `cosine`/`correlation` follow the same rule (they are
  `float` on NumPy, arrays elsewhere).
- **Guarded division, not branches.** All-zero / 0-0 cases (`jaccard`, `yule`, `canberra`, …)
  use the `x / (d + (d == 0))` idiom instead of a Python `if`, keeping the metrics jit-able.
  `sokalsneath` keeps its documented ValueError for entirely-false input and is the one metric
  that is not jit-able.
- **`jensenshannon`** inlines `scipy.special.rel_entr` (dropping that dependency) with a
  warning-free, fully faithful implementation (0 at x=0, +inf for x<0 or y≤0).
- **String inputs** (`hamming` on byte strings) have no array-API namespace; `_validate_vector`
  falls back to NumPy for them.

## Correctness

- Full `test_distance.py`: **596 passed, 1659 skipped, 0 failed** across numpy, torch cpu/gpu,
  jax cpu/gpu, cupy gpu, array_api_strict.
- Direct cross-backend check of all 19 metrics: every metric agrees with NumPy to float
  precision on torch/jax/cupy (jax's ~1e-7 gaps are its float32 default).
- `jax.jit` verified on euclidean, cosine, cityblock, jaccard, canberra.
- A real bug was surfaced by cross-backend testing: integer input reaching
  `xp.linalg.vector_norm` fails on torch (NumPy promoted silently); fixed by casting the
  difference to float in `minkowski`.

The weighted-invariance tests use a NumPy-coupled harness (`_weight_checked`/`_chk_weights`,
built on `np.append`/`np.array`), so those methods stay `np_only`; the direct-call metric tests
(`minkowski`, `correlation`, `mahalanobis`) run on every backend.

## Benchmarks

Same-build A/B on `build-install`, `distance.py` swapped between base and T2 (pure-Python
change, identical compiled scipy). min-of-10 (`repeat=10, number=100`). Metrics are benchmarked
as one 1-D pair scaling vector length. Baseline is the pre-port NumPy path (`numpy_baseline`,
dashed in the plots). Figures are four-subplot-per-function, one panel per framework, matching
the rotation benchmark.

### NumPy: T2 vs pre-port (min-of-10, ratio = base / T2, >1 means T2 faster)

| metric group | small N (≤1e5) | N = 1e6 | N = 1e7 |
|---|---|---|---|
| float (euclidean, sqeuclidean, minkowski, cosine, correlation, seuclidean) | ~1.0× | 1.0–1.2× | ~1.05× |
| **chebyshev** | ~0.35× (dispatch) | **46×** | **4.1×** |
| bool (jaccard, yule, dice, rogerstanimoto, russellrao, sokalsneath) + hamming | 0.6–0.8× | ~0.95× | **0.06–0.17×** |

Three distinct behaviours:

- **Float metrics: essentially free.** The `array_api_compat.numpy` wrappers add nothing
  measurable; a few metrics are marginally faster.
- **chebyshev: a large win.** The original computed `max(abs(u - v))` with the Python builtin
  `max`, which iterates the array element-by-element in Python (O(n) interpreter loop). `xp.max`
  is a vectorized reduction: 46× faster at N=1e6, converging to 4× at 1e7 where both are
  memory-bound.
- **Boolean metrics: a real regression at large N.** Two costs: a constant ~1.3 µs of
  dispatch overhead (visible only at small N), and — the important one — `xp.astype(u & v,
  int64)` / `xp.astype(u_ne_v, float64)` materializes a full-size integer/float array before the
  reduction, where NumPy sums booleans in place. At N=1e7 this makes russellrao 15× and hamming
  6× slower. **This is the clearest optimization target**: sum the boolean mask directly instead
  of casting to int64.

### Cross-framework (speedup vs NumPy at N = 1e7, min-of-10)

| metric | torch gpu | jax cpu | jax gpu | cupy gpu |
|---|--:|--:|--:|--:|
| euclidean | 4672× | 9.6× | 564× | — |
| minkowski | 6052× | 12× | 770× | 2365× |
| chebyshev | 4115× | 9.5× | 632× | 3141× |
| braycurtis | 4387× | 9.4× | 1295× | 3522× |
| jaccard | 406× | 6.5× | 2675× | 306× |
| cosine | 59× | 0.21× | 41× | 2.4× |

As with T1 there is a **crossover**: at small vectors NumPy wins on dispatch latency (single µs
vs tens–hundreds of µs of device overhead); GPUs overtake once the vector is large enough to
amortize launch cost. The four-subplot figures show this per metric — e.g. torch CPU degrades
badly at large N (per-element dispatch) while torch GPU stays flat. `cosine`/`correlation` gain
least (jax CPU is even slower) because their reductions are latency-bound, not throughput-bound.
cupy aborts on some 1-D metrics past N≈100 (per-call host-sync overhead exceeds the timeout
guard), which is why some cupy series are short.

## Net

All 19 metrics run and jit on every backend, numerically matching NumPy. On NumPy the float
metrics are free, chebyshev is much faster, and the boolean metrics carry a dtype-materialization
penalty at large N that is worth optimizing before merge. GPU backends give 1–3 orders of
magnitude at large vector length, with the usual small-N crossover.
