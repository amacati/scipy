# T1 distance functions: array API port

Branch `xp-spatial-t1` (off `xp-spatial-tests`). T1 covers the four validation/counting
helpers: `is_valid_dm`, `num_obs_dm`, `is_valid_y`, `num_obs_y`.

## What changed

- **`is_valid_dm`** — the only T1 function with real array work. Symmetry and zero-diagonal
  checks now go through the array namespace: `xp.all(D == D.T)` and
  `xp.all(xp.linalg.diagonal(D) == 0)`, replacing `(D == D.T).all()` and the
  `D[range(n), range(n)]` fancy-index diagonal.
- **`num_obs_dm`** — `np.asarray` → `_asarray`; it delegates to `is_valid_dm`.
- **`is_valid_y`, `num_obs_y`** — already backend-agnostic (they only touch `.shape` and do
  scalar `ceil`/`sqrt`), so no code change was needed. They are exercised on all backends now.

Test suite: the module-level `np_only` marker was removed and re-applied per class/function
to the tiers not yet ported. The four T1 classes run on every backend. Two `*_multi_matrix`
tests stay numpy-only because they build their input via `pdist`/`squareform`, which are not
yet ported.

**Correctness:** full `test_distance.py` on the all-frameworks env: **576 passed, 1679 skipped, 0 failed**
(numpy, torch cpu/gpu, jax cpu/gpu, cupy gpu, array_api_strict).

## Methodology

Same-build A/B on the `spin` `build-install` tree (MKL, `mkl-dynamic-lp64-seq`). Because the
T1 change is pure Python, baseline and ported builds are byte-identical except for
`distance.py`, which is swapped in place — so numpy before/after differs only by the ported
code, nothing else. Timings are min-of-10 (`repeat=10, number=100`). The DM validators build a
`k×k` matrix, so their sweep is capped at `k=10⁴` (they are O(k²) and abort or OOM beyond that);
the O(1) `y` validators run to length 10⁷.

## Results

### numpy: ported vs baseline (min of 10)

| function      | size | baseline | ported | ratio |
|---------------|------|----------|--------|-------|
| `is_valid_dm` | k=1000 | 642 µs | 582 µs | **1.10× faster** |
| `num_obs_dm`  | k=1000 | 781 µs | 708 µs | **1.10× faster** |
| `is_valid_y`  | n=10⁷ | 1.66 µs | 1.59 µs | 1.05× (flat) |
| `num_obs_y`   | n=10⁷ | 2.99 µs | 2.94 µs | 1.02× (flat) |

The DM validators are slightly faster on numpy: `xp.linalg.diagonal` is a strided view, where
the old `D[range(n), range(n)]` allocated an index array and a gathered copy. The `y` validators
are unchanged (they never touched an array kernel).

### Cross-framework scaling — `is_valid_dm` (min µs, by matrix dim k)

| k | numpy cpu | torch cpu | torch gpu | jax cpu | jax gpu | cupy gpu |
|--:|--:|--:|--:|--:|--:|--:|
| 1     | 4.2 | 7.6 | 33.6 | 43.0 | 251 | 70.5 |
| 100   | 8.0 | 15.5 | 34.7 | 75.4 | 221 | 71.5 |
| 1000  | 582 | 77.8 | **52.6** | 403 | 239 | 98.6 |
| 10000 | — (abort) | 49961 | 3904 | — (abort) | **2589** | 5512 |

Scaling is shown in the four-subplot-per-function figures (`distance_plots/is_valid_dm.png`,
`num_obs_dm.png`), one panel per framework, matching the rotation benchmark layout.

## Interpretation

- **New capability is the headline.** `is_valid_dm`/`num_obs_dm` now run on torch/jax/cupy and
  on GPU, which was impossible before. At `k=10⁴` numpy aborts (an O(k²) host symmetry compare)
  while jax-gpu finishes in 2.6 ms.
- **Clear crossover at k≈1000.** For small matrices numpy wins on dispatch overhead (single-µs vs
  tens–hundreds of µs of framework/device latency). GPUs overtake once the matrix is large enough
  to amortize launch cost: at k=1000 torch-gpu is ~11× faster than numpy, and it is the only tier
  that scales to k=10⁴.
- **torch CPU is the outlier** — pathologically slow at large k (50 ms at k=10⁴), so torch on CPU
  is not a good target for these checks; its GPU path is fine.
- **The `y` validators do not benefit from non-numpy backends** and never will: they perform no
  array computation, only shape arithmetic. Off-numpy they carry a small constant overhead
  (2–12 µs, jax the slowest at ~4× numpy) with nothing to accelerate. Correct to support, not
  worth steering onto a device.

## Net

numpy is not regressed (DM validators are marginally faster, y validators flat), and the DM
validators gain a real GPU path with a ~k=1000 crossover. T1 is a clean, low-risk win.
