# xp_bench

Cross-backend benchmarks for the scipy array API port. Every case is timed on numpy,
torch, jax and cupy, on cpu and gpu, across a sweep of sample sizes, and the raw
timings are stored so plots and reports can be regenerated without re-running.

## Quick start

Set the environment up once on a fresh checkout of any branch.

```bash
xp_bench/setup_env.sh
```

The script declares the `all-frameworks` pixi environment, installs it and prints a
readiness check. That environment is not upstreamable, so its manifest block is not
committed and disappears whenever `pixi.toml` is synced from main. Re-run the script
afterwards. It is idempotent.

The benchmarks import scipy, so they only mean anything against a build.

```bash
pixi run -e all-frameworks spin run --build-dir=build-all-frameworks \
    python xp_bench/bench.py run --module spatial/distance
```

`spin run` rebuilds before it executes, so the numbers always come from the working
tree you are standing on.

### Subcommands

| command | what it does |
| --- | --- |
| `run` | time cases and store the raw timings |
| `plot` | draw one png and svg per case under `plots/` |
| `report` | write `summary.md` and `benchmark.pdf` under `reports/` |
| `list` | show the registered suites and their cases |

Every subcommand takes `--module`, a mirror path prefix such as `spatial/distance`,
and `--fn`, a single case name. Both default to everything.

`run` additionally takes

| option | default | meaning |
| --- | --- | --- |
| `--xp` | all four | one of numpy, torch, jax, cupy |
| `--device` | both | cpu or gpu |
| `--low` | 0 | log10 of the smallest sample size |
| `--high` | 7 | log10 of the largest sample size |
| `--repeat` | 5 | samples collected per size |
| `--number` | 100 | calls per sample |
| `--variant` | `current` | result tree to write into |
| `--append` | off | add to the stored samples instead of replacing them |

`plot` and `report` take `--variant` as a list and default to `current baseline`.
Speedups in the report divide the later variants by the first one, so a number above
1 means the first variant is faster.

### A/B workflow

Measure the reference on main, measure the change on the feature branch, then compare.

```bash
git switch main
pixi run -e all-frameworks spin run --build-dir=build-all-frameworks \
    python xp_bench/bench.py run --module spatial/distance --variant baseline

git switch my-feature
pixi run -e all-frameworks spin run --build-dir=build-all-frameworks \
    python xp_bench/bench.py run --module spatial/distance --variant current

pixi run -e all-frameworks spin run --build-dir=build-all-frameworks \
    python xp_bench/bench.py plot --module spatial/distance
pixi run -e all-frameworks spin run --build-dir=build-all-frameworks \
    python xp_bench/bench.py report --module spatial/distance
```

`plot` and `report` read nothing but the stored json, so they can be re-run at any
time on any branch. Suite discovery still imports scipy, so keep the same invocation.

## Layout

Suites under `xp_bench/suites` mirror the scipy module tree, and everything they
produce mirrors it too.

```
suites/spatial/distance.py                                   the cases
results/spatial/distance/<variant>/<xp>/<device>/<case>.json  raw timings
plots/spatial/distance/<case>.{png,svg}                       figures
reports/spatial/distance/{summary.md,benchmark.pdf}           summaries
```

### Adding a suite

Create the module at the path that mirrors the scipy module and register each case.
Discovery is automatic and the mirror path is derived from the module path, so there is
nothing to declare anywhere else. Decorating takes the case name from the function name.

```python
from xp_bench import register, to_xp

@register
def my_case(xp, device, n_samples):
    data = None

    def setup():
        nonlocal data
        data = to_xp(xp, np.random.rand(n_samples), device)

    def test():
        return scipy_function(data)

    def jax_test():
        jax.block_until_ready(scipy_function(data))

    return setup, test, jax_test
```

Cases that differ only in which callable they time share one builder and bind the rest
with `partial`. Those pass the name explicitly, which is how the transform suites
register their methods and constructors.

```python
register("as_rotvec", partial(method_case, call=R.as_rotvec))
register("pow", partial(method_case, call=R.__pow__, args=(2.0,)))
```

## Methodology

Each of the points below is a way to produce plausible wrong numbers. Read them
before trusting or extending anything in `results/`.

### GPU synchronization

torch and cupy launch kernels asynchronously. A timer wrapped around a bare call
returns as soon as the kernel is queued, so it measures launch latency rather than
compute, and a GPU looks absurdly fast. `core.timed_call` wraps the timed callable
with `torch.cuda.synchronize` and `cupy.cuda.Device().synchronize` so the timer only
stops once the device is actually done. jax has no global synchronize, so its cases
call `jax.block_until_ready` on the result instead.

Setup is not synchronized. Asynchronous data creation can therefore bleed into the
first timed iteration of a repeat.

### JIT warmup

jax is timed on a jitted callable while every other backend is timed on the plain
Python method. The numbers answer "how fast is this backend at its best", not "how
fast is the same code path on every backend".

The jit wrapper is built inside `setup` and warmed there with `block_until_ready`, so
compilation never lands inside the timed region. This matters because `timeit` re-runs
setup once per repeat, which means jax recompiles once per repeat. The warmup is the
only thing keeping that cost out of the measurement.

Not every case is jitted. In the distance suite only the pair metrics are, and
`sokalsneath` is deliberately excluded because it has a data dependent guard. jax
numbers are therefore not uniformly jitted numbers, check the suite before comparing.

### The nonlocal contract

`timeit` re-runs `setup` once per repeat, so the inputs are rebuilt for every repeat.
Cases rebind their state with `nonlocal` and the timed closure reads the current
binding. A case that captured its inputs by value would silently measure the very
first dataset in every repeat while appearing to work.

### What is stored

Result files hold the raw per-repeat seconds per call, not a reduction, so the
consumer picks the statistic. `min` is the usual choice for a timing distribution
because it is the least contaminated by scheduler noise. The plots draw mean with std
error bars and the report gives both min and mean, along with their speedups.

### Process to process variation

The `repeat` samples inside one run share a process, a warmed allocator and one jax
compilation cache, so their spread understates the real uncertainty. Re-running the
same sweep in a fresh process moves the result by far more than the within-run spread
suggests, and on jax gpu it dominates everything else.

Measured on `spatial/transform/rotation` `as_rotvec`, jax gpu at n=1000, 20 fresh
processes: mean 33.1 us, std 2.6 us, ranging 26.2 to 35.7 us. That is a process to
process CV near 9 percent, against a within-run spread an order of magnitude tighter.
numpy cpu sits near 0.7 percent, torch and cupy near 1 percent.

Two consequences. A single run cannot resolve a small change on jax gpu, and a
significance test computed inside one process will call a chance difference real. At a
CV of 9 percent, the number of runs per side needed for 80 percent power is roughly
`15.7 * CV^2 / effect^2`:

| effect to detect | runs per side |
| --- | --- |
| 30% | 1 |
| 20% | 3 |
| 10% | 12 |
| 5% | 50 |
| 2% | 312 |

Use `--append` to gather them. Each invocation adds its samples to the stored list
instead of replacing it, so the file accumulates across processes and the error bars
and significance tests then reflect the variation that actually matters.

```bash
for i in $(seq 20); do
    pixi run -e all-frameworks spin run --build-dir=build-all-frameworks \
        python xp_bench/bench.py run --module spatial/transform/rotation \
        --fn as_rotvec --xp jax --device gpu --low 3 --high 3 --append
done
```

### The timeout abort

If a single call exceeds `TIMEOUT / (repeat * number)`, the case returns nothing and
the sweep breaks out of the remaining, larger sizes for that framework and device.
Missing large sizes in the data mean "too slow to measure", not "not run".

### Skips

Out of memory and unsupported backend operations also break the size sweep for that
framework and device. This is why some cases have no data at all on some backends.

### Dtype is not controlled

This is the biggest interpretation trap in the suite. Inputs are built as float64
numpy arrays. numpy, torch and cupy carry them through unchanged, jax downcasts them
to float32 because jax x64 is off. A jax bar next to a numpy bar is single precision
next to double precision. Cross-framework speedup claims taken from these numbers are
comparing different precisions and are not valid on their own.

The A/B comparison within one framework is unaffected, since both variants build the
same inputs on the same backend.

### What "sample size" means

The stored key is the requested size, and it means something different per case.

| case group | sample size |
| --- | --- |
| scalar pair metrics, `mahalanobis`, `seuclidean` | vector length |
| `pdist`, `cdist` | observation count, at a fixed 10 features |
| `is_valid_dm`, `num_obs_dm` | side of the square distance matrix |
| `squareform`, `is_valid_y`, `num_obs_y` | requested condensed length |

A requested condensed length is snapped to the nearest valid binomial length, since
only those correspond to a real number of observations. The file records the requested
size, not the realized one.

### Machine hygiene

Never run competing CPU work while a benchmark runs. Compiles, test suites and other
benchmarks all corrupt the timings, and nothing in the data records that it happened.

## Known gaps

- Setup is not synchronized, so asynchronous data creation can leak into the first
  timed iteration.
- Only jax gets a real warmup. Every other backend gets the single timeout probe call
  and nothing more, so first call cuBLAS and autotune costs can land inside the
  measurement and can spuriously trip the timeout abort.
- Result files are merged rather than rewritten. Stale sizes from an earlier sweep
  survive alongside fresh ones, and nothing records which scipy build produced any
  given number.
