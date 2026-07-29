"""Case registry. Suites under xp_bench/suites mirror the scipy module tree."""

import importlib
from functools import partial
from pathlib import Path

SUITES = Path(__file__).parent / "suites"

_cases = {}


def mirror_of(module):
    """Scipy module path a suite mirrors, so xp_bench.suites.spatial.distance is spatial/distance."""
    return module.split(".", 2)[2].replace(".", "/")


def register(name, builder=None):
    """Register a case builder under the scipy module its suite mirrors.

    Takes the name and the builder, `register("as_rotvec", partial(method_case, ...))`,
    or decorates a function, `@register`, which takes the case name from the function.

    Args:
        name: Case name, which is also the name of its result file. The decorated
            function itself when used as a decorator.
        builder: Callable taking (xp, device, n_samples) and returning the
            (setup, test, jax_test) triple that `core.time_case` drives. A partial is
            resolved to the module of the function it wraps.

    Returns:
        The builder, so the decorator form leaves the function bound in its module.
    """
    if builder is None:
        builder, name = name, name.__name__
    module = builder.func if isinstance(builder, partial) else builder
    _cases.setdefault(mirror_of(module.__module__), {})[name] = builder
    return builder


def discover():
    """Import every suite and return {mirror: {case name: builder}}."""
    for path in sorted(SUITES.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        module = path.relative_to(SUITES.parent.parent).with_suffix("")
        importlib.import_module(".".join(module.parts))
    return _cases
