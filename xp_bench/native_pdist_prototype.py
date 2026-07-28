"""Prototype: native array-API cdist vs the convert-to-numpy path.

`distance.cdist`/`pdist` run the compiled C kernels, so array-API inputs must be copied to
NumPy and back. This prototype implements a native, device-resident euclidean cdist via
broadcasting and measures it against that round-trip on each backend.
"""
import os
os.environ["SCIPY_ARRAY_API"] = "1"
from pathlib import Path
import json
import timeit
import numpy as np
import torch
import jax
import jax.numpy as jnp
import cupy
from scipy.spatial import distance
from scipy._lib._array_api import array_namespace

N_FEATURES = 10
TIMEOUT = 60


def cdist_native(XA, XB):
    """Device-resident euclidean cdist via broadcasting. Returns (mA, mB)."""
    xp = array_namespace(XA, XB)
    diff = XA[:, None, :] - XB[None, :, :]
    return xp.sqrt(xp.sum(diff * diff, axis=-1))


def to_xp(xp_name, a, device):
    if xp_name == "numpy":
        return a
    if xp_name == "torch":
        return torch.asarray(a, device="cuda" if device == "gpu" else "cpu")
    if xp_name == "jax":
        return jax.device_put(jnp.asarray(a), jax.devices(device)[0])
    if xp_name == "cupy":
        return cupy.asarray(a)


def to_numpy(a, xp_name):
    if xp_name == "numpy":
        return a
    if xp_name == "cupy":
        return cupy.asnumpy(a)
    if xp_name == "torch":
        return a.detach().cpu().numpy()
    return np.asarray(a)  # jax handles device->host


def sync(a, xp_name):
    if xp_name == "jax":
        jax.block_until_ready(a)
    elif xp_name == "torch" and a.is_cuda:
        torch.cuda.synchronize()
    elif xp_name == "cupy":
        cupy.cuda.Device().synchronize()
    return a


def bench(fn, repeat=10, number=5):
    fn()  # warmup
    t0 = timeit.default_timer()
    fn()
    if timeit.default_timer() - t0 > TIMEOUT / (repeat * number):
        return None
    return min(timeit.Timer(fn).repeat(repeat=repeat, number=number)) / number


def run():
    configs = [("numpy", "cpu"), ("torch", "cpu"), ("torch", "gpu"),
               ("jax", "cpu"), ("jax", "gpu"), ("cupy", "gpu")]
    sizes = [100, 300, 1000, 3000]
    out = {}
    for xp_name, device in configs:
        if xp_name == "cupy" and device == "cpu":
            continue
        out[f"{xp_name} {device}"] = {}
        for m in sizes:
            a = np.random.rand(m, N_FEATURES)
            b = np.random.rand(m, N_FEATURES)
            try:
                XA = to_xp(xp_name, a, device)
                XB = to_xp(xp_name, b, device)
            except Exception as e:
                print(f"{xp_name} {device}: setup failed {e!r}"[:80])
                break

            native = bench(lambda: sync(cdist_native(XA, XB), xp_name))

            def bridge():
                na = to_numpy(XA, xp_name)
                nb = to_numpy(XB, xp_name)
                r = distance.cdist(na, nb)
                return to_xp(xp_name, r, device)
            roundtrip = bench(lambda: sync(bridge(), xp_name))

            out[f"{xp_name} {device}"][m] = {"native": native, "bridge": roundtrip}
            n_s = f"{native*1e3:.3f}ms" if native else "abort"
            b_s = f"{roundtrip*1e3:.3f}ms" if roundtrip else "abort"
            speed = f"{roundtrip/native:.2f}x" if native and roundtrip else "-"
            print(f"{xp_name+' '+device:<12} m={m:<5} native={n_s:<10} bridge={b_s:<10} native speedup={speed}")
            if native is None:
                break

    path = Path(__file__).parent / "native_prototype_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    run()
