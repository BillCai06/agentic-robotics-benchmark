"""Numeric fingerprint of this machine against the reference (maintainer change, 2026-10-31).

On one machine the simulator is bit-reproducible. Across machines it was not: the same oracle plans passed 37/44 pieces
on the author's host and 26/44 on the maintainers' AMD Threadripper PRO 5975WX, each run bit-identical on its own host.
The cause is OpenBLAS, bundled with NumPy and SciPy, which picks its kernels by CPU. The wrist and quaternion maths do
many small dot products (numpy `@`, np.linalg.norm), and with another kernel family their last bit differs; the
open-loop contact replay then turns that into different pieces passing (replacing the Haswell kernels by the Prescott
ones changed 6 to 7 pieces in one scene).
What was ruled out:
- SuperDex: compiled for AVX2 + FMA with IEEE-exact operations only (no approximate reciprocals).
- NumPy's own SIMD dispatch: X86_V2 and X86_V3 gave the same result.
- The visible CPU count.
- The build-time SDF bake: same data on every bake.

The image therefore pins OpenBLAS to its Haswell kernels (OPENBLAS_CORETYPE=Haswell). Every CPU that SuperDex runs on
(x86-64 with AVX2 and FMA) can execute them. As a precaution it also turns NumPy's AVX-512 paths off
(NPY_DISABLE_CPU_FEATURES). With these settings the scores are the same on every such machine.

check() compares this machine with the reference values below and raises if anything differs: a machine that would
score differently refuses to grade instead of grading differently. The grader runs it before it simulates anything;
`python3 -m sorter.numerics` (from the folder that holds `sorter/`) runs it by hand and prints the fingerprint.
Both settings are read when NumPy is first imported. Set them in the environment before Python starts, as the image
does; setting them later in Python has no effect.
"""
from __future__ import annotations

import ctypes
import glob
import hashlib
import json
import math
import os

# Measured on the maintainers' host (AMD Ryzen Threadripper PRO 5975WX, Docker, this image) with the pinned settings;
# the same values came out of every run there, and the ale validate run that set the anchor used these numerics.
REFERENCE = {
    "openblas_core": "Haswell",
    "numpy_dispatch": ["X86_V3"],
    "blas_probe": "ed8072dac11ebe5b5ba4fbb73e31a9d87a9062e2d1ec2169295e68fdc67de19d",
    "libm_probe": "45037c555bbf0a9bc5d1adf362e81fca4ce3cdccf283f60dd2e9c212ed888f21",
    "sdf": {
        "shape_box_body_fine.mochi.h5": "0289a6e0df74c8b0058a1b507ae2c5a81efc88b537e9ca385da7007e766d5dec",
        "shape_box_lid_fine.mochi.h5": "737b64a49e16d00272e29bd99a48535a263497579bb77916bd09ccad09ee4e38",
    },
    "physics_probe": "10fc5872f3949959d5dd08f51f9f4d343c7ecdfcf75e2f5380c91f47cced2735",
}


def _openblas_cores():
    """Kernel family OpenBLAS uses, per bundled library (NumPy's and SciPy's)."""
    import numpy as np
    roots = [os.path.dirname(np.__file__) + ".libs"]
    try:
        import scipy  # noqa: F401
        roots.append(os.path.dirname(scipy.__file__) + ".libs")
    except ImportError:
        pass
    cores = {}
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "libscipy_openblas*.so*"))):
            lib = ctypes.CDLL(path)
            for sym in ("scipy_openblas_get_corename64_", "scipy_openblas_get_corename", "openblas_get_corename"):
                fn = getattr(lib, sym, None)
                if fn is not None:
                    fn.restype = ctypes.c_char_p
                    cores[os.path.basename(root)] = fn().decode()
                    break
    return cores


def _numpy_dispatch():
    from numpy._core._multiarray_umath import __cpu_dispatch__, __cpu_features__
    return [k for k in __cpu_dispatch__ if __cpu_features__.get(k)]


def _vec(n, salt):
    import numpy as np
    return np.array([((i * 2654435761 + salt * 40503) % 1000003) / 1000003.0 - 0.5 for i in range(n)], np.float64)


def _blas_probe():
    """Exact bits of small BLAS-backed products like the simulator's (each kernel family gives other bits)."""
    import numpy as np
    h = hashlib.sha256()
    for n in (3, 4, 7, 16, 20, 101, 1000):
        a, b = _vec(n, 1), _vec(n, 2)
        h.update(np.float64(a @ b).tobytes())
        h.update(np.float64(np.linalg.norm(a)).tobytes())
    for n in (3, 4, 20):
        A, B = _vec(n * n, 3).reshape(n, n), _vec(n * n, 4).reshape(n, n)
        h.update((A @ B).tobytes())
        h.update((A @ _vec(n, 5)).tobytes())
    return h.hexdigest()


def _libm_probe():
    """Exact bits of the libm functions the simulator's Python side calls (glibc picks an implementation by CPU)."""
    h = hashlib.sha256()
    for i in range(500):
        x = ((i * 7919) % 10007) / 10007.0 * 6.0 - 3.0
        c = max(-1.0, min(1.0, x / 3.0))
        for v in (math.sin(x), math.cos(x), math.tan(x), math.atan(x), math.atan2(x, 1.3), math.exp(x),
                  math.log(abs(x) + 1e-3), math.acos(c), math.asin(c), math.pow(abs(x) + 0.5, 1.7),
                  math.sqrt(abs(x)), math.hypot(x, 0.7)):
            h.update(float(v).hex().encode())
    return h.hexdigest()


def _sdf_digests():
    """sha256 of the data of the box SDFs baked at image build (HDF5 timestamps excluded)."""
    import h5py
    from .world import box_shape_file
    out = {}
    for name in ("shape_box_body", "shape_box_lid"):
        path = box_shape_file(name)
        parts = []

        def visit(key, obj):
            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                raw = data.tobytes() if hasattr(data, "tobytes") else repr(data).encode()
                parts.append(key + ":" + hashlib.sha256(raw).hexdigest())

        with h5py.File(path, "r") as fh:
            fh.visititems(visit)
        out[os.path.basename(path)] = hashlib.sha256(" ".join(parts).encode()).hexdigest()
    return out


PROBE_LAYOUT = {
    # Public and synthetic: no shipped or graded scene. Ten pieces dropped from 6 cm onto the table, the square
    # dropped onto the lid (the box SDF), nominal physics, no calibration error.
    "pieces": {name: {"pos": [-0.40 + 0.09 * (k % 4), -0.27 + 0.18 * (k // 4), 0.06], "quat": [0.0, 0.0, 0.0, 1.0]}
               for k, name in enumerate(["cross", "triangle", "trapezoid", "rectangle", "pentagon", "parallelogram",
                                         "octagon", "hexagon", "ellipse", "diamond"])},
}
PROBE_LAYOUT["pieces"]["square"] = {"pos": [0.10, 0.0, 0.20], "quat": [0.0, 0.0, 0.0, 1.0]}


def _physics_probe():
    """Final-state hash of a short public simulation: the drop settles for 1 s, then the wrist moves 5 cm down and
    2 cm forward from its park pose over 25 waypoints (1 s)."""
    import numpy as np
    from . import world as W
    w = W.World(json.loads(json.dumps(PROBE_LAYOUT)))
    w.settle(1.0)
    p0, q0, f0 = (np.copy(x) for x in w.cmd)
    for k in range(1, 26):
        w.step_waypoint((p0 + (k / 25.0) * np.array([0.02, 0.0, -0.05]), q0, f0))
    return w.state_digest()


def fingerprint(physics=True):
    fp = {
        "openblas_core": sorted(set(_openblas_cores().values())),
        "numpy_dispatch": _numpy_dispatch(),
        "blas_probe": _blas_probe(),
        "libm_probe": _libm_probe(),
        "sdf": _sdf_digests(),
    }
    if physics:
        fp["physics_probe"] = _physics_probe()
    return fp


def check(physics=True):
    """Raise RuntimeError if this machine does not reproduce the reference numerics."""
    fp = fingerprint(physics)
    bad = []
    if fp["openblas_core"] != [REFERENCE["openblas_core"]]:
        bad.append("OpenBLAS kernels %s, expected %s (set OPENBLAS_CORETYPE=Haswell before Python starts)"
                   % (fp["openblas_core"], REFERENCE["openblas_core"]))
    if fp["numpy_dispatch"] != REFERENCE["numpy_dispatch"]:
        bad.append("NumPy SIMD dispatch %s, expected %s (the CPU needs AVX2 and FMA; set NPY_DISABLE_CPU_FEATURES="
                   "'X86_V4 AVX512_ICL AVX512_SPR')" % (fp["numpy_dispatch"], REFERENCE["numpy_dispatch"]))
    for key in ("blas_probe", "libm_probe", "sdf") + (("physics_probe",) if physics else ()):
        if fp[key] != REFERENCE[key]:
            bad.append("%s differs from the reference" % key)
    if bad:
        raise RuntimeError("this machine does not reproduce the reference numerics, so its scores would not be "
                           "comparable: " + "; ".join(bad))
    return fp


if __name__ == "__main__":
    print(json.dumps(fingerprint(), indent=1))
    check()
    print("numerics match the reference")
