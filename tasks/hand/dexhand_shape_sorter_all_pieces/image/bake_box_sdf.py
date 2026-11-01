"""Re-bake the shape_box body and lid collision SDFs at 0.5 mm from the PUBLIC collision meshes (SuperDex bake_sdf).

The public asset ships a ~2.5 mm SDF grid for the box body; at that resolution the rectangle cannot pass its hole
(measured). The finer grid only changes the collision representation of the same public mesh.
Usage: python bake_box_sdf.py <out_dir>   (needs SUPERDEX_ASSETS_PATH; takes ~4 min on one core)
"""
import os
import sys

os.environ.setdefault("SUPERDEX_PRECISION", "double")
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import superdex.physics as sp  # noqa: E402

RES = 0.0005


def bake(src, dst):
    with h5py.File(src, "r") as f:
        v = f["mesh/coordinates"][:].astype(np.float64)
        t = f["mesh/connectivity"][:].astype(np.int64)
    obj = dst + ".tmp.obj"
    with open(obj, "w") as fh:
        for x in v:
            fh.write("v %r %r %r\n" % (float(x[0]), float(x[1]), float(x[2])))
        for tri in t:
            fh.write("f %d %d %d\n" % (tri[0] + 1, tri[1] + 1, tri[2] + 1))
    data = sp.model.load_from_file(obj)
    p = sp.GridSdfParams()
    p.resolution_mode = sp.GridSdfResolutionMode.EXPLICIT
    p.resolution_delta = [RES, RES, RES]
    p.boundary_padding_dist = 0.005
    sp.model.bake_sdf(data, p)
    sp.model.save_to_file(data, dst, sp.FileFormat.H5)
    os.remove(obj)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sp.initialize(0)
    col = os.path.join(os.environ["SUPERDEX_ASSETS_PATH"], "prefabs", "shape_box", "collision")
    for n in ("shape_box_body", "shape_box_lid"):
        bake(os.path.join(col, n + ".mochi.h5"), os.path.join(out, n + "_fine.mochi.h5"))
        print("baked", n, flush=True)


if __name__ == "__main__":
    main()
