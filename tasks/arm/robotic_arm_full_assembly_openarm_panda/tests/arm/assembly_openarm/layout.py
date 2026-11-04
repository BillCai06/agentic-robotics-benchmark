"""The episode's bench layout around the bench plan: every unit (with its riser blocks) turned about the vertical and
moved on the bench by a random amount, in the stable resting orientation of the plan.

`jitter_layout(kit, plan, seed, ...)` -> {"parts": {unit: (pos, quat)}, "risers": {riser: (pos, quat, span)}, "draw"}:
per unit a turn of up to `yaw_deg` about the vertical through the centre of its footprint and a shift of up to `xy_m`;
a unit on a riser takes its riser with it. The footprints (the collision pieces' convex outlines on a 2 mm grid, the
blocks' rectangles) must stay `gap_m` apart, on the bench (`margin_m` from its edges) and off the keep-outs (the
Pandas' bases, the fixture's plate). A failed draw redraws the WHOLE layout (keeping one unit's planned pose left it
inside a neighbour once, in another task); the amplitudes halve every 50 tries; the plan itself is the last resort.
"""
import math
import os

import numpy as np

__all__ = ["jitter_layout", "unit_outlines", "KEEPOUT_HALF_M"]

GRID_M = 0.002
KEEPOUT_HALF_M = 0.13            # a square around each Panda's base axis (its base is 0.21 m across)


def _quat_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _mat_quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def unit_outlines(kit):
    """{unit: [convex piece vertices (n x 3, in the unit root's frame)]}: the collision pieces of the unit's rigid bodies and
    hinged children at the kit's design angles (body frames are the arm frame: one transform per unit)."""
    import trimesh
    out = {}
    root_T = {}
    for bn, bb in kit["bodies"].items():
        u = bb["unit"]
        if u not in kit["loose"]:
            continue
        T = np.asarray(bb["T_design_arm"], float)
        if bn == u:
            root_T[u] = T
    for bn, bb in kit["bodies"].items():
        u = bb["unit"]
        if u not in kit["loose"]:
            continue
        T_rel = np.linalg.inv(root_T[u]) @ np.asarray(bb["T_design_arm"], float)
        for f in bb.get("collision", []):
            v = np.asarray(trimesh.load(os.path.join(kit["dir"], f), force="mesh").vertices, float)
            out.setdefault(u, []).append(v @ T_rel[:3, :3].T + T_rel[:3, 3])
    return out


def _fill(polys_xy, x0, y0, nx, ny):
    """boolean grid (nx x ny, cell (i, j) centred at x0 + i g, y0 + j g) of the union of convex polygons"""
    from scipy.spatial import ConvexHull
    mask = np.zeros((nx, ny), bool)
    for P in polys_xy:
        if len(P) < 3:
            continue
        try:
            h = ConvexHull(P)
        except Exception:  # noqa: BLE001 - a degenerate piece (a flat sliver): its bounding box
            lo, hi = P.min(0), P.max(0)
            i0, i1 = int((lo[0] - x0) / GRID_M), int((hi[0] - x0) / GRID_M) + 1
            j0, j1 = int((lo[1] - y0) / GRID_M), int((hi[1] - y0) / GRID_M) + 1
            mask[max(0, i0):min(nx, i1), max(0, j0):min(ny, j1)] = True
            continue
        lo, hi = P.min(0), P.max(0)
        i0, i1 = max(0, int((lo[0] - x0) / GRID_M)), min(nx, int((hi[0] - x0) / GRID_M) + 2)
        j0, j1 = max(0, int((lo[1] - y0) / GRID_M)), min(ny, int((hi[1] - y0) / GRID_M) + 2)
        if i0 >= i1 or j0 >= j1:
            continue
        gx, gy = np.meshgrid(x0 + GRID_M * np.arange(i0, i1), y0 + GRID_M * np.arange(j0, j1), indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        inside = np.all(pts @ h.equations[:, :2].T + h.equations[:, 2] <= 1e-9, axis=1)
        mask[i0:i1, j0:j1] |= inside.reshape(gx.shape)
    return mask


def _riser_rects(T_r, span, block_m):
    """the two blocks' footprints (4 x 2 corners each) of a riser at T_r"""
    out = []
    hx, hy = block_m[0] / 2, block_m[1] / 2
    for sx in (-0.5, 0.5):
        c = np.array([[sx * span - hx, -hy, 0], [sx * span + hx, -hy, 0], [sx * span + hx, hy, 0], [sx * span - hx, hy, 0]])
        out.append((c @ T_r[:3, :3].T + T_r[:3, 3])[:, :2])
    return out


def _planar(T, c, dxy, th):
    """T moved: turned by th about the vertical through c (world xy), then shifted by dxy"""
    ct, st = math.cos(th), math.sin(th)
    Rz = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    out = np.array(T, float).copy()
    out[:3, :3] = Rz @ T[:3, :3]
    p = T[:3, 3].copy()
    p[:2] = c + Rz[:2, :2] @ (p[:2] - c) + dxy
    out[:3, 3] = p
    return out


def jitter_layout(kit, plan, seed, xy_m=0.03, yaw_deg=30.0, gap_m=0.006, bench_xy=(1.6, 1.4), margin_m=0.02, keepouts=(),
                  block_m=(0.010, 0.04, 0.035), tries=300, outlines=None):
    """the episode's layout (see the module doc). `plan` = {"rest": {unit: [pos, quat]}, "risers": {riser: [pos, quat, span]}};
    `keepouts` = [(x0, x1, y0, y1)] world boxes no footprint may enter."""
    outlines = outlines if outlines is not None else unit_outlines(kit)
    rest = {u: (np.asarray(p, float), np.asarray(q, float)) for u, (p, q) in plan["rest"].items()}
    risers = {n: (np.asarray(v[0], float), np.asarray(v[1], float), float(v[2])) for n, v in (plan.get("risers") or {}).items()}
    riser_of = {n[len("riser_"):]: n for n in risers}
    units = [u for u in kit["loose"] if u in rest]
    T0 = {}
    for u in units:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = _quat_mat(rest[u][1]), rest[u][0]
        T0[u] = T
    R0 = {}
    for n, (p, q, _s) in risers.items():
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = _quat_mat(q), p
        R0[n] = T
    hx, hy = bench_xy[0] / 2 - margin_m, bench_xy[1] / 2 - margin_m
    x0, y0 = -bench_xy[0] / 2, -bench_xy[1] / 2
    nx, ny = int(bench_xy[0] / GRID_M) + 1, int(bench_xy[1] / GRID_M) + 1
    gap_px = max(1, int(round(gap_m / GRID_M)))
    from scipy.ndimage import binary_dilation
    st = np.ones((2 * gap_px + 1, 2 * gap_px + 1), bool)
    keep = np.zeros((nx, ny), bool)
    for (a0, a1, b0, b1) in keepouts:
        keep[max(0, int((a0 - x0) / GRID_M)):max(0, int((a1 - x0) / GRID_M)), max(0, int((b0 - y0) / GRID_M)):max(0, int((b1 - y0) / GRID_M))] = True
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, 0x0A11])

    def footprint(u, T, T_r):
        polys = [(v @ T[:3, :3].T + T[:3, 3])[:, :2] for v in outlines[u]]
        if T_r is not None:
            polys += _riser_rects(T_r, risers[riser_of[u]][2], block_m)
        return polys

    def check(Ts, Trs):
        occ = np.zeros((nx, ny), np.int16)
        for u in units:
            polys = footprint(u, Ts[u], Trs.get(u))
            allp = np.vstack(polys)
            if np.any(np.abs(allp[:, 0]) > hx) or np.any(np.abs(allp[:, 1]) > hy):
                return "off the bench: %s" % u
            m = _fill(polys, x0, y0, nx, ny)
            if np.any(m & keep):
                return "in a keep-out: %s" % u
            occ += binary_dilation(m, structure=st) if gap_px else m
        if np.any(occ > 1):
            return "overlap"
        return None

    amp, why, n_try = 1.0, None, 0
    for k in range(int(tries)):
        if k and k % 50 == 0:
            amp *= 0.5
        n_try = k + 1
        Ts, Trs, draw = {}, {}, {}
        for u in units:
            pts = np.vstack([(v @ T0[u][:3, :3].T + T0[u][:3, 3])[:, :2] for v in outlines[u]])
            c = 0.5 * (pts.min(0) + pts.max(0))
            th = float(rng.uniform(-1.0, 1.0)) * math.radians(yaw_deg) * amp
            ang = float(rng.uniform(0.0, 2 * math.pi))
            rad = float(np.sqrt(rng.uniform(0.0, 1.0))) * xy_m * amp
            dxy = np.array([math.cos(ang), math.sin(ang)]) * rad
            Ts[u] = _planar(T0[u], c, dxy, th)
            if u in riser_of:
                Trs[u] = _planar(R0[riser_of[u]], c, dxy, th)
            draw[u] = {"yaw_deg": round(math.degrees(th), 2), "shift_mm": [round(1000 * float(v), 1) for v in dxy]}
        why = check(Ts, Trs) if (xy_m > 0 or yaw_deg > 0) else None
        if why is None:
            break
    else:
        Ts, Trs = dict(T0), {u: R0[riser_of[u]] for u in units if u in riser_of}
        draw = {"fallback": "the bench plan (no jittered layout in %d tries: %s)" % (tries, why)}
    parts = {u: (Ts[u][:3, 3].tolist(), _mat_quat(Ts[u][:3, :3]).tolist()) for u in units}
    rs = {riser_of[u]: (T[:3, 3].tolist(), _mat_quat(T[:3, :3]).tolist(), risers[riser_of[u]][2]) for u, T in Trs.items()}
    return {"parts": parts, "risers": rs, "draw": draw, "tries": n_try, "amplitude": amp}
