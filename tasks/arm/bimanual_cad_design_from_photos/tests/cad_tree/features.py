"""Feature probes of the tree grader (feature_fidelity): the GT folder's cut-out and spacer records (CONTRACTS 6) measured on
a submitted part after its rigid alignment onto the reference part (both in B at configuration Z, mm).

    cutout_item(agent_mesh, gt_mesh, feature, bands)   a through cut-out of a plate
    spacer_item(agent_mesh, gt_mesh, feature, bands)   a disc / ring fused to a driven part's mating face
    extent_errors(agent_mesh, gt_mesh)                 per-axis bounding-box extent errors

Cut-out rule (copied from chain/shared/arm/cad_photo/features.py, `cutout_item`, with the field names of CONTRACTS 6:
centre, normal, e1, e2, footprint, depth_range, and a numpy rasteriser in place of PIL's polygon fill). Both parts are
sliced by the plane through the cut's centre at mid-depth
(centre + normal x the middle of depth_range, which is read as signed offsets from the centre along the normal) and
rasterised at 0.1 mm. The reference region R = the reference section with the cut filled (its material + the
footprint): the submitted part's EMPTY region is E = R minus its material, so only the region where the reference part is
otherwise solid counts. The footprint is cross-correlated with E over +-6 mm; the cut-out is present iff the best overlap
reaches 0.6 of the footprint area; the connected empty component with the largest overlap gives the centre offset and
the area; item = present x mean(band(offset; cutout_offset_mm), band(|area error| / area; cutout_area_rel)).

Spacer rule (new; the analogue of the worked example's pocket floor): rays start 2 mm on the motor's side of the output
face (`centre`) on three circles, at 0.25 / 0.50 / 0.75 of the way across the ring from inner_radius_mm to radius_mm (a
disc without inner_radius_mm: 0.70 / 0.80 / 0.90 of radius_mm), 16 angles each, and run along `normal` (away from the motor);
the median distance to the first surface is where the driven part's material begins. Both the reference part and the
aligned submitted part are probed with the same rays, so a spacer the submission leaves out (or makes thinner) shows as
the floor error, band(spacer_mm).
"""

import math

import numpy as np

PX = 0.1
SEARCH_MM = 6.0
PRESENT_FRAC = 0.6
SPACER_BACK_MM = 2.0
SPACER_RADII = (0.70, 0.80, 0.90)       # fractions of the radius (a disc)
SPACER_RING = (0.25, 0.50, 0.75)        # fractions across the ring when the record gives inner_radius_mm
SPACER_ANGLES = 16
RAY_MISS_MM = 60.0


def _band(value, b):
    v = abs(float(value))
    if not math.isfinite(v):
        return 0.0
    if v <= b[0]:
        return 1.0
    if v >= b[1]:
        return 0.0
    return (b[1] - v) / (b[1] - b[0])


def _manifold(mesh):
    import manifold3d as m3d
    return m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32), tri_verts=np.asarray(mesh.faces, np.uint32)))


def _vec(feature, *keys):
    for k in keys:
        if feature.get(k) is not None:
            return np.asarray(feature[k], float)
    raise KeyError("feature record without %s" % " / ".join(keys))


def plane_frame(feature):
    """(probe centre, normal, e1, e2) of a cut-out record: the centre moved to mid-depth when depth_range is given."""
    c = _vec(feature, "centre", "centre_mm")
    d = _vec(feature, "normal")
    e1 = _vec(feature, "e1")
    e2 = _vec(feature, "e2")
    d, e1, e2 = d / np.linalg.norm(d), e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)
    dr = feature.get("depth_range", feature.get("depth_range_mm"))
    if dr is not None and len(dr) == 2:
        c = c + d * 0.5 * (float(dr[0]) + float(dr[1]))
    return c, d, e1, e2


def _loops_in_plane(mesh, frame):
    """Section loops (u, v) of `mesh` (trimesh) cut by the plane through c normal d; u = e1, v = e2."""
    c, d, e1, e2 = frame
    R = np.vstack([e1, e2, d])                     # rows: plane coordinates
    V = (np.asarray(mesh.vertices, float) - c) @ R.T
    import trimesh
    m = trimesh.Trimesh(V, np.asarray(mesh.faces), process=False)
    try:
        cs = _manifold(m).slice(0.0)
        return [np.asarray(p, float) for p in cs.to_polygons() if len(p) >= 3]
    except Exception:  # noqa: BLE001 - an invalid submitted mesh has no section
        return []


def rasterise(loops, window, px=PX):
    """Even-odd fill of the loops over window (u0, v0, u1, v1) at px mm: bool array [nv, nu] (row = v); a pixel is
    inside when its centre is (scanline crossings, half-open in v). Numpy only (the cad_photo version drew with PIL,
    which the vendored package may not import)."""
    u0, v0, u1, v1 = window
    nu, nv = int(round((u1 - u0) / px)), int(round((v1 - v0) / px))
    acc = np.zeros((nv, nu), bool)
    xs = u0 + (np.arange(nu) + 0.5) * px
    ys = (v0 + (np.arange(nv) + 0.5) * px)[:, None]
    for lp in loops:
        P = np.asarray(lp, float)
        if len(P) < 3:
            continue
        Q = np.roll(P, -1, axis=0)
        x1, y1, x2, y2 = P[None, :, 0], P[None, :, 1], Q[None, :, 0], Q[None, :, 1]
        cross = ((y1 <= ys) & (ys < y2)) | ((y2 <= ys) & (ys < y1))
        dy = np.where(y2 != y1, y2 - y1, 1.0)
        xi = np.where(cross, x1 + (ys - y1) / dy * (x2 - x1), np.inf)
        xi.sort(axis=1)
        for r in range(nv):
            row = xi[r]
            row = row[np.isfinite(row)]
            if len(row):
                acc[r] ^= (np.searchsorted(row, xs, side="right") % 2).astype(bool)
    return acc


def section_mask(mesh, frame, window, px=PX):
    return rasterise(_loops_in_plane(mesh, frame), window, px)


def cutout_item(agent_mesh, gt_mesh, feature, bands, probes=None):
    """Score one reference cut-out on the aligned submitted part (module docstring; `probes` = rules.json probes, the
    module constants by default). Returns a dict with score and stats."""
    from scipy import ndimage, signal
    P = probes or {}
    px, search_mm = float(P.get("cutout_px_mm", PX)), float(P.get("cutout_search_mm", SEARCH_MM))
    present_frac = float(P.get("cutout_present_frac", PRESENT_FRAC))
    frame = plane_frame(feature)
    fp = _vec(feature, "footprint", "footprint_mm")
    lo, hi = fp.min(axis=0), fp.max(axis=0)
    pad = search_mm + 1.0
    win = (float(lo[0] - pad), float(lo[1] - pad), float(hi[0] + pad), float(hi[1] + pad))
    F = rasterise([fp], win, px)
    gt = section_mask(gt_mesh, frame, win, px)
    ag = section_mask(agent_mesh, frame, win, px)
    R = gt | F
    E = R & ~ag
    fa = int(F.sum())
    out = {"present": 0.0, "overlap": 0.0, "offset_mm": float("nan"), "area_error": float("nan"), "score": 0.0}
    if fa == 0:
        return out
    ys, xs = np.nonzero(F)
    ky0, ky1, kx0, kx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    K = F[ky0:ky1, kx0:kx1].astype(np.float32)
    s = int(round(search_mm / px))
    y0, x0 = ky0 - s, kx0 - s
    y0c, x0c = max(0, y0), max(0, x0)
    sub = E[y0c:ky1 + s, x0c:kx1 + s].astype(np.float32)
    corr = signal.fftconvolve(sub, K[::-1, ::-1], mode="valid")
    k = np.unravel_index(int(np.argmax(corr)), corr.shape)
    best = float(corr[k]) / fa
    out["overlap"] = round(best, 4)
    if best < present_frac:
        return out
    dy, dx = (k[0] + y0c) - ky0, (k[1] + x0c) - kx0
    Fs = np.zeros_like(F)
    Fs[ky0 + dy:ky1 + dy, kx0 + dx:kx1 + dx] = F[ky0:ky1, kx0:kx1]
    lab, n = ndimage.label(E, structure=np.ones((3, 3)))
    if n == 0:
        return out
    ov = ndimage.sum(Fs, lab, index=np.arange(1, n + 1))
    comp = lab == (int(np.argmax(ov)) + 1)
    area = float(comp.sum()) * px * px
    cy, cx = [float(v) for v in ndimage.center_of_mass(comp)]
    u = win[0] + (cx + 0.5) * px
    v = win[1] + (cy + 0.5) * px
    fcy, fcx = [float(q) for q in ndimage.center_of_mass(F)]
    fu, fv = win[0] + (fcx + 0.5) * px, win[1] + (fcy + 0.5) * px
    off = math.hypot(u - fu, v - fv)
    aerr = abs(area - fa * px * px) / (fa * px * px)
    out.update({"present": 1.0, "offset_mm": round(off, 4), "area_error": round(aerr, 4),
                "score": 0.5 * (_band(off, bands["cutout_offset_mm"]) + _band(aerr, bands["cutout_area_rel"]))})
    return out


def spacer_rays(feature, probes=None):
    """(origins, direction) of the spacer probe rays in B (module docstring)."""
    P = probes or {}
    c = _vec(feature, "centre", "centre_mm")
    n = _vec(feature, "normal")
    n = n / np.linalg.norm(n)
    r = float(feature.get("radius_mm", 10.0))
    ri = float(feature.get("inner_radius_mm") or 0.0)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = ref - n * float(ref @ n)
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    ring, disc = P.get("spacer_ring", SPACER_RING), P.get("spacer_disc", SPACER_RADII)
    radii = [ri + f * (r - ri) for f in ring] if ri > 0 else [f * r for f in disc]
    back, n_ang = float(P.get("spacer_back_mm", SPACER_BACK_MM)), int(P.get("spacer_angles", SPACER_ANGLES))
    o = [c - back * n + rr * (math.cos(t) * a + math.sin(t) * b)
         for rr in radii for t in np.linspace(0.0, 2.0 * math.pi, n_ang, endpoint=False)]
    return np.asarray(o), n


def first_hit_median(mesh, origins, direction):
    """Median distance from the ray origins to the first surface along `direction`; a miss counts RAY_MISS_MM."""
    import trimesh
    dirs = np.tile(direction, (len(origins), 1))
    dist = np.full(len(origins), RAY_MISS_MM)
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            inter = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
            locs, idx_ray, _tri = inter.intersects_location(origins, dirs, multiple_hits=True)
    except Exception:  # noqa: BLE001
        return RAY_MISS_MM
    for p, i in zip(locs, idx_ray):
        dd = float(np.dot(p - origins[i], dirs[i]))
        if 1e-6 < dd < dist[i]:
            dist[i] = dd
    return float(np.median(dist))


def spacer_item(agent_mesh, gt_mesh, feature, bands, probes=None):
    """Score one reference spacer on the aligned submitted part: band of the floor error (module docstring)."""
    origins, n = spacer_rays(feature, probes)
    dg = first_hit_median(gt_mesh, origins, n)
    da = first_hit_median(agent_mesh, origins, n)
    err = da - dg
    return {"floor_ref_mm": round(dg, 4), "floor_mm": round(da, 4), "error_mm": round(err, 4), "score": _band(err, bands["spacer_mm"])}


def extent_errors(agent_mesh, gt_mesh):
    """|extent_agent - extent_reference| per B axis of the two bounding boxes (the agent part aligned)."""
    ea = agent_mesh.bounds[1] - agent_mesh.bounds[0]
    eg = gt_mesh.bounds[1] - gt_mesh.bounds[0]
    return np.abs(np.asarray(ea, float) - np.asarray(eg, float))
