"""Render the B930 bookcase's drawing set (PNG sheets) from the furniture description.

The set is what a manufacturer ships: a cover with the overall dimensions, a parts list, dimensioned part drawings,
wordless assembly steps (exploded axonometric views with arrows, the two-person pictogram, detail callouts) and the
customer's floor plan with the bookcase's place. Every number on a sheet comes from `furniture` / `room`, so the
drawings and the physics cell cannot disagree.

    python -m furniture_cell.drawings OUT_DIR      (from the tests/ folder; authoring tool, not run by the grader)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle  # noqa: E402

from . import furniture as F  # noqa: E402
from . import room as RM  # noqa: E402

SHEET_IN = (11.69, 8.27)   # A4 landscape
DPI = 170
INK = "black"
BAND = "#b9b9b9"           # fill of an edge-banded face in the axonometric views
FINISH = "#dcdcdc"         # fill of the back panel's white face
LW = 1.3

FACES = {  # name -> (axis index, sign)
    "+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1),
}


# ---------------------------------------------------------------- projection

_Z = [10.0]


def _z() -> float:
    """Increasing zorder, so later-drawn hidden-surface elements cover earlier ones (lines and patches alike)."""
    _Z[0] += 0.01
    return _Z[0]


class Axo:
    """Orthographic axonometric projection of furniture-frame millimetres onto the sheet (inches)."""

    def __init__(self, azim_deg: float = -38.0, elev_deg: float = 24.0, scale: float = 0.0042,
                 origin: tuple[float, float] = (0.0, 0.0)) -> None:
        a, e = math.radians(azim_deg), math.radians(elev_deg)
        # direction from the scene towards the viewer (front = -y, left = -x for a negative azimuth)
        self.d = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
        up = np.array([0.0, 0.0, 1.0])
        self.right = np.cross(up, self.d)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.d, self.right)
        self.scale = scale
        self.origin = np.array(origin, dtype=float)

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        return np.stack([pts @ self.right, pts @ self.up], axis=1) * self.scale + self.origin

    def depth(self, pts: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(pts, dtype=float)) @ self.d


def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class Placed:
    """A part drawn at a pose: world = R @ local + t (mm)."""

    def __init__(self, part: F.Part, t: tuple[float, float, float] | None = None,
                 R: np.ndarray | None = None) -> None:
        self.part = part
        self.R = np.eye(3) if R is None else R
        self.t = np.array(part.design_center if t is None else t, dtype=float)

    def world(self, local: np.ndarray) -> np.ndarray:
        return np.atleast_2d(local) @ self.R.T + self.t

    def faces(self):
        """(polygon world corners, outward world normal, face name, box index) for every box face."""
        out = []
        for bi, box in enumerate(self.part.boxes):
            c, h = np.array(box.center), np.array(box.half)
            for name, (ax, sg) in FACES.items():
                n = np.zeros(3)
                n[ax] = sg
                u_ax, v_ax = [i for i in range(3) if i != ax]
                corners = []
                for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    p = c.copy()
                    p[ax] += sg * h[ax]
                    p[u_ax] += su * h[u_ax]
                    p[v_ax] += sv * h[v_ax]
                    corners.append(p)
                out.append((self.world(np.array(corners)), self.R @ n, name, bi))
        return out


def _occlusion_order(proj: Axo, items: list[tuple[Placed, int, np.ndarray, np.ndarray]]) -> list[int]:
    """Back-to-front order of world-axis-aligned boxes: for two disjoint boxes, the one on the viewer's side of a
    separating axis is drawn later. Ties and cycles fall back to the mean depth."""
    n = len(items)
    after: list[set[int]] = [set() for _ in range(n)]      # after[i] = boxes that must be drawn after i
    for i in range(n):
        for j in range(i + 1, n):
            lo_i, hi_i = items[i][2], items[i][3]
            lo_j, hi_j = items[j][2], items[j][3]
            best, nearer = 0.0, None
            for k in range(3):
                for gap, i_lower in ((lo_j[k] - hi_i[k], True), (lo_i[k] - hi_j[k], False)):
                    if gap < -1e-6:
                        continue
                    score = (gap + 1.0) * abs(proj.d[k])
                    if score > best:
                        best = score
                        # the box at the higher coordinate is nearer when the viewer sits on the + side
                        upper_is_j = i_lower
                        nearer = (j if upper_is_j else i) if proj.d[k] > 0 else (i if upper_is_j else j)
            if nearer is None:
                continue
            farther = i if nearer == j else j
            after[farther].add(nearer)
    depth = [float(proj.depth(((it[2] + it[3]) / 2)[None])[0]) for it in items]
    indeg = [0] * n
    for i in range(n):
        for j in after[i]:
            indeg[j] += 1
    ready = sorted([i for i in range(n) if indeg[i] == 0], key=lambda i: depth[i])
    order: list[int] = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for j in after[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
                ready.sort(key=lambda k: depth[k])
    order += sorted([i for i in range(n) if i not in order], key=lambda i: depth[i])
    return order


def draw_axo(ax, proj: Axo, placed: list[Placed], *, label: bool = True, holes: bool = True,
             label_offset: dict[str, tuple[float, float]] | None = None, lw: float = LW) -> None:
    """Hidden-surface line drawing: boxes back to front, each box's visible faces (which never overlap)."""
    items = []
    for pl in placed:
        for bi, box in enumerate(pl.part.boxes):
            c, h = np.array(box.center), np.array(box.half)
            corners = pl.world(np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]))
            items.append((pl, bi, corners.min(axis=0), corners.max(axis=0)))
    for idx in _occlusion_order(proj, items):
        pl, bi = items[idx][0], items[idx][1]
        p = pl.part
        for corners, normal, name, fbi in pl.faces():
            if fbi != bi or normal @ proj.d <= 1e-9:
                continue
            fill = "white"
            if p.banded_face == name and bi == 0:
                fill = BAND
            if p.finished_face == name:
                fill = FINISH
            ax.add_patch(Polygon(proj(corners), closed=True, facecolor=fill, edgecolor=INK, lw=lw,
                                 joinstyle="round", zorder=_z()))
            if holes and bi == 0:
                _draw_features_on_face(ax, proj, pl, name)
    if label:
        for pl in placed:
            off = (label_offset or {}).get(pl.part.key, (0.0, 0.0))
            c = proj(pl.t[None, :])[0] + np.array(off)
            part_letter(ax, c, pl.part.letter)


def _draw_features_on_face(ax, proj: Axo, pl: Placed, face: str) -> None:
    axis_i, sign = FACES[face]
    for hole in pl.part.holes:
        a = np.array(hole.axis)
        if a[axis_i] * -sign < 0.5:      # the hole opens on the face whose outward normal is -axis
            continue
        _ellipse(ax, proj, pl, np.array(hole.pos), a, F.DOWEL_D / 2, filled=True)
    for dowel in pl.part.dowels:
        a = np.array(dowel.axis)
        if a[axis_i] * sign < 0.5:
            continue
        base = np.array(dowel.pos)
        tip = base + a * F.DOWEL_OUT
        w0, w1 = pl.world(base)[0], pl.world(tip)[0]
        s0, s1 = proj(w0[None])[0], proj(w1[None])[0]
        ax.plot([s0[0], s1[0]], [s0[1], s1[1]], color=INK, lw=2.2, solid_capstyle="round", zorder=_z())
        _ellipse(ax, proj, pl, tip, a, F.DOWEL_D / 2, filled=False)


def _ellipse(ax, proj: Axo, pl: Placed, center_local, axis_local, radius, filled: bool) -> None:
    a = axis_local / np.linalg.norm(axis_local)
    u = np.cross(a, [0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.cross(a, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    ts = np.linspace(0, 2 * math.pi, 24)
    ring = center_local[None, :] + radius * (np.cos(ts)[:, None] * u + np.sin(ts)[:, None] * v)
    pts = proj(pl.world(ring))
    ax.add_patch(Polygon(pts, closed=True, facecolor=INK if filled else "white", edgecolor=INK, lw=0.6, zorder=_z()))


def part_letter(ax, xy, letter: str, r: float = 0.17) -> None:
    ax.add_patch(Circle(xy, r, facecolor="white", edgecolor=INK, lw=1.4, zorder=900))
    ax.text(xy[0], xy[1], letter, ha="center", va="center", fontsize=13, fontweight="bold", zorder=901)


def arrow3d(ax, proj: Axo, p0, p1, lw: float = 2.6) -> None:
    s0, s1 = proj(np.array([p0]))[0], proj(np.array([p1]))[0]
    ax.add_patch(FancyArrowPatch(s0, s1, arrowstyle="-|>", mutation_scale=22, lw=lw, color=INK, zorder=800))


# ---------------------------------------------------------------- sheet furniture

def new_sheet(title: str | None = None, number: int | None = None):
    fig = plt.figure(figsize=SHEET_IN, dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, SHEET_IN[0])
    ax.set_ylim(0, SHEET_IN[1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0.25, 0.25), SHEET_IN[0] - 0.5, SHEET_IN[1] - 0.5, fill=False, lw=1.0, ec=INK))
    if number is not None:
        ax.text(SHEET_IN[0] - 0.45, 0.42, f"{number}", ha="right", va="bottom", fontsize=16, fontweight="bold")
    if title:
        ax.text(0.5, SHEET_IN[1] - 0.55, title, ha="left", va="top", fontsize=14, fontweight="bold")
    ax.text(0.45, 0.42, "B930", ha="left", va="bottom", fontsize=10, color="#444")
    return fig, ax


def persons(ax, xy, n: int = 2, size: float = 0.42) -> None:
    """The 'n persons' pictogram (stick figures in a rounded frame)."""
    x0, y0 = xy
    w = size * (0.75 * n + 0.35)
    ax.add_patch(Rectangle((x0, y0), w, size * 1.35, fill=False, lw=1.4, ec=INK))
    for i in range(n):
        cx = x0 + size * (0.55 + 0.75 * i)
        ax.add_patch(Circle((cx, y0 + size * 1.02), size * 0.15, color=INK))
        ax.add_patch(Polygon([(cx - size * 0.22, y0 + size * 0.15), (cx + size * 0.22, y0 + size * 0.15),
                              (cx + size * 0.17, y0 + size * 0.82), (cx - size * 0.17, y0 + size * 0.82)],
                             closed=True, color=INK))


def step_number(ax, n: int) -> None:
    ax.text(0.55, SHEET_IN[1] - 0.6, f"{n}", ha="left", va="top", fontsize=40, fontweight="bold")


def dim_line(ax, p0, p1, text: str, offset: float, *, fontsize: float = 9, ext: bool = True) -> None:
    """A dimension between two sheet points, drawn `offset` inches to the left of the direction p0 -> p1."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-9:
        return
    d /= L
    n = np.array([-d[1], d[0]])
    q0, q1 = p0 + n * offset, p1 + n * offset
    if ext:
        for p, q in ((p0, q0), (p1, q1)):
            ax.plot([p[0] + n[0] * 0.04 * np.sign(offset), q[0] + n[0] * 0.06 * np.sign(offset)],
                    [p[1] + n[1] * 0.04 * np.sign(offset), q[1] + n[1] * 0.06 * np.sign(offset)],
                    color=INK, lw=0.6, zorder=_z())
    ax.add_patch(FancyArrowPatch(q0, q1, arrowstyle="<|-|>", mutation_scale=8, lw=0.7, color=INK,
                                 shrinkA=0, shrinkB=0, zorder=_z()))
    mid = (q0 + q1) / 2 + n * 0.09 * np.sign(offset if offset else 1)
    ang = math.degrees(math.atan2(d[1], d[0]))
    if ang > 90.5 or ang < -89.5:
        ang += 180
    ax.text(mid[0], mid[1], text, ha="center", va="center", rotation=ang, fontsize=fontsize,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=_z())


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)


def _fmt(v: float) -> str:
    return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


# ---------------------------------------------------------------- sheets

def sheet_cover(out: Path) -> None:
    fig, ax = new_sheet()
    ax.text(0.6, SHEET_IN[1] - 0.7, "B930", fontsize=40, fontweight="bold", va="top")
    ax.text(0.62, SHEET_IN[1] - 1.45, "bookcase  ·  assembly drawings  ·  dimensions in mm", fontsize=12, va="top")
    proj = Axo(scale=0.0047, origin=(3.7, 1.55))
    draw_axo(ax, proj, [Placed(p) for p in F.PARTS.values()], label=False)
    # overall dimensions along the visible bottom-front and left edges
    W, D, H = F.WIDTH, F.DEPTH, F.HEIGHT
    fl, fr, bl, tb = (np.array(v) for v in ([-W / 2, 0, 0], [W / 2, 0, 0], [-W / 2, D, 0], [-W / 2, D, H]))
    dim_line(ax, proj(fl[None])[0], proj(fr[None])[0], _fmt(W), -0.35)
    dim_line(ax, proj(bl[None])[0], proj(fl[None])[0], _fmt(D), -0.35)
    dim_line(ax, proj(bl[None])[0], proj(tb[None])[0], _fmt(H), 0.35)
    persons(ax, (8.4, 5.6), 2, size=0.55)
    ax.text(8.4, 5.35, "2", fontsize=14, fontweight="bold", va="top")
    # legend of the two finishes
    lx, ly = 8.0, 3.6
    ax.add_patch(Rectangle((lx, ly), 0.35, 0.22, facecolor=BAND, edgecolor=INK, lw=1))
    ax.text(lx + 0.5, ly + 0.11, "white edge banding (visible edge)", va="center", fontsize=9)
    ax.add_patch(Rectangle((lx, ly - 0.45), 0.35, 0.22, facecolor=FINISH, edgecolor=INK, lw=1))
    ax.text(lx + 0.5, ly - 0.34, "white face (inside of the bookcase)", va="center", fontsize=9)
    ax.add_patch(Rectangle((lx, ly - 0.9), 0.35, 0.22, facecolor="white", edgecolor=INK, lw=1))
    ax.text(lx + 0.5, ly - 0.79, "raw edge / raw face", va="center", fontsize=9)
    ax.text(lx, ly - 1.45, "dowels Ø8 pre-installed\nno tools", fontsize=9, va="top")
    save(fig, out)


def sheet_parts(out: Path) -> None:
    fig, ax = new_sheet("Parts", 1)
    cells = [("L", 1, (1.6, 3.2)), ("R", 1, (3.7, 3.2)), ("T", 2, (6.0, 5.7)), ("D", 1, (6.0, 3.0)),
             ("K", 1, (9.0, 3.4))]
    for key, qty, (x, y) in cells:
        p = F.PARTS[key]
        proj = Axo(scale=0.0036, origin=(x, y))
        draw_axo(ax, proj, [Placed(p, t=(0.0, 0.0, 0.0))], label=False)
        ax.text(x - 0.15, y - 2.25 if key in "LRK" else y - 1.05, f"{qty}x", fontsize=13, fontweight="bold")
        part_letter(ax, (x + 0.45, y - 2.18 if key in "LRK" else y - 0.98), p.letter)
        sx, sy, sz = p.size
        dims = f"{_fmt(max(sx, sz))} x {_fmt(sy if p.key not in 'LRK' else (sy if p.key != 'K' else sx))}"
        ax.text(x + 0.75, y - 2.18 if key in "LRK" else y - 0.98, _part_size_text(p), fontsize=8, va="center")
    save(fig, out)


def _part_size_text(p: F.Part) -> str:
    dims = sorted(p.size, reverse=True)
    return f"{_fmt(dims[0])} x {_fmt(dims[1])} x {_fmt(dims[2])}"


def sheet_side_panels(out: Path) -> None:
    """Dimensioned views of the two side panels, seen on their inner (drilled) face."""
    fig, ax = new_sheet("A / B  side panels  (view on the drilled inner face)", 2)
    s = 0.0062                        # inches per mm
    for key, x0 in (("L", 1.3), ("R", 6.3)):
        p = F.PARTS[key]
        y0 = 1.0
        D, H = F.DEPTH, F.HEIGHT
        # inner-face view: horizontal = depth (front at the left for A, at the right for B), vertical = height
        front_left = key == "L"

        def X(y_mm: float) -> float:
            return x0 + s * (y_mm if front_left else D - y_mm)

        def Y(z_mm: float) -> float:
            return y0 + s * z_mm

        ax.add_patch(Rectangle((X(0) if front_left else X(D), Y(0)), s * D, s * H, fill=False, ec=INK, lw=1.4))
        # rabbet along the back edge (seen on the inner face as a band of width RABBET_W)
        yb = D - F.RABBET_W
        ax.plot([X(yb), X(yb)], [Y(0), Y(H)], color=INK, lw=1.0)
        ax.add_patch(Rectangle((min(X(yb), X(D)), Y(0)), s * F.RABBET_W, s * H, facecolor="#e6e6e6", ec="none"))
        # edge banding on the front edge
        ax.plot([X(0), X(0)], [Y(0), Y(H)], color=BAND, lw=6, solid_capstyle="butt", zorder=1)
        ax.plot([X(0), X(0)], [Y(0), Y(H)], color=INK, lw=1.4)
        # holes
        zs, ys = set(), set()
        for h in p.holes:
            yw = h.pos[1] + p.design_center[1]
            zw = h.pos[2] + p.design_center[2]
            ax.add_patch(Circle((X(yw), Y(zw)), s * F.DOWEL_D / 2 * 1.6, color=INK))
            zs.add(zw)
            ys.add(yw)
        # height chain on the outer side of the front edge
        side = -1 if front_left else 1
        for i, z in enumerate(sorted(zs)):
            dim_line(ax, (X(0), Y(0)), (X(0), Y(z)), _fmt(z), side * (0.35 + 0.28 * i), fontsize=8)
        dim_line(ax, (X(0), Y(0)), (X(0), Y(H)), _fmt(H), side * (0.35 + 0.28 * len(zs)), fontsize=8)
        # depth chain below
        for i, yv in enumerate(sorted(ys)):
            dim_line(ax, (X(0), Y(0)), (X(yv), Y(0)), _fmt(yv), (-0.25 - 0.22 * i) * (1 if front_left else -1),
                     fontsize=8)
        dim_line(ax, (X(0), Y(H)), (X(D), Y(H)), _fmt(D), 0.25 * (1 if front_left else -1), fontsize=8)
        part_letter(ax, (X(D / 2), Y(H) + 0.55), p.letter)
        ax.text(X(0), Y(H) + 0.62, "front edge (banded)", ha="center", fontsize=8)
        ax.text(X(D / 2), Y(-60), f"holes Ø{_fmt(F.DOWEL_D)}", ha="center", fontsize=8)
    # rabbet section detail: the last 30 mm of the board at its back edge, inner face up
    cx, cy = 9.2, 5.4
    k = 0.045
    ax.text(cx + 0.7, cy + 1.3, "section at the back edge", fontsize=9, ha="center")
    x_cut = F.DEPTH - 30.0
    pts = [(x_cut, 0), (x_cut, F.BOARD), (F.DEPTH - F.RABBET_W, F.BOARD), (F.DEPTH - F.RABBET_W, F.BOARD - F.RABBET_D),
           (F.DEPTH, F.BOARD - F.RABBET_D), (F.DEPTH, 0)]
    pts = [(cx + k * (x - x_cut), cy + k * y) for x, y in pts]
    ax.add_patch(Polygon(pts, closed=True, fill=False, ec=INK, lw=1.3, hatch="////"))
    xr = cx + k * (F.DEPTH - F.RABBET_W - x_cut)
    xe = cx + k * (F.DEPTH - x_cut)
    dim_line(ax, (xr, cy + k * F.BOARD), (xe, cy + k * F.BOARD), _fmt(F.RABBET_W), 0.3, fontsize=8)
    dim_line(ax, (xe, cy + k * (F.BOARD - F.RABBET_D)), (xe, cy + k * F.BOARD), _fmt(F.RABBET_D), -0.3, fontsize=8)
    dim_line(ax, (xe, cy), (xe, cy + k * (F.BOARD - F.RABBET_D)), _fmt(F.BOARD - F.RABBET_D), -0.3, fontsize=8)
    ax.text(cx - 0.1, cy + k * F.BOARD + 0.12, "inner face", fontsize=8)
    ax.text(xe + 0.12, cy - 0.25, "back edge", fontsize=8)
    save(fig, out)


def sheet_horizontal_parts(out: Path) -> None:
    fig, ax = new_sheet("C  top / bottom panel     D  fixed shelf     E  back panel", 3)
    s = 0.0068
    for key, (x0, y0) in (("T", (1.0, 4.6)), ("D", (1.0, 1.1))):
        p = F.PARTS[key]
        L, Dp = p.size[0], p.size[1]
        y_front = p.design_center[1] - Dp / 2

        def X(x_mm: float) -> float:
            return x0 + s * (x_mm + L / 2)

        def Y(y_mm: float) -> float:   # top view: front edge at the bottom
            return y0 + s * (y_mm - y_front)

        ax.add_patch(Rectangle((X(-L / 2), Y(y_front)), s * L, s * Dp, fill=False, ec=INK, lw=1.4))
        ax.plot([X(-L / 2), X(L / 2)], [Y(y_front), Y(y_front)], color=BAND, lw=6, solid_capstyle="butt")
        ax.plot([X(-L / 2), X(L / 2)], [Y(y_front), Y(y_front)], color=INK, lw=1.4)
        ys = []
        for d in p.dowels:
            xw = d.pos[0] + d.axis[0] * 0
            yw = d.pos[1] + p.design_center[1]
            ax.add_patch(Rectangle((X(xw) if d.axis[0] > 0 else X(xw) - s * F.DOWEL_OUT, Y(yw) - s * F.DOWEL_D / 2),
                                   s * F.DOWEL_OUT, s * F.DOWEL_D, facecolor=INK, ec=INK))
            ys.append(yw)
        for i, yv in enumerate(sorted(set(ys))):
            dim_line(ax, (X(L / 2) + s * F.DOWEL_OUT, Y(y_front)), (X(L / 2) + s * F.DOWEL_OUT, Y(yv)),
                     _fmt(yv - y_front), -(0.3 + 0.25 * i), fontsize=8)
        dim_line(ax, (X(-L / 2), Y(y_front)), (X(-L / 2), Y(y_front + Dp)), _fmt(Dp), 0.3, fontsize=8)
        dim_line(ax, (X(-L / 2), Y(y_front + Dp)), (X(L / 2), Y(y_front + Dp)), _fmt(L), 0.3, fontsize=8)
        dim_line(ax, (X(L / 2), Y(y_front + Dp)), (X(L / 2) + s * F.DOWEL_OUT, Y(y_front + Dp)),
                 _fmt(F.DOWEL_OUT), 0.3, fontsize=7)
        qty = 2 if key == "T" else 1
        part_letter(ax, (X(-L / 2) - 0.3, Y(y_front + Dp) + 0.45), p.letter)
        ax.text(X(-L / 2) - 0.05, Y(y_front + Dp) + 0.45, f"{qty}x", fontsize=12, fontweight="bold", va="center")
        ax.text(X(0), Y(y_front) - 0.28, f"front edge (banded)   ·   thickness {_fmt(F.BOARD)}   ·   "
                f"dowels Ø{_fmt(F.DOWEL_D)}, centred in the thickness", ha="center", fontsize=8)
    # back panel
    k = F.PARTS["K"]
    s2 = 0.0042
    x0, y0 = 7.4, 1.5
    W, H = k.size[0], k.size[2]
    ax.add_patch(Rectangle((x0, y0), s2 * W, s2 * H, facecolor=FINISH, ec=INK, lw=1.4))
    dim_line(ax, (x0, y0 + s2 * H), (x0 + s2 * W, y0 + s2 * H), _fmt(W), 0.3, fontsize=8)
    dim_line(ax, (x0, y0), (x0, y0 + s2 * H), _fmt(H), 0.3, fontsize=8)
    part_letter(ax, (x0 + s2 * W / 2, y0 + s2 * H / 2), k.letter)
    ax.text(x0 + s2 * W / 2, y0 - 0.3, f"white face shown  ·  thickness {_fmt(F.BACK)}", ha="center", fontsize=8)
    save(fig, out)


def _exploded(ax, proj: Axo, placed_now: list[Placed], moving: list[tuple[Placed, np.ndarray]],
              offsets: dict[str, tuple[float, float]] | None = None) -> None:
    allp = placed_now + [m for m, _ in moving]
    draw_axo(ax, proj, allp, label_offset=offsets)
    for m, off in moving:
        # over the gap between the moving part's leading face and where that face ends up
        direction = off / np.linalg.norm(off)
        lead = float(np.abs(direction) @ np.array(m.part.half))
        start = m.t + direction * (lead + 15.0)
        arrow3d(ax, proj, start, start + off * 0.8)


def sheet_step(out: Path, number: int, done: list[str], moving: dict[str, tuple[float, float, float]],
               two_persons: bool, callout=None, offsets=None, scale: float = 0.0046,
               origin=(4.3, 1.4), azim: float = -38.0) -> None:
    fig, ax = new_sheet(None, number + 3)
    step_number(ax, number)
    if two_persons:
        persons(ax, (1.4, SHEET_IN[1] - 1.25), 2, size=0.42)
    proj = Axo(scale=scale, origin=origin, azim_deg=azim)
    now = [Placed(F.PARTS[k]) for k in done]
    mv = []
    for key, back_off in moving.items():
        p = F.PARTS[key]
        off = np.array(back_off)
        mv.append((Placed(p, t=tuple(np.array(p.design_center) - off)), off))
    _exploded(ax, proj, now, mv, offsets)
    if callout:
        callout(fig, ax)
    save(fig, out)


def _callout_banding(fig, ax) -> None:
    """Detail: the banded (white) edges of the side panel and of the inserted panels face the same way (front)."""
    cx, cy, r = 9.4, 2.0, 1.15
    ax.add_patch(Circle((cx, cy), r, fill=False, ec=INK, lw=1.4))
    proj = Axo(scale=0.009, origin=(cx + 0.35, cy - 0.35), azim_deg=-20, elev_deg=30)
    L, C = F.PARTS["L"], F.PARTS["C"]
    corner = np.array([-391.0, 0.0, 109.0])
    sub = [Placed(L, t=tuple(np.array(L.design_center) - corner)),
           Placed(C, t=tuple(np.array(C.design_center) - corner + np.array([0.0, 0.0, 0.0])))]
    # clip by drawing in a separate small axes
    sax = fig.add_axes([(cx - r) / SHEET_IN[0], (cy - r) / SHEET_IN[1], 2 * r / SHEET_IN[0], 2 * r / SHEET_IN[1]])
    sax.set_xlim(cx - r, cx + r)
    sax.set_ylim(cy - r, cy + r)
    sax.set_aspect("equal")
    sax.axis("off")
    clip = Circle((cx, cy), r * 0.98, transform=sax.transData)
    before = len(sax.patches)
    draw_axo(sax, proj, sub, label=False, lw=1.6)
    for patch in sax.patches[before:]:
        patch.set_clip_path(clip)
    ax.plot([7.0, cx - r], [3.0, cy + 0.3], color=INK, lw=0.8)


def _callout_back(fig, ax) -> None:
    cx, cy, r = 9.5, 2.0, 1.1
    ax.add_patch(Circle((cx, cy), r, fill=False, ec=INK, lw=1.4))
    ax.add_patch(Rectangle((cx - 0.7, cy - 0.45), 0.55, 0.9, facecolor=FINISH, ec=INK, lw=1.2))
    ax.add_patch(Rectangle((cx + 0.15, cy - 0.45), 0.55, 0.9, facecolor="white", ec=INK, lw=1.2))
    ax.text(cx - 0.42, cy - 0.65, "✓", ha="center", fontsize=18, fontweight="bold")
    ax.text(cx + 0.42, cy - 0.65, "✗", ha="center", fontsize=18, fontweight="bold")
    ax.text(cx, cy + 0.7, "seen from the front", ha="center", fontsize=8)


def sheet_floor_plan(out: Path) -> None:
    fig, ax = new_sheet("Place of the bookcase  (floor plan, mm)", 9)
    s = 0.00125
    x0, y0 = 1.2, 0.9
    W, Dm = RM.ROOM_X * 1000, RM.ROOM_Y * 1000

    def P(x_mm: float, y_mm: float) -> tuple[float, float]:
        return x0 + s * x_mm, y0 + s * y_mm

    t = 0.12 / s   # wall thickness on the sheet
    for (xa, ya, xb, yb) in ((0, -t, W, 0), (0, Dm, W, Dm + t), (-t, -t, 0, Dm + t), (W, -t, W + t, Dm + t)):
        ax.add_patch(Rectangle(P(xa, ya), s * (xb - xa), s * (yb - ya), facecolor="#555", ec=INK))
    dx0, dx1 = RM.DOOR_X[0] * 1000, RM.DOOR_X[1] * 1000
    ax.add_patch(Rectangle(P(dx0, -t), s * (dx1 - dx0), s * t, facecolor="white", ec="white"))
    ax.add_patch(matplotlib.patches.Arc(P(dx0, 0), 2 * s * (dx1 - dx0), 2 * s * (dx1 - dx0), theta1=0, theta2=90,
                                        lw=0.8, ec=INK))
    ax.plot(*zip(P(dx0, 0), P(dx0, dx1 - dx0)), color=INK, lw=1.2)
    wy0, wy1 = RM.WINDOW_Y[0] * 1000, RM.WINDOW_Y[1] * 1000
    ax.add_patch(Rectangle(P(W, wy0), s * t, s * (wy1 - wy0), facecolor="white", ec=INK, lw=0.8))
    ax.plot(*zip(P(W + t / 2, wy0), P(W + t / 2, wy1)), color=INK, lw=0.8)
    # the customer's base cabinet (existing) and the bookcase on top of it
    cx, cy = RM.SIDEBOARD["center"][0] * 1000, RM.SIDEBOARD["center"][1] * 1000
    cw, cd, ch = (v * 1000 for v in RM.SIDEBOARD["size"])
    ax.add_patch(Rectangle(P(cx - cw / 2, cy - cd / 2), s * cw, s * cd, facecolor="#e8e8e8", ec=INK, lw=1.0))
    tx, ty, tz, tyaw = RM.TARGET
    c, sn = math.cos(tyaw), math.sin(tyaw)
    corners = []
    for fx, fy in ((-F.WIDTH / 2, 0), (F.WIDTH / 2, 0), (F.WIDTH / 2, F.DEPTH), (-F.WIDTH / 2, F.DEPTH)):
        corners.append(P(tx * 1000 + c * fx - sn * fy, ty * 1000 + sn * fx + c * fy))
    ax.add_patch(Polygon(corners, closed=True, facecolor="#cfcfcf", ec=INK, lw=1.6, hatch="xx"))
    front_mid = P(tx * 1000, ty * 1000)
    ax.add_patch(FancyArrowPatch(front_mid, (front_mid[0], front_mid[1] - 0.45), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.8, color=INK))
    ax.text(front_mid[0] + 0.1, front_mid[1] - 0.35, "front", fontsize=9)
    # dimensions of the cabinet's place: west wall to its left side, north wall to its back
    left_x = cx - cw / 2
    back_y = cy + cd / 2
    dim_line(ax, P(0, cy), P(left_x, cy), _fmt(left_x), 0.0, fontsize=9, ext=False)
    dim_line(ax, P(cx, back_y), P(cx, Dm), _fmt(Dm - back_y), 0.0, fontsize=9, ext=False)
    dim_line(ax, P(cx - cw / 2, cy - cd / 2), P(cx + cw / 2, cy - cd / 2), _fmt(cw), -0.25, fontsize=8)
    dim_line(ax, P(0, -t), P(W, -t), _fmt(W), -0.3, fontsize=9)
    dim_line(ax, P(W + t, 0), P(W + t, Dm), _fmt(Dm), -0.25, fontsize=9)
    ax.text(*P((dx0 + dx1) / 2, 150), "door", ha="center", fontsize=8)
    ax.text(P(W, 0)[0] - 0.55, P(0, (wy0 + wy1) / 2)[1], "window", fontsize=8, va="center")
    # legend and the elevation detail: the bookcase on the cabinet, flush at the sides and the back
    lx, ly = 9.15, 1.9
    ax.add_patch(Rectangle((lx, ly), 0.35, 0.22, facecolor="#e8e8e8", ec=INK, lw=1))
    ax.text(lx + 0.5, ly + 0.11, "existing base cabinet, %s high" % _fmt(ch), va="center", fontsize=8)
    ax.add_patch(Rectangle((lx, ly - 0.45), 0.35, 0.22, facecolor="#cfcfcf", ec=INK, lw=1, hatch="xx"))
    ax.text(lx + 0.5, ly - 0.34, "B930 bookcase on top of it", va="center", fontsize=8)
    ex, ey, k = 9.45, 3.1, 0.0011
    ax.add_patch(Rectangle((ex, ey), k * cw, k * ch, facecolor="#e8e8e8", ec=INK, lw=1.0))
    ax.add_patch(Rectangle((ex, ey + k * ch), k * F.WIDTH, k * F.HEIGHT, facecolor="white", ec=INK, lw=1.4))
    dim_line(ax, (ex + k * cw, ey), (ex + k * cw, ey + k * ch), _fmt(ch), -0.25, fontsize=8)
    ax.text(ex + k * cw / 2, ey - 0.25, "front view:\nflush at both sides\nand at the back", ha="center", va="top",
            fontsize=8)
    nx, ny = 10.6, 6.6
    ax.add_patch(FancyArrowPatch((nx, ny - 0.5), (nx, ny + 0.2), arrowstyle="-|>", mutation_scale=16, lw=1.5,
                                 color=INK))
    ax.text(nx, ny + 0.3, "N", ha="center", fontsize=12, fontweight="bold")
    save(fig, out)


def render_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def p(name: str) -> Path:
        path = out_dir / name
        paths.append(path)
        return path

    sheet_cover(p("sheet_00_cover.png"))
    sheet_parts(p("sheet_01_parts.png"))
    sheet_side_panels(p("sheet_02_side_panels.png"))
    sheet_horizontal_parts(p("sheet_03_panels.png"))
    sheet_step(p("sheet_04_step1.png"), 1, ["L"], {"C": (-420.0, 0.0, 0.0)}, True, _callout_banding,
               azim=38.0, origin=(3.2, 1.5))
    sheet_step(p("sheet_05_step2.png"), 2, ["L", "C"], {"D": (-420.0, 0.0, 0.0)}, True, azim=38.0, origin=(3.2, 1.5))
    sheet_step(p("sheet_06_step3.png"), 3, ["L", "C", "D"], {"T": (-420.0, 0.0, 0.0)}, True, azim=38.0,
               origin=(3.2, 1.5))
    sheet_step(p("sheet_07_step4.png"), 4, ["L", "C", "D", "T"], {"R": (-380.0, 0.0, 0.0)}, True)
    sheet_step(p("sheet_08_step5.png"), 5, ["L", "C", "D", "T", "R"], {"K": (0.0, -330.0, 0.0)}, False,
               _callout_back)
    sheet_floor_plan(p("sheet_09_floor_plan.png"))
    return paths


if __name__ == "__main__":
    for path in render_all(Path(sys.argv[1] if len(sys.argv) > 1 else "drawings_out")):
        print(path)
