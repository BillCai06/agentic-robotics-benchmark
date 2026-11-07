"""Per-episode layout draw: where each part lies, which way it faces, where the robots start.

The tall parts (L, R, K) stand upright in a random groove each of the panel cart, either face north; the horizontal parts (T, C, D) lie flat on the kitting table's battens
in a random slot each, front edge north or south, with a few millimetres / degrees of jitter. The robots start near
the south wall. Everything is a pure function of the episode seed.
"""

from __future__ import annotations

import math

import numpy as np

from . import furniture as F
from . import room as RM

MM = 1e-3


def _q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _q_axis(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * axis])


def _q_rotate(q, v):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ np.asarray(v, float)


def draw(seed: int, nominal: bool = False) -> dict:
    """{"parts": {key: (pos, quat)}, "robots": {name: (x, y, yaw)}, "facts": {...}} for one episode.
    `nominal`: the reference layout (no jitter; fixed spots and facings) used by the privileged reference."""
    rng = np.random.default_rng(int(seed) & ((1 << 64) - 1))
    parts, facts = {}, {}
    # ---- tall panels on the panel cart: each stands upright in a groove (its board plane = the groove's x-z plane),
    # either face may look north; the board's long axis (L, R: its depth, K: its width) runs along the groove (x)
    st = RM.PANEL_CART
    order = ["L", "R", "K"] if nominal else list(rng.permutation(["L", "R", "K"]))
    groove_gap = {}
    for i, (key, y_slot) in enumerate(zip(order, st["slots_y"])):
        p = F.PARTS[key]
        groove_gap[i] = min(p.size) * MM + st["play"]
        # design frames: L, R thickness along x, depth along y; K thickness along y, width along x
        q = _q_axis([0, 0, 1], math.pi / 2) if key in "LR" else np.array([1.0, 0.0, 0.0, 0.0])
        flip = False if nominal else bool(rng.integers(2))    # yaw 180 deg: the other face looks north
        if flip:
            q = _q_mul(_q_axis([0, 0, 1], math.pi), q)
        x = st["x"] + (0.0 if nominal else float(rng.uniform(-0.04, 0.04)))
        parts[key] = (np.array([x, y_slot, st["bed_h"] + p.size[2] / 2 * MM + 0.001]), q)
        facts[key] = {"where": "cart", "slot_y": y_slot, "flip": flip}
    # ---- horizontal panels on the table battens
    tx, ty = RM.TABLE["center"]
    top = RM.TABLE["height"] + RM.TABLE["batten_h"]
    order = ["T", "C", "D"] if nominal else list(rng.permutation(["T", "C", "D"]))
    for key, sx in zip(order, RM.TABLE_SLOTS_X):
        p = F.PARTS[key]
        flip = False if nominal else bool(rng.integers(2))   # front edge north instead of south
        yaw = (math.pi if flip else 0.0) + (0.0 if nominal else float(rng.uniform(-0.04, 0.04)))
        q = _q_axis([0, 0, 1], yaw)
        x = tx + sx + (0.0 if nominal else float(rng.uniform(-0.015, 0.015)))
        y = ty + (0.0 if nominal else float(rng.uniform(-0.03, 0.03)))
        parts[key] = (np.array([x, y, top + p.size[2] / 2 * MM + 0.001]), q)
        facts[key] = {"where": "table", "slot_x": sx, "flip": flip}
    # ---- robots
    robots = {}
    for name, (x, y, yaw) in RM.ROBOT_HOME.items():
        if not nominal:
            x += float(rng.uniform(-0.15, 0.15))
            y += float(rng.uniform(-0.10, 0.10))
            yaw += float(rng.uniform(-0.25, 0.25))
        robots[name] = (x, y, yaw)
    return {"parts": {k: (v[0], v[1]) for k, v in parts.items()}, "robots": robots, "facts": facts,
            "groove_gap": groove_gap}
