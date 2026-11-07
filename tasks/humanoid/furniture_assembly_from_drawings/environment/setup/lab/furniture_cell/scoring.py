"""The episode's assembly score (grader only).

Credit, out of furniture.TOTAL_WEIGHT = 9:
  * each of the six hole groups of the side panels (top, fixed shelf, bottom; left and right): 1.0 when a part of the
    right kind (a top/bottom panel in a top or bottom group, the fixed shelf in a shelf group) is LOCKED in it with its
    banded edge to the front of that side panel (front-preserving relative orientation, 3 deg);
  * the back panel LOCKED in the rabbets of both side panels with its white face to the front: 1.5 (one side panel's
    rabbet alone earns nothing);
  * placement, 1.5: all six dowel joints credited, and the carcass stands where the floor plan puts it (the left side
    panel within 30 mm and 3 deg of its place on the sideboard, upright within 3 deg).
A safety violation zeroes the episode (the grader applies it).
"""

from __future__ import annotations

import math

import numpy as np

from . import furniture as F
from . import room as RM

MM = 1e-3
ORIENT_COS = math.cos(math.radians(3.0))
PLACE_POS, PLACE_YAW, PLACE_TILT = 0.030, math.radians(3.0), math.radians(3.0)
GROUP_KIND = {"top": "C", "bottom": "C", "shelf": "D"}


def _front_ok(R_rel: np.ndarray) -> bool:
    """The guest's -y (front) axis maps onto the host's -y axis."""
    v = R_rel @ np.array([0.0, -1.0, 0.0])
    return float(v @ np.array([0.0, -1.0, 0.0])) >= ORIENT_COS


def placement_error(cell) -> dict:
    """The left side panel's pose against its place on the sideboard (target pose of the floor plan)."""
    p, R = cell.part_pose("L")
    tx, ty, tz, tyaw = RM.TARGET
    c, s = math.cos(tyaw), math.sin(tyaw)
    Rt = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    want = np.array([tx, ty, tz]) + Rt @ (np.array(F.PARTS["L"].design_center) * MM)
    dp = p - want
    yaw = math.atan2(R[1, 0], R[0, 0])
    tilt = math.acos(float(np.clip(R[2, 2], -1.0, 1.0)))
    dyaw = (yaw - tyaw + math.pi) % (2 * math.pi) - math.pi
    return {"dx": float(dp[0]), "dy": float(dp[1]), "dz": float(dp[2]), "dist_xy": float(math.hypot(dp[0], dp[1])),
            "dyaw_deg": math.degrees(dyaw), "tilt_deg": math.degrees(tilt)}


def score_cell(cell) -> dict:
    co = cell.coupling
    groups = {}
    for h in ("L", "R"):
        for name in ("top", "shelf", "bottom"):
            groups[f"{h}.{name}"] = {"credit": 0.0, "guest": None, "front_ok": None}
    k_front = {}                                    # side panel -> white face to the front, for each rabbet K is LOCKED in
    for (h, g), st in co.state.items():
        if st["state"] != "LOCKED":
            continue
        p_t, R_t = st["target"]
        if g == "K":
            k_front[h] = bool(_front_ok(R_t))
            continue
        hg = st["match"][0]
        kind_ok = F.PARTS[g].letter == GROUP_KIND[hg.split(".")[1]]
        ok = kind_ok and _front_ok(R_t)
        rec = groups[hg]
        rec.update(guest=g, front_ok=bool(_front_ok(R_t)), kind_ok=bool(kind_ok))
        if ok:
            rec["credit"] = 1.0
    k_ok = all(k_front.get(h) is True for h in ("L", "R"))
    k_credit = 1.5 if k_ok else 0.0
    k_rec = {"locked_in": sorted(k_front), "white_face_front": dict(k_front), "credited": k_ok} if k_front else None
    dowel = sum(r["credit"] for r in groups.values())
    place = placement_error(cell)
    placed = (dowel >= 6.0 - 1e-9 and place["dist_xy"] <= PLACE_POS and abs(place["dz"]) <= PLACE_POS
              and abs(place["dyaw_deg"]) <= math.degrees(PLACE_YAW) and place["tilt_deg"] <= math.degrees(PLACE_TILT))
    total = dowel + k_credit + (F.PLACEMENT_WEIGHT if placed else 0.0)
    return {
        "stage_score": total / F.TOTAL_WEIGHT,
        "credit": total,
        "groups": groups,
        "back_panel": k_rec,
        "placement": dict(place, ok=bool(placed)),
        "n_locked": sum(1 for st in co.state.values() if st["state"] == "LOCKED"),
        "n_engaged": sum(1 for st in co.state.values() if st["state"] == "ENGAGED"),
        "joint_events": [list(e[:4]) for e in co.events][:200],
        "grip_events": [list(e) for e in cell.grips.events][:400],
        "safety_ok": cell.safety.ok,
        "violations": list(cell.safety.violations),
        "safety_peaks": dict(cell.safety.peaks),
        "sim_time_s": cell.sim_time,
        "camera_frames": dict(cell.frames),
        "camera_frames_total": cell.frames_total,
        "render_wall_s": cell.render_wall,
        "done_reason": cell.done_reason,
    }
