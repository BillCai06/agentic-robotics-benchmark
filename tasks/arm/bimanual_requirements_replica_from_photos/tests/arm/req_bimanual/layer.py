"""The keys of an instance layer (layer.json, format bimanual-req-layer/1) that the acceptance procedure reads (public
code; the numbers live in the instance's layer.json, whose "brief" block carries the customer's requirements and the
brief / build-log constants of the formulas)."""
import math

ACC_KEYS = ("T", "H", "m_hi", "com_drop", "proof", "mu", "slip_sf", "close_s")
BRIEF_KEYS = ("harness_drop_v", "untested_cap", "p_quantile_z", "bus_voltages_v", "eco", "band_lines")


def acc_layer(lay):
    """The duty of B-14: the tray area and heights, the handle spacing, the grasp, the motion, the payload, the grip, the
    ceiling (math.inf while an instance draw has not set one)."""
    a, b = lay["area"], lay["brief"]
    out = {k: float(a[k]) for k in ("x_min", "x_max", "y_min", "y_max")}
    out.update({k: float(lay[k]) for k in ("z_lo", "z_hi", "handle_spacing_m", "approach_pitch_deg", "swivel_deg")})
    out.update({k: float(b[k]) for k in ACC_KEYS})
    out["ceiling"] = math.inf if b.get("ceiling") is None else float(b["ceiling"])
    return out


def brief_constants(lay):
    """The formula constants of the brief and the build log: harness drop (BL-03), the untested-vendor cap (B-10), the
    weakest-unit quantile (B-14 e), the bus voltages (B-11), the engineering-change cost (B-13) and the economy band in
    lines (C5)."""
    b = lay["brief"]
    return {k: b[k] for k in BRIEF_KEYS}
