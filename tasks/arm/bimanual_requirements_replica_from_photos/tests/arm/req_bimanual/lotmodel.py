"""Capability of the delivered lot's weakest expected unit (brief B-14 e, B-10) under a GIVEN lot truth and a GIVEN
catalogue (public formula code: no catalogue, lot or price number in it).

Lot truth (format bimanual-req-lot-truth/1): {"delivered_lot": key, "lots": {key: {"lot_id", "report", "date", "vendor",
"f_peak", "f_speed", "f_rated", "sigma", "droop", "tested_skus", "report_seed"}}, "sku_factor": {part_id: e}}.

Capability of SKU s on a bus at V (the module sees V_s = V - harness_drop_v), q = 1 - z sigma with z = p_quantile_z:
  inspected vendor:   tau_s = peak f_peak (1 + e) q,  tau_c = rated f_rated (1 + e) q,  w0 = no_load(V_s) f_speed q,
                      wk = speed_at_peak(V_s) f_speed q,  droop d = the delivered lot's
  uninspected vendor: tau_s, tau_c, w0, wk = untested_cap x the datasheet values, d = 0 (the straight-line envelope)
Torque available at speed |w|: tau_s for |w| <= wk; tau_s u* for wk < |w| < w0, with x = (|w| - wk) / (w0 - wk) and u*
the root in [0, 1] of (1 - u)(1 - d u) = x; 0 for |w| >= w0.
Brief constants (layer.json "brief"): harness_drop_v, untested_cap, p_quantile_z.
"""
import numpy as np


def droop_fraction(x, d):
    """u* in [0, 1] with (1 - u)(1 - d u) = x (the smaller root, in a form that is stable for d -> 0)."""
    x = min(max(float(x), 0.0), 1.0)
    b = 1.0 + float(d)
    return 2.0 * (1.0 - x) / (b + max(b * b - 4.0 * float(d) * (1.0 - x), 0.0) ** 0.5)


def droop_fraction_array(x, d):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    b = 1.0 + float(d)
    return 2.0 * (1.0 - x) / (b + np.sqrt(np.maximum(b * b - 4.0 * float(d) * (1.0 - x), 0.0)))


def available_torque(w, cap):
    """The torque available at speed w (rad/s) of a capability (unit_capability)."""
    a = abs(float(w))
    if a <= cap["wk"]:
        return cap["tau_s"]
    if a >= cap["w0"]:
        return 0.0
    return cap["tau_s"] * droop_fraction((a - cap["wk"]) / (cap["w0"] - cap["wk"]), cap["droop"])


def available_torque_array(w, cap):
    a = np.abs(np.asarray(w, float))
    span = cap["w0"] - cap["wk"]
    x = (a - cap["wk"]) / span if span > 0 else np.ones_like(a)
    mid = cap["tau_s"] * droop_fraction_array(x, cap["droop"])
    return np.where(a <= cap["wk"], cap["tau_s"], np.where(a >= cap["w0"], 0.0, mid))


def unit_capability(pid, bus_v, lot, cat, brief, z=None):
    """SI capability of a unit of module `pid` on a bus at `bus_v`. Default: the delivered lot's weakest expected unit;
    `z` = an explicit standard-normal draw for all three unit factors (statistics). Returns {tau_s, tau_c, w0, wk, droop,
    idle_a, cu_loss_w, rated_nm (datasheet), J_out, mass_kg, v_servo, housing, protocol, inspected}."""
    m = cat.module(pid)
    v_s = float(bus_v) - float(brief["harness_drop_v"])
    w0, wk = cat.speeds_at(m, v_s)
    if not m["inspected"]:
        k = float(brief["untested_cap"])
        tau_s, tau_c, w0, wk, d = k * m["peak_nm"], k * m["rated_nm"], k * w0, k * wk, 0.0
    else:
        L = lot["lots"][lot["delivered_lot"]]
        e = float(lot["sku_factor"].get(pid, 0.0))
        q = 1.0 - float(brief["p_quantile_z"]) * float(L["sigma"]) if z is None else 1.0 + float(z) * float(L["sigma"])
        tau_s = m["peak_nm"] * float(L["f_peak"]) * (1.0 + e) * q
        tau_c = m["rated_nm"] * float(L["f_rated"]) * (1.0 + e) * q
        w0 = w0 * float(L["f_speed"]) * q
        wk = wk * float(L["f_speed"]) * q
        d = float(L["droop"])
    return {"tau_s": tau_s, "tau_c": tau_c, "w0": w0, "wk": wk, "droop": d, "idle_a": m["idle_a"],
            "cu_loss_w": m["cu_loss_w"], "rated_nm": m["rated_nm"], "J_out": m["J_out"], "mass_kg": m["mass_kg"],
            "v_servo": v_s, "housing": m["housing"], "protocol": m["protocol"], "inspected": m["inspected"]}
