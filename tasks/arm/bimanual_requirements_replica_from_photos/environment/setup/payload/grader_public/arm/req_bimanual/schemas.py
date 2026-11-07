"""Deliverable schemas of the bimanual requirements task (public): requirements.yaml (req-bimanual/1), bom.json
(bom-bimanual/1), kinematic_sketch.json (sketch-bimanual/1), the unit table and the requirement vocabulary (with the 47
derived quantities of brief B-16), plus the schema of a catalogue TRANSCRIPTION (bimanual-req-catalogue/1: an agent's own
reading of the datasheet pages, the erratum and the quote, in the format catalog.Catalogue loads).

    write_schemas(out_dir)             the four JSON schema files
    validate(doc, which)               -> list of problems (jsonschema); which in SCHEMAS
    to_si(value, unit)                 -> (value_si, si_unit) or None
"""
import json
import math
import os

ARMS = ("left", "right")
JOINTS = tuple("j%d" % k for k in range(1, 8))
MODULES = JOINTS + ("j8",)
BODY_CLASSES = ("large", "medium", "small")
BUS_VOLTAGES = (24, 48)
STANDARD_GRAVITY = 9.80665
SOURCE_PATTERN = r"^(B|BL)-[0-9]{2}$|^PH-[0-9]{2}$|^DS-[A-Za-z0-9-]+$|^LR(-[0-9]{4}-[0-9]{4})?$|^derived$"

UNIT_TO_SI = {
    "m": ("m", 1.0), "mm": ("m", 1e-3), "cm": ("m", 1e-2),
    "rad": ("rad", 1.0), "deg": ("rad", math.pi / 180.0),
    "kg": ("kg", 1.0), "g": ("kg", 1e-3),
    "s": ("s", 1.0), "ms": ("s", 1e-3),
    "N.m": ("N.m", 1.0), "N m": ("N.m", 1.0), "Nm": ("N.m", 1.0), "kgf.cm": ("N.m", STANDARD_GRAVITY * 1e-2),
    "rad/s": ("rad/s", 1.0), "rpm": ("rad/s", math.pi / 30.0), "deg/s": ("rad/s", math.pi / 180.0),
    "A": ("A", 1.0), "mA": ("A", 1e-3), "V": ("V", 1.0), "W": ("W", 1.0),
    "USD": ("USD", 1.0), "count": ("count", 1.0), "fraction": ("fraction", 1.0),
}

QUANTITY_SI_UNIT = {
    "payload_mass_max": "kg", "payload_mass_min": "kg", "payload_com_drop": "m",
    "tray_x_min": "m", "tray_x_max": "m", "tray_y_min": "m", "tray_y_max": "m",
    "tray_height_min": "m", "tray_height_max": "m", "handle_spacing": "m",
    "approach_pitch": "rad", "elbow_swivel": "rad", "move_time": "s", "dwell_time": "s",
    "proof_load_factor": "fraction", "pad_friction": "fraction", "slip_safety_factor": "fraction",
    "gripper_close_time": "s", "gripper_opening_min": "m", "psu_count": "count", "module_count": "count",
    "revolute_dof": "count", "total_cost": "USD", "eco_cost": "USD", "unit_count": "count", "bus_voltage": "V",
    "untested_vendor_derating": "fraction", "harness_drop": "V",
}
DERIVED_QUANTITIES = tuple(
    ["joint_peak_torque:%s_%s" % (a, j) for a in ARMS for j in JOINTS]
    + ["joint_rms_torque:%s_%s" % (a, j) for a in ARMS for j in JOINTS]
    + ["joint_peak_speed:%s_%s" % (a, j) for a in ARMS for j in JOINTS]
    + ["grip_torque:%s" % a for a in ARMS] + ["moving_mass:%s" % a for a in ARMS] + ["supply_peak_current"])
for _q in DERIVED_QUANTITIES:
    QUANTITY_SI_UNIT[_q] = ("rad/s" if _q.startswith("joint_peak_speed") else "kg" if _q.startswith("moving_mass")
                            else "A" if _q == "supply_peak_current" else "N.m")

SHARED_KEYS = ("shoulder_height", "shoulder_half_span", "shoulder_forward_offset")
ARM_LENGTH_KEYS = ("upper_arm", "forearm", "wrist_to_tcp")
GRIPPER_KEYS = ("finger_lever", "grip_stroke")
ARM_KEYS = ARM_LENGTH_KEYS + GRIPPER_KEYS


def to_si(value, unit):
    if unit not in UNIT_TO_SI:
        return None
    c, f = UNIT_TO_SI[unit]
    try:
        return float(value) * f, c
    except (OverflowError, TypeError, ValueError):
        return None


def requirements_schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "req-bimanual/1 requirements sheet",
        "description": ("Traced requirements of the replica pairs: one item per quantity the brief, the build log, the "
                        "photos or the documents fix, plus the derived items of brief B-16 (source_id 'derived' with a "
                        "derivation note). Bounds in the listed units; every item traced to its source. Comparator "
                        "convention: a duty quantity is bounded the way the brief bounds the duty (for example "
                        "payload_mass_max <= the band top, payload_mass_min >= the band bottom, move_time <= the move "
                        "time); a working-range quantity (tray_x_min, tray_x_max, tray_y_min, tray_y_max, tray_height_min, "
                        "tray_height_max, gripper_opening_min) states what the replica must cover: a *_min with <= the "
                        "lowest value the duty needs, a *_max with >= the highest."),
        "type": "object",
        "required": ["requirements"],
        "properties": {
            "format": {"type": "string"}, "project": {"type": "string"}, "units": {"type": "object"},
            "notes": {"type": "string"},
            "requirements": {"type": "array", "minItems": 1, "items": {
                "type": "object", "required": ["id", "quantity", "comparator", "bound", "unit", "source_id"],
                "properties": {
                    "id": {"type": "string", "pattern": "^REQ-[A-Za-z0-9_.-]+$"},
                    "quantity": {"type": "string", "enum": sorted(QUANTITY_SI_UNIT)},
                    "comparator": {"type": "string", "enum": [">=", "<=", "==", ">", "<"]},
                    "bound": {"type": "number", "minimum": -1e6, "maximum": 1e6},
                    "unit": {"type": "string", "enum": sorted(UNIT_TO_SI)},
                    "source_id": {"type": "string", "pattern": SOURCE_PATTERN},
                    "source_clause": {"type": "string"},
                    "derivation": {"type": "string"}},
                "additionalProperties": True}}},
        "additionalProperties": True}


def bom_schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "bom-bimanual/1 bill of materials (per replica pair)",
        "description": ("One module line per arm and joint (j1..j7 and the gripper j8, count 1), the bus board line(s), "
                        "exactly one psu line (count 1); part_id and role as in the quote."),
        "type": "object",
        "required": ["bus_voltage_v", "parts"],
        "properties": {
            "format": {"type": "string"},
            "bus_voltage_v": {"type": "number", "enum": list(BUS_VOLTAGES)},
            "total_cost_usd": {"type": "number"},
            "parts": {"type": "array", "minItems": 2 * len(MODULES) + 2, "items": {
                "type": "object", "required": ["part_id", "count", "role"],
                "properties": {"part_id": {"type": "string"}, "count": {"type": "integer", "minimum": 1, "maximum": 8},
                               "role": {"type": "string", "enum": ["module", "bus_board", "psu"]},
                               "arm": {"type": "string", "enum": list(ARMS)},
                               "joint": {"type": "string", "enum": list(MODULES)}, "note": {"type": "string"}},
                "additionalProperties": True}},
            "notes": {"type": "string"}},
        "additionalProperties": True}


def _nt():
    return {"type": "object", "required": ["nominal", "tol"], "additionalProperties": True,
            "properties": {"nominal": {"type": "number"}, "tol": {"type": "number", "minimum": 0}}}


def _per_arm(item):
    return {"type": "object", "required": list(ARMS), "properties": {a: item for a in ARMS}, "additionalProperties": False}


def sketch_schema():
    rng = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
    arm_geom = {"type": "object", "required": list(ARM_KEYS), "properties": {k: _nt() for k in ARM_KEYS},
                "additionalProperties": True}
    ranges = {"type": "object", "required": list(MODULES), "properties": {m: rng for m in MODULES},
              "additionalProperties": False}
    joints = {"type": "object", "required": list(MODULES), "additionalProperties": False, "properties": {
        m: {"type": "object", "required": ["module_id", "body_class"],
            "properties": {"module_id": {"type": "string"}, "body_class": {"type": "string", "enum": list(BODY_CLASSES)}},
            "additionalProperties": True} for m in MODULES}}
    geom = {"type": "object", "required": list(SHARED_KEYS) + list(ARMS), "additionalProperties": True,
            "properties": dict({k: _nt() for k in SHARED_KEYS}, **{a: arm_geom for a in ARMS})}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "sketch-bimanual/1 kinematic sketch of the replica pair",
        "description": ("Appendix-D quantities (shared and per arm, nominal + tolerance), the stops of every joint of both "
                        "arms in the Appendix-D conventions (j8 = the gripper's [0, open angle]) and, per arm and joint, "
                        "the BOM's module and the pocket's housing class. Lengths in m, angles in rad."),
        "type": "object",
        "required": ["units", "geometry", "joint_ranges", "joints"],
        "properties": {
            "format": {"type": "string"},
            "units": {"type": "object", "required": ["length", "angle"],
                      "properties": {"length": {"const": "m"}, "angle": {"const": "rad"}}},
            "geometry": geom, "joint_ranges": _per_arm(ranges), "joints": _per_arm(joints)},
        "additionalProperties": True}


def catalogue_schema():
    num = {"type": "number"}
    rng = {"type": "array", "items": num, "minItems": 2, "maxItems": 2}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "bimanual-req-catalogue/1 transcription of the vendor pages and the quote (SI units)",
        "description": ("A reading of datasheets/DS-*.png (with the erratum applied) and price_list.md in SI units: "
                        "torques N.m, speeds rad/s, currents A, voltages V, power W, inertia kg m2, masses kg, prices USD; "
                        "lists in the quote's order; a vendor with an incoming-inspection report is inspected."),
        "type": "object",
        "required": ["modules", "boards", "psus", "vendors"],
        "properties": {
            "format": {"type": "string"},
            "vendors": {"type": "object", "additionalProperties": {
                "type": "object", "required": ["inspected"], "properties": {"inspected": {"type": "boolean"}}}},
            "modules": {"type": "array", "items": {
                "type": "object",
                "required": ["part_id", "vendor", "housing", "protocol", "voltage_range_v", "points", "peak_torque_nm",
                             "rated_torque_nm", "idle_current_a", "copper_loss_rated_w", "output_inertia_kgm2", "mass_kg",
                             "price_usd"],
                "properties": {
                    "part_id": {"type": "string"}, "vendor": {"type": "string"},
                    "housing": {"type": "string", "enum": list(BODY_CLASSES)}, "protocol": {"type": "string"},
                    "voltage_range_v": rng,
                    "points": {"type": "array", "minItems": 1, "maxItems": 2, "items": {
                        "type": "object", "required": ["voltage_v", "no_load_speed_rad_s", "speed_at_peak_rad_s"],
                        "properties": {"voltage_v": num, "no_load_speed_rad_s": num, "speed_at_peak_rad_s": num}}},
                    "peak_torque_nm": num, "rated_torque_nm": num, "idle_current_a": num, "copper_loss_rated_w": num,
                    "output_inertia_kgm2": num, "mass_kg": num, "price_usd": num},
                "additionalProperties": True}},
            "boards": {"type": "array", "items": {
                "type": "object",
                "required": ["part_id", "protocols", "channels", "modules_per_channel", "input_range_v", "price_usd"],
                "properties": {"part_id": {"type": "string"}, "protocols": {"type": "array", "items": {"type": "string"}},
                               "channels": {"type": "integer", "minimum": 1},
                               "modules_per_channel": {"type": "integer", "minimum": 1}, "input_range_v": rng,
                               "price_usd": num}, "additionalProperties": True}},
            "psus": {"type": "array", "items": {
                "type": "object", "required": ["part_id", "output_v", "rated_current_a", "price_usd"],
                "properties": {"part_id": {"type": "string"}, "output_v": num, "rated_current_a": num, "price_usd": num},
                "additionalProperties": True}}},
        "additionalProperties": True}


SCHEMAS = {"requirements": requirements_schema, "bom": bom_schema, "kinematic_sketch": sketch_schema,
           "catalogue": catalogue_schema}


def write_schemas(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for k, f in SCHEMAS.items():
        p = os.path.join(out_dir, "%s.schema.json" % k)
        with open(p, "w") as fh:
            json.dump(f(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        out[k] = p
    return out


def validate(doc, which):
    import jsonschema
    v = jsonschema.Draft202012Validator(SCHEMAS[which]())
    return ["%s: %s" % ("/".join(str(x) for x in e.absolute_path) or "(root)", e.message)
            for e in sorted(v.iter_errors(doc), key=lambda e: list(e.absolute_path))]
