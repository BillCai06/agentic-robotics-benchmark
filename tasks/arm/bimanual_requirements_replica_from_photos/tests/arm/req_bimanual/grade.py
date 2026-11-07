"""Criteria of the bimanual requirements task (public code, catalogue-free): gate G0 (structure) and the criteria C1-C5,
computed on the true pair with the grader's catalogue. Static: no submitted code is executed. Gate G1 (submission
integrity) belongs to the grader's entry point.

    gt = load_gt(gt_dir)                          pair_dynamics.xml, pair_model.json, layer.json, lot_truth.json,
                                                  key_dimensions.json, reference_sheet.json, optimum.json
    cat = catalog.Catalogue(path)                 the catalogue the BOM criteria use
    docs, problems = load_submission(sub, roles)  parse + schema + G0 rules (roles = {part_id: role} of the quote)
    out = grade(docs, gt, cat, cfg, acc)          {"criteria", "stats", "diagnostics", "overall"}

C1 kinematic_replication 0.30: lengths 0.55 (3 shared + 3 per arm), stops 0.30 (j1..j7 lower / upper per arm + each
gripper's open stop), gripper 0.10 (finger lever, grip stroke per arm), body classes 0.05 (16, exact); linear bands.
C2 requirements_sheet 0.10: facet F1 against the reference sheet (same quantity and comparator; a source outside the
reference item's clauses keeps clause_weight). C3 derived_requirements 0.20: the 47 derived items of the submitted BOM on
the true pair, relative error over max(|true|, floor of the unit), linear band; a missing item scores 0. C4
bom_acceptance 0.10: all 19 items of B-14. C5 bom_economy 0.30: repair.economy with the live optimum J* (in "check" mode
it must equal the stored one, else a task error), band = band_lines x eco.
"""
import json
import math
import os

import yaml

from . import optimum as OPT
from . import repair as RP
from . import schemas as S
from .layer import acc_layer, brief_constants
from .procedure import ARMS, ITEMS, JOINTS, MODULES, Acceptance, PairModel, bom_from_json

CRITERIA = ("kinematic_replication", "requirements_sheet", "derived_requirements", "bom_acceptance", "bom_economy")
DEFAULT_CFG = {
    "weights": {"kinematic_replication": 0.30, "requirements_sheet": 0.10, "derived_requirements": 0.20,
                "bom_acceptance": 0.10, "bom_economy": 0.30},
    "c1": {"group_weights": {"lengths": 0.55, "stops": 0.30, "gripper": 0.10, "body_classes": 0.05},
           "length_mm": [1.0, 6.0], "stop_deg": [1.0, 6.0], "gripper_mm": [0.5, 3.0]},
    "c2": {"brief_rel": [0.02, 0.04], "photo_rel": [0.01, 0.06], "photo_abs_m": [0.0015, 0.006], "angle_deg": [1.5, 6.0],
           "clause_weight": 0.5},
    "c3": {"rel": [0.03, 0.15], "floors": {"N.m": 0.05, "rad/s": 0.2, "A": 0.5, "kg": 0.02}},
}
FILES = {"requirements": "requirements.yaml", "bom": "bom.json", "kinematic_sketch": "kinematic_sketch.json"}
GT_JSON = ("pair_model", "layer", "lot_truth", "key_dimensions", "reference_sheet", "optimum")


def band(err, full, zero):
    if err <= full:
        return 1.0
    if err >= zero:
        return 0.0
    return (zero - err) / (zero - full)


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def load_gt(gt_dir):
    gt = {"dir": gt_dir}
    for k in GT_JSON:
        p = os.path.join(gt_dir, k + ".json")
        gt[k] = _load(p) if os.path.isfile(p) else None
    gt["xml_path"] = os.path.join(gt_dir, "pair_dynamics.xml")
    return gt


def acceptance_for(gt):
    return Acceptance(PairModel.from_files(gt["xml_path"], os.path.join(gt["dir"], "pair_model.json")),
                      acc_layer(gt["layer"]))


# ------------------------------------------------------------------------------------------------ submission + G0
def load_submission(sub_dir, part_roles):
    docs, problems = {}, []
    for k, f in FILES.items():
        p = os.path.join(sub_dir, f)
        if not os.path.isfile(p) or os.path.islink(p):
            problems.append("%s missing" % f)
            continue
        try:
            with open(p) as fh:
                txt = fh.read(2_000_000)
            docs[k] = yaml.safe_load(txt) if f.endswith(".yaml") else json.loads(txt)
        except Exception as e:  # noqa: BLE001
            problems.append("%s does not parse: %s" % (f, str(e)[:200]))
            continue
        if not isinstance(docs[k], dict):
            problems.append("%s is not a mapping" % f)
            docs.pop(k)
            continue
        errs = S.validate(docs[k], k)
        if errs:
            problems += ["%s: %s" % (f, e) for e in errs[:8]]
            docs.pop(k)
    if problems:
        return docs, problems
    return docs, g0_rules(docs, part_roles)


def _finite(*xs):
    try:
        return all(math.isfinite(float(x)) for x in xs)
    except (OverflowError, TypeError, ValueError):
        return False


def g0_rules(docs, part_roles):
    """Structure rules of gate G0; `part_roles` = {part_id: role} of the quote."""
    p = []
    req = docs["requirements"]["requirements"]
    ids = [r["id"] for r in req]
    if len(set(ids)) != len(ids):
        p.append("requirement ids are not unique")
    for r in req:
        conv = S.to_si(r["bound"], r["unit"])
        if conv is None or conv[1] != S.QUANTITY_SI_UNIT[r["quantity"]]:
            p.append("%s: unit %r is not a unit of %s" % (r["id"], r["unit"], r["quantity"]))
        if not _finite(r["bound"]):
            p.append("%s: bound not finite" % r["id"])
    bom = docs["bom"]
    lines, psu = {}, []
    for line in bom["parts"]:
        pid = line["part_id"]
        if pid not in part_roles:
            p.append("bom part %r is not in the quote" % pid)
            continue
        if part_roles[pid] != line["role"]:
            p.append("bom part %r has role %r, the quote says %r" % (pid, line["role"], part_roles[pid]))
        if line["role"] == "module":
            key = (line.get("arm"), line.get("joint"))
            if None in key:
                p.append("module line %r without arm and joint" % pid)
                continue
            if key in lines:
                p.append("two module lines for %s %s" % key)
            lines[key] = line
            if int(line["count"]) != 1:
                p.append("module line of %s %s has count %s" % (key[0], key[1], line["count"]))
        elif line["role"] == "psu":
            psu.append(line)
    for a in ARMS:
        for m in MODULES:
            if (a, m) not in lines:
                p.append("no module line for %s %s" % (a, m))
    if len(psu) != 1 or int(psu[0]["count"]) != 1:
        p.append("exactly one psu line with count 1 is required")
    if float(bom["bus_voltage_v"]) not in [float(v) for v in S.BUS_VOLTAGES]:
        p.append("bus_voltage_v must be one of %s" % list(S.BUS_VOLTAGES))
    sk = docs["kinematic_sketch"]
    g = sk["geometry"]
    for a in ARMS:
        for m in MODULES:
            sid = sk["joints"][a][m]["module_id"]
            if (a, m) in lines and lines[(a, m)]["part_id"] != sid:
                p.append("sketch module_id of %s %s (%s) is not the BOM line (%s)" % (a, m, sid, lines[(a, m)]["part_id"]))
            if not _finite(*sk["joint_ranges"][a][m]):
                p.append("joint range of %s %s not finite" % (a, m))
        for k in S.ARM_KEYS:
            if not _finite(g[a][k]["nominal"], g[a][k]["tol"]):
                p.append("geometry %s %s not finite" % (a, k))
    for k in S.SHARED_KEYS:
        if not _finite(g[k]["nominal"], g[k]["tol"]):
            p.append("geometry %s not finite" % k)
    return p


# ------------------------------------------------------------------------------------------------ criteria
def c1_kinematic(sk, kd, cfg):
    c = cfg["c1"]
    g = sk["geometry"]
    items, groups = {}, {"lengths": [], "stops": [], "gripper": [], "body_classes": []}
    for k in S.SHARED_KEYS:
        err = 1000.0 * abs(float(g[k]["nominal"]) - float(kd["shared"][k]))
        items["len_err_mm:" + k] = err
        groups["lengths"].append(band(err, *c["length_mm"]))
    for a in ARMS:
        for k in S.ARM_KEYS:
            err = 1000.0 * abs(float(g[a][k]["nominal"]) - float(kd["arms"][a][k]))
            grp, b = ("lengths", c["length_mm"]) if k in S.ARM_LENGTH_KEYS else ("gripper", c["gripper_mm"])
            items[("len_err_mm:" if grp == "lengths" else "grip_err_mm:") + "%s_%s" % (a, k)] = err
            groups[grp].append(band(err, *b))
        for j in JOINTS:
            s_ = sorted(float(x) for x in sk["joint_ranges"][a][j])
            t_ = sorted(float(x) for x in kd["stops_rad"][a][j])
            for side, si, ti in (("lower", s_[0], t_[0]), ("upper", s_[1], t_[1])):
                err = abs(math.degrees(si - ti))
                items["stop_err_deg:%s_%s:%s" % (a, j, side)] = err
                groups["stops"].append(band(err, *c["stop_deg"]))
        err = abs(math.degrees(max(float(x) for x in sk["joint_ranges"][a]["j8"]) - float(kd["stops_rad"][a]["j8"][1])))
        items["stop_err_deg:%s_j8:open" % a] = err
        groups["stops"].append(band(err, *c["stop_deg"]))
        for m in MODULES:
            groups["body_classes"].append(1.0 if sk["joints"][a][m]["body_class"] == kd["pocket_class"][a][m] else 0.0)
    means = {k: sum(v) / len(v) for k, v in groups.items()}
    w = c["group_weights"]
    cov, tols = [], []
    for k, nom, tol in ([(k, g[k]["nominal"], g[k]["tol"]) for k in S.SHARED_KEYS]
                        + [("%s_%s" % (a, k), g[a][k]["nominal"], g[a][k]["tol"]) for a in ARMS for k in S.ARM_KEYS]):
        truth = kd["shared"][k] if k in S.SHARED_KEYS else kd["arms"][k.split("_", 1)[0]][k.split("_", 1)[1]]
        cov.append(1.0 if abs(float(nom) - float(truth)) <= float(tol) + 1e-12 else 0.0)
        tols.append(1000.0 * float(tol))
    items["tol_coverage"] = sum(cov) / len(cov)
    items["tol_mean_mm"] = sum(tols) / len(tols)
    return sum(w[k] * means[k] for k in w) / sum(w.values()), means, items


def _item_band(ref, cfg):
    """(mode, (full, zero)) of a reference item: angles in degrees; brief items relative; photo lengths relative with an
    absolute floor (full / zero at the larger of the relative band x |reference| and photo_abs_m), in metres."""
    kind = ref.get("band", "brief")
    if kind == "angle":
        return "abs_deg", cfg["c2"]["angle_deg"]
    if kind == "photo" and cfg["c2"].get("photo_abs_m"):
        b = abs(float(ref["bound"]))
        (rf, rz), (af, az) = cfg["c2"]["photo_rel"], cfg["c2"]["photo_abs_m"]
        return "abs", (max(rf * b, af), max(rz * b, az))
    return "rel", cfg["c2"]["photo_rel" if kind == "photo" else "brief_rel"]


COMPARATOR_NORM = {"<=": "<=", "<": "<=", "=<": "<=", ">=": ">=", ">": ">=", "=>": ">=", "==": "==", "=": "=="}


def _cmp(c):
    return COMPARATOR_NORM.get(str(c).strip())


def c2_sheet(items, rstar, cfg):
    """F1 between the submitted non-derived items and the reference sheet: an item matches a reference item with the same
    quantity AND the same comparator (strict and non-strict forms are one comparator), scored by its bound; a match whose
    source_id is outside the reference item's source ids keeps clause_weight of its score."""
    agent = [r for r in items if str(r.get("source_id")) != "derived" and r["quantity"] not in S.DERIVED_QUANTITIES]
    ref = rstar["items"]
    used, total = set(), 0.0
    rep = {"matched": [], "partial": [], "missing": [], "extra": [], "clause_mismatch": []}
    for rf in ref:
        mode, (full, zero) = _item_band(rf, cfg)
        best, best_i, best_clause = 0.0, None, True
        for i, it in enumerate(agent):
            if i in used or it["quantity"] != rf["quantity"]:
                continue
            if _cmp(it.get("comparator", "")) != _cmp(rf["comparator"]):
                continue
            conv = S.to_si(it["bound"], it["unit"])
            if conv is None or conv[1] != S.QUANTITY_SI_UNIT[rf["quantity"]]:
                continue
            if mode == "abs_deg":
                err = abs(math.degrees(conv[0] - float(rf["bound"])))
            elif mode == "abs":
                err = abs(conv[0] - float(rf["bound"]))
            else:
                err = abs(conv[0] - float(rf["bound"])) / max(abs(float(rf["bound"])), 1e-9)
            w = band(err, full, zero)
            clause_ok = str(it.get("source_id", "")) in rf.get("source_ids", [])
            if not clause_ok:
                w *= cfg["c2"]["clause_weight"]
            if w > best:
                best, best_i, best_clause = w, i, clause_ok
        if best_i is not None:
            used.add(best_i)
        total += best
        if best_i is None or best <= 0:
            rep["missing"].append(rf["id"])
        elif best >= 1 - 1e-9:
            rep["matched"].append(rf["id"])
        else:
            rep["partial"].append(rf["id"])
        if best_i is not None and not best_clause:
            rep["clause_mismatch"].append(rf["id"])
    rep["extra"] = [str(agent[i].get("id")) for i in range(len(agent)) if i not in used]
    if not agent or not ref or total <= 0:
        return 0.0, rep
    P, R = total / len(agent), total / len(ref)
    return 2 * P * R / (P + R), rep


def c3_derived(items, truth, cfg):
    first = {}
    for it in items:
        if it["quantity"] in S.DERIVED_QUANTITIES and it["quantity"] not in first:
            first[it["quantity"]] = it
    full, zero = cfg["c3"]["rel"]
    scores, errs = [], {}
    for q in S.DERIVED_QUANTITIES:
        unit = S.QUANTITY_SI_UNIT[q]
        it = first.get(q)
        conv = None if it is None else S.to_si(it["bound"], it["unit"])
        if conv is None or conv[1] != unit:
            scores.append(0.0)
            errs[q] = None
            continue
        g = float(truth[q])
        e = abs(conv[0] - g) / max(abs(g), float(cfg["c3"]["floors"][unit]))
        errs[q] = e
        scores.append(band(e, full, zero))
    return sum(scores) / len(scores), errs


def grade(docs, gt, cat, cfg=None, acc=None, optimum_mode="check"):
    """optimum_mode "check": the live optimum over `cat` must equal gt optimum.json (else RuntimeError, a task error);
    "stored": J* = the stored optimum (development only)."""
    cfg = cfg or DEFAULT_CFG
    out = {"criteria": {c: 0.0 for c in CRITERIA}, "stats": {}, "diagnostics": {}}
    w = cfg["weights"]

    def finish():
        out["overall"] = sum(w[c] * out["criteria"][c] for c in CRITERIA) / sum(w[c] for c in CRITERIA)
        return out

    acc = acc or acceptance_for(gt)
    lot, brief, lay = gt["lot_truth"], brief_constants(gt["layer"]), gt["layer"]
    items = docs["requirements"]["requirements"]
    c1, groups, c1i = c1_kinematic(docs["kinematic_sketch"], gt["key_dimensions"], cfg)
    out["criteria"]["kinematic_replication"] = c1
    out["stats"].update({"c1_" + k: v for k, v in groups.items()})
    out["stats"].update(c1i)
    c2, rep = c2_sheet(items, gt["reference_sheet"], cfg)
    out["criteria"]["requirements_sheet"] = c2
    out["stats"].update({"c2_" + k: float(len(v)) for k, v in rep.items()})
    out["diagnostics"]["requirements_sheet"] = "missing %s; partial %s; extra %s" % (rep["missing"], rep["partial"],
                                                                                      rep["extra"][:10])
    bom = bom_from_json(docs["bom"])
    parts = [p for arm in bom["modules"].values() for p in arm.values()] + bom["boards"] + [bom["psu"]]
    missing = sorted({str(p) for p in parts if p is None or not cat.has(p)})
    if missing:
        out["diagnostics"]["bom"] = "catalogue lacks %s: the BOM criteria are 0" % missing
        return finish()
    truth = acc.derived(bom, lot, cat, brief)
    c3, errs = c3_derived(items, truth, cfg)
    out["criteria"]["derived_requirements"] = c3
    out["stats"].update({"c3_relerr:" + q: (-1.0 if e is None else e) for q, e in errs.items()})
    worst = sorted(((e, q) for q, e in errs.items() if e is not None), reverse=True)[:6]
    out["diagnostics"]["derived_requirements"] = "missing %d; largest errors %s" % (
        sum(e is None for e in errs.values()), ", ".join("%s %.3f" % (q, e) for e, q in worst))
    ev = acc.evaluate(bom, lot, cat, brief)
    out["criteria"]["bom_acceptance"] = 1.0 if ev["accepted"] else 0.0
    out["stats"]["c4_item_fraction"] = ev["pass_fraction"]
    out["stats"].update({"c4_pass:" + i: (1.0 if ev["items"][i] else 0.0) for i in ITEMS})
    # stats stay finite (the verdict record refuses inf): -1 = an infinite utilisation (a required speed >= w0)
    out["stats"].update({"util:" + i: float(u["worst"]) if math.isfinite(u["worst"]) else -1.0
                         for i, u in ev["util"].items()})
    out["stats"].update({"peak_current_a": ev["peak_current_a"], "bom_cost_usd": ev["cost"],
                         "bus_voltage": float(bom["bus_voltage_v"])})
    out["diagnostics"]["bom_acceptance"] = ", ".join(
        "%s FAIL%s" % (i, " (u %.3f)" % ev["util"][i]["worst"] if i in ev["util"] else "")
        for i in ITEMS if not ev["items"][i]) or "all 19 items pass"
    stored = float(gt["optimum"]["cost"])
    if optimum_mode == "check":
        best = OPT.optimal_bom(acc, lot, cat, brief)
        if best is None or abs(best["cost"] - stored) > 1e-6:
            raise RuntimeError("live optimum %s != stored optimum %s (task error)" % (
                None if best is None else best["cost"], stored))
        J_star = best["cost"]
    else:
        J_star = stored
    rp = RP.repair(acc, bom, lot, cat, brief)
    ceiling = float(acc.layer["ceiling"])
    band_usd = float(brief["band_lines"]) * float(brief["eco"])
    out["criteria"]["bom_economy"] = RP.economy(J_star, rp["J_eff"], ceiling, band_usd)
    fin = math.isfinite(rp["J_eff"])
    out["stats"].update({"J": rp["J"], "J_eff": rp["J_eff"] if fin else -1.0, "J_star": J_star, "repairs": rp["replaced"],
                         "repairable": 1.0 if rp["repairable"] else 0.0,
                         "ceiling": ceiling if math.isfinite(ceiling) else -1.0, "c5_band_usd": band_usd,
                         "c5_over_usd": (rp["J_eff"] - J_star) if fin else -1.0,
                         "optimum_beaten": 1.0 if (fin and rp["J_eff"] < J_star - 1e-9 and ev["accepted"]) else 0.0})
    out["diagnostics"]["bom_economy"] = ("; ".join(rp["log"]) or "no engineering change")[:1500]
    if lay.get("brief", {}).get("ceiling") is None:
        out["diagnostics"]["bom_economy"] += " (the layer has no ceiling)"
    return finish()
