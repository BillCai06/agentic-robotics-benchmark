"""The grader of a unit with a kinematic tree and closed loops (first user: bimanual_cad_design_from_photos): a submitted
unit (cad_tree.convert's output) against the reference GT folder, per arm. Lengths mm, angles rad unless named.

    classes = C.load_classes(bought_dir)
    ref = load_reference(ref_dir, classes)                         the GT folder (CONTRACTS 6) and its operating box
    rewards, stats, details = grade(conv_dir, ref, seed_seq)       seed_seq: numpy SeedSequence (verify: OS entropy)

Checks. Unit-wide gates (rules.json gates.unit): cad_valid (the conversion found the unit: every tag, the bought counts,
a grounded torso with the shared parts and the column; T01 loads as a valid watertight solid) and public_model_reuse (no
part's vertex fingerprint in the public blocklist; no loose blocklisted file in the submission). Arm gates
(gates.arm): cad_json_consistent (the arm's joint axes lie on their motors' axes, the finger axes on the module's hinges,
the pins parallel to the crank, each joint's frames coincident at the saved pose: 1 deg / 0.5 mm). An arm whose
mechanism differs from the reference's (cad_tree.convert), or one of whose parts does not load as a valid watertight
solid, scores 0 on its items; the other arm is still scored (a part fixed to another link and a motor case on the other
side of its joint are reported in the arm's details under rules.json link_membership / case_side, and graded as built).

Criteria, per arm (the arm's links and joints; its parts with the shared base plate T01):
  kinematic_fidelity  axes: each motor / linkage joint's axis line against the reference's (the distance of the
                      reference anchor from the submitted line, the angle); key lengths (rules.json key_lengths); the tool
                      point and both fingertips over fk_configs draws of the arm's tree joints inside the reference box
                      (tool point median and 90th percentile, fingertip medians); the declared limits against the box
                      (rules.json tolerances.range_metric: "iou" = overlap / union, "coverage" = overlap / the box)
  feature_fidelity    gt.json cut-outs and spacers on the arm's parts (cad_tree.features), each part's extents
  shape_fidelity      per part after a rigid alignment onto the reference part: 90th-percentile symmetric surface distance
                      and volume IoU
  mass_properties     per link: mass (part volume x the reference parts.json density + the bought masses) and the CoM at Z
  component_fit       per motor: keep-out x case-side retention x output-side retention (rules.json fit); a motor of
                      another type than the unit's at that joint scores 0
  interference_free   the arm against itself and the torso over draws inside the box where the REFERENCE is collision-
                      free (the other arm at Z), and the two arms against each other over draws of both arms (where the
                      reference arms do not touch); per draw the worst pair (score_ratio of overlap / smaller volume);
                      rules.json interference.reference = "pair" keeps every draw and leaves out only the pairs that
                      collide in the reference
Every band, weight, floor, sample count and tolerance is read from rules.json. The draws come from one SeedSequence
(checks.child_seed keys, so adding a draw never moves another); the reference's collision results are cached on the
Reference per seed, so grading several submissions against one reference pays for them once.
Floors per arm: F_s = mean over the arm's parts of clamp((IoU - lo) / (hi - lo)) multiplies kinematic and feature;
F_s x band(tool-point median; floor_fk_mm) multiplies mass, fit and interference; shape is not floored. Arm score = the
weighted mean (rules.json weights); design_score = the mean over the arms (an invalid arm counts 0) x the unit gates;
each reported criterion is the mean over the arms of its floored value x the unit gates.
"""
import collections
import itertools
import json
import math
import os
import time

import numpy as np

from . import checks as C
from . import features as FT
from . import kinematics as K

GRADER_VERSION = "cad-tree-grade/1.0.0"
CHECKS = ["cad_valid", "public_model_reuse", "cad_json_consistent"]
CRITERIA = ["kinematic_fidelity", "feature_fidelity", "shape_fidelity", "mass_properties", "component_fit", "interference_free"]


# ----------------------------------------------------------------------------------------------- units
def _m(a):
    return np.asarray(a, float).reshape(4, 4)


def tree_only(unit):
    """The unit without its loops and loop bodies (the forward kinematics of the tree joints alone)."""
    joints = [j for j in unit["joints"] if j.get("tree", True)]
    reach = set()
    children = {j["child"] for j in joints}
    roots = [l["name"] for l in unit["links"] if l["name"] not in children]
    loop_bodies = set()
    names = {j["name"]: j for j in unit["joints"]}
    for lp in unit.get("loops", []):
        loop_bodies.update(names[n]["child"] for n in lp["joints"][:2] if n in names)
    reach.update(r for r in roots if r not in loop_bodies)
    grew = True
    while grew:
        grew = False
        for j in joints:
            if j["parent"] in reach and j["child"] not in reach:
                reach.add(j["child"])
                grew = True
    return {"links": [l for l in unit["links"] if l["name"] in reach], "joints": [j for j in joints if j["parent"] in reach],
            "loops": []}


class Unit(object):
    """A unit folder in bimanual-cad-unit/1: unit.json and parts/<tag>.brep (or .step), posed by cad_tree.kinematics."""

    def __init__(self, folder, classes, tags=None):
        self.dir = folder
        with open(os.path.join(folder, "unit.json")) as fh:
            self.unit = json.load(fh)
        u = self.unit
        self.classes = classes
        self.links = {l["name"]: l for l in u["links"]}
        self.joints = {j["name"]: j for j in u["joints"]}
        self.loops = list(u.get("loops", []))
        self.mech = K.Mechanism(u)
        self.fk = K.Mechanism(tree_only(u))
        self.occ = {}
        for l in u["links"]:
            for b in l["bought"]:
                if b["class"] not in classes:
                    raise RuntimeError("%s: bought class %s is not among the given classes" % (folder, b["class"]))
                self.occ[b["id"]] = {"class": b["class"], "link": l["name"], "P": _m(b["placement"])}
        self.tcp = {a: (t["link"], np.asarray(t["point"], float)) for a, t in u.get("tcp", {}).items()}
        self.tips = {k: np.asarray(v, float) for k, v in u.get("fingertips", {}).items()}
        self.parts, self.part_errors = {}, {}
        want = sorted(set(t for l in u["links"] for t in l["parts"]) | set(tags or []))
        for tag in want:
            shape, err = C.load_shape(os.path.join(folder, "parts", tag + ".brep"))
            if err:
                shape, err = C.load_shape(os.path.join(folder, "parts", tag + ".step"))
            if err:
                self.part_errors[tag] = err
                continue
            b = C.Body(tag, shape, strict=True)
            if not b.ok:
                self.part_errors[tag] = "; ".join(b.errors)
                continue
            self.parts[tag] = b
        self._items = None

    def link_of_part(self, tag):
        for n, l in self.links.items():
            if tag in l["parts"]:
                return n
        return None

    def items(self):
        """Per link the collision items at Z: [(manifold, local AABB (2, 3), volume)] of its parts and bought pieces."""
        if self._items is None:
            out = {}
            for n, l in self.links.items():
                its = [(self.parts[t].manifold, self.parts[t].bounds, self.parts[t].volume) for t in l["parts"] if t in self.parts]
                for b in l["bought"]:
                    Kc = self.classes[b["class"]]
                    P = _m(b["placement"])
                    corners = C.aabb_of(P, Kc.body.bounds)
                    its.append((C.manifold_transform(Kc.body.manifold, P), np.array([corners[:3], corners[3:]]), Kc.volume))
                out[n] = its
            self._items = out
        return self._items

    def anchor(self, jn):
        """The joint's anchor: its driving motor's output-face centre projected on the joint axis (else axis_point)."""
        j = self.joints[jn]
        p, d = np.asarray(j["axis_point"], float), C.unit(j["axis_dir"])
        o = self.occ.get(j.get("driven_by") or "")
        if o is None:
            return p
        c, _z = self.classes[o["class"]].output_centre(o["P"])
        return C.project_on_line(c, p, d)

    def axis(self, jn):
        j = self.joints[jn]
        return np.asarray(j["axis_point"], float), C.unit(j["axis_dir"])


class Reference(object):
    """The GT folder: the Unit, gt.json (features, operating_box_rad), parts.json (densities), the operating box."""

    def __init__(self, ref_dir, classes, rules=None):
        self.rules = rules or C.load_rules()
        self.dir = ref_dir
        self.classes = classes
        self.unit = Unit(ref_dir, classes)
        if self.unit.part_errors:
            raise RuntimeError("reference %s: parts do not load: %s" % (ref_dir, self.unit.part_errors))
        p = os.path.join(ref_dir, "gt.json")
        self.gt = json.load(open(p)) if os.path.isfile(p) else {}
        p = os.path.join(ref_dir, "parts.json")
        self.parts_json = json.load(open(p)) if os.path.isfile(p) else {}
        self.arms = sorted(set(l["arm"] for l in self.unit.unit["links"] if l.get("arm")))
        self.box, self.provisional = operating_box(self.unit, self.gt, self.rules)
        self.cache = {}
        self.part_index = {t: k for k, t in enumerate(sorted(self.unit.parts))}

    def density(self, tag):
        return float(self.parts_json.get(tag, {}).get("density_kg_m3", 2700.0))


def load_reference(ref_dir, classes, rules=None):
    return Reference(ref_dir, classes, rules)


def operating_box(unit, gt, rules):
    """({tree joint: [lo, hi]}, [joints on the provisional box]): unit.json limits, else gt.json operating_box_rad, else
    rules.json operating_box.provisional_rad by joint kind."""
    box, prov = {}, []
    gtb = gt.get("operating_box_rad") or {}
    provisional = rules["operating_box"]["provisional_rad"]
    for j in unit.fk.order:
        n = j["name"]
        if j.get("limits") is not None:
            box[n] = [float(j["limits"][0]), float(j["limits"][1])]
        elif n in gtb and gtb[n] is not None:
            box[n] = [float(gtb[n][0]), float(gtb[n][1])]
        else:
            box[n] = [float(v) for v in provisional.get(j.get("kind"), provisional["motor"])]
            prov.append(n)
    return box, prov


# ----------------------------------------------------------------------------------------------- the per-arm structure
def arm_prefix(arm, rules, unit):
    pre = rules.get("arms", {}).get(arm, {}).get("joint_prefix")
    if pre:
        return pre
    names = [j["name"] for j in unit["joints"] if j.get("arm") == arm]
    return os.path.commonprefix(names) if names else ""


def arm_tags(ref, arm):
    """The arm's parts (its tag prefix) and the shared ones (no arm prefix: the base plate)."""
    from .convert import tag_arm
    out = []
    for t in sorted(ref.unit.parts):
        a = tag_arm(t, ref.arms, ref.rules)
        if a == arm or a is None:
            out.append(t)
    return out


def arm_tree_joints(ref, arm):
    return [j["name"] for j in ref.unit.fk.order if j.get("arm") == arm and j["name"] in ref.box]


def make_cases(ss, ref):
    """Per-run draws from the SeedSequence `ss`: per arm the FK draws and the self-collision candidates in [0, 1)^d, the
    inter-arm candidates over both arms, and the sampling seed of the part alignments."""
    R = ref.rules["samples"]
    cases = {"fk": {}, "self": {}, "seed": int(np.random.default_rng(C.child_seed(ss, 0)).integers(2 ** 31 - 1))}
    for k, arm in enumerate(ref.arms):
        d = len(arm_tree_joints(ref, arm))
        cases["fk"][arm] = C.unit_cube_draws(int(R["fk_configs"]), d, C.child_seed(ss, 10 + k))
        cases["self"][arm] = C.unit_cube_draws(int(R["interference_candidates"]), d, C.child_seed(ss, 20 + k))
    dims = sum(len(arm_tree_joints(ref, a)) for a in ref.arms)
    cases["inter"] = C.unit_cube_draws(int(R["inter_arm_candidates"]), dims, C.child_seed(ss, 30))
    cases["key"] = (tuple(ss.entropy) if isinstance(ss.entropy, (list, tuple)) else ss.entropy, tuple(ss.spawn_key))
    return cases


def to_q(u, names, box):
    lo = np.array([box[n][0] for n in names])
    hi = np.array([box[n][1] for n in names])
    return {n: float(v) for n, v in zip(names, lo + np.asarray(u) * (hi - lo))}


# ----------------------------------------------------------------------------------------------- kinematic items
def _entity(unit, pre, ent):
    kind = ent[0]
    if kind == "anchor":
        return ("point", unit.anchor(pre + ent[1])) if pre + ent[1] in unit.joints else None
    if kind == "axis":
        return ("line", unit.axis(pre + ent[1])) if pre + ent[1] in unit.joints else None
    if kind == "centre":
        names = [pre + n for n in ent[1:]]
        if any(n not in unit.joints for n in names):
            return None
        return ("point", C.closest_point_to_lines([unit.axis(n) for n in names]))
    if kind == "tcp":
        return None
    raise ValueError("unknown key-length entity %s" % kind)


def key_lengths(unit, arm, rules):
    pre = arm_prefix(arm, rules, unit.unit)
    out = {}
    for name, spec in rules["key_lengths"].items():
        if name == "note":
            continue
        ea, eb = spec
        ents = []
        for e in (ea, eb):
            if e[0] == "tcp":
                ents.append(("point", C.apply(np.eye(4), unit.tcp[arm][1])) if arm in unit.tcp else None)
            else:
                ents.append(_entity(unit, pre, e))
        if None in ents:
            continue
        (ka, va), (kb, vb) = ents
        if ka == "point" and kb == "point":
            out[name] = float(np.linalg.norm(va - vb))
        elif ka == "point":
            out[name] = C.point_line_distance(va, vb[0], vb[1])
        elif kb == "point":
            out[name] = C.point_line_distance(vb, va[0], va[1])
        else:
            out[name] = C.line_line_distance(va[0], va[1], vb[0], vb[1])
    return out


def tool_points(unit, arm, qs, names):
    """(tool points (n, 3), {fingertip link: (n, 3)}) at the draws qs over the arm's tree joints; points at infinity
    when the unit has no tool point for the arm."""
    if arm not in unit.tcp or unit.tcp[arm][0] not in unit.links:
        return np.full((len(qs), 3), np.inf), {}
    tl, tp = unit.tcp[arm]
    tips = {k: v for k, v in unit.tips.items() if k in unit.links and unit.links[k].get("arm") == arm}
    P, F = [], {k: [] for k in tips}
    for q in qs:
        T = unit.fk.poses(q)
        P.append(C.apply(T[tl], tp))
        for k, v in tips.items():
            F[k].append(C.apply(T[k], v))
    return np.asarray(P), {k: np.asarray(v) for k, v in F.items()}


def kinematic_items(sub, ref, arm, cases):
    R = ref.rules
    B, W = R["bands"], R["subweights"]
    kw, aw, fw = W["kinematic"], W["axes"], W["fk"]
    U, S = ref.unit, sub
    items, m = [], {"axis": {}, "angle": {}, "length": {}, "range": {}, "notes": []}
    axj = [n for n, j in U.joints.items() if j.get("arm") == arm and j["kind"] in ("motor", "linkage")]
    for n in sorted(axj):
        pa = U.anchor(n)
        ps, ds = S.axis(n)
        e = C.point_line_distance(pa, ps, ds)
        g = C.line_angle_deg(U.axis(n)[1], ds)
        m["axis"][n], m["angle"][n] = e, g
        items += [("axis:" + n, kw["axes"] * aw["distance"] / len(axj), C.band(e, B["axis_mm"])),
                  ("angle:" + n, kw["axes"] * aw["angle"] / len(axj), C.band(g, B["axis_deg"]))]
    lr, ls = key_lengths(U, arm, R), key_lengths(S, arm, R)
    for n in sorted(lr):
        e = ls[n] - lr[n] if n in ls else float("inf")
        m["length"][n] = e
        items.append(("length:" + n, kw["lengths"] / len(lr), C.band(e, B["length_mm"])))
    names = arm_tree_joints(ref, arm)
    qs = [to_q(u, names, ref.box) for u in cases["fk"][arm]]
    pr, fr = tool_points(U, arm, qs, names)
    ps, fs = tool_points(S, arm, qs, names)
    with np.errstate(invalid="ignore"):
        e = np.linalg.norm(ps - pr, axis=1)
    e = np.where(np.isfinite(e), e, float(R["tolerances"]["fk_failure_mm"]))
    med, p90 = float(np.median(e)), float(np.percentile(e, 90))
    m.update(tcp_median=med, tcp_p90=p90, tips={})
    sfk = sum(fw.values())
    items += [("fk:median", kw["fk"] * fw["median"] / sfk, C.band(med, B["fk_median_mm"])),
              ("fk:p90", kw["fk"] * fw["p90"] / sfk, C.band(p90, B["fk_p90_mm"]))]
    tw = W.get("fingertip", {"median": 1.0})
    stw = sum(tw.values())
    for k in sorted(fr):
        et = np.linalg.norm(fs[k] - fr[k], axis=1) if k in fs else np.array([float("inf")])
        tm, tp = float(np.median(et)), float(np.percentile(et, 90))
        m["tips"][k] = tm
        wk = kw["fk"] * fw["fingertips"] / sfk / len(fr) / stw
        items.append(("fk:tip_%s_median" % k, wk * tw.get("median", 0.0), C.band(tm, B["fingertip_median_mm"])))
        if tw.get("p90"):
            items.append(("fk:tip_%s_p90" % k, wk * tw["p90"], C.band(tp, B.get("fingertip_p90_mm", B["fk_p90_mm"]))))
    for n in names:
        lim = S.joints[n].get("limits")
        b = ref.box[n]
        if lim is None or lim[0] is None or lim[1] is None:
            cov = float(R["tolerances"]["missing_limit_coverage"])
            m["notes"].append("%s: limits not set" % n)
        else:
            inter = max(0.0, min(lim[1], b[1]) - max(lim[0], b[0]))
            if R["tolerances"].get("range_metric", "coverage") == "iou":     # too wide and too narrow both cost
                cov = inter / max(1e-9, max(lim[1], b[1]) - min(lim[0], b[0]))
            else:
                cov = inter / max(1e-9, b[1] - b[0])
        m["range"][n] = cov
        items.append(("range:" + n, kw["range"] / len(names), cov))
    return items, m


# ----------------------------------------------------------------------------------------------- parts: shape and features
def part_result(sub, ref, tag, seed):
    """Per-part alignment and the shape quantities (the submitted part's outer shell onto the reference part)."""
    from scipy.spatial import cKDTree
    k = ref.part_index[tag]
    ma = C.outer_shell(sub.parts[tag].mesh).copy()
    mg = ref.unit.parts[tag].mesh
    al = C.align_part(ma, mg, seed + 101 * k, params=ref.rules.get("part_alignment"))
    m_al = ma.copy()
    m_al.apply_transform(al["T"])
    PA = ref.rules.get("part_alignment") or {}
    n_src = int(PA.get("n_src", C.N_SRC))
    pa, _n = C.sample_surface(m_al, n_src, seed + 7 * k + 3)
    pg, _n = C.sample_surface(mg, n_src, seed + 7 * k + 4)
    dense_a, dense_an = C.sample_surface(m_al, C.n_dst_for(m_al, PA), seed + 7 * k + 5)
    d1 = C.surface_distances(pa, al["gt_pts"], al["gt_normals"], al["gt_tree"])
    d2 = C.surface_distances(pg, dense_a, dense_an, cKDTree(dense_a))
    sd = float(np.percentile(np.concatenate([d1, d2]), 90))
    man_a = C.trimesh_to_manifold(m_al)
    man_g = ref.unit.parts[tag].manifold
    if C.manifold_ok(man_a):
        vi = (man_a ^ man_g).volume()
        vu = man_a.volume() + man_g.volume() - vi
        iou = float(vi / vu) if vu > 0 else 0.0
    else:
        iou = 0.0
    return {"iou": iou, "surf": sd, "ext": FT.extent_errors(m_al, mg), "mesh": m_al,
            "align": {"rms_mm": al["rms_mm"], "offset_mm": al["offset_mm"], "rotation_deg": al["rotation_deg"], "multistart": al["multistart"]}}


def shape_feature_items(pr, ref, tags):
    R = ref.rules
    B, W = R["bands"], R["subweights"]
    sw, fw = W["shape"], W["feature"]
    ssw = sum(sw.values())
    shape, feat, det = [], [], []
    for t in tags:
        r = pr[t]
        shape += [("surface:" + t, sw["surface"] / (ssw * len(tags)), C.band(r["surf"], B["surface_mm"])),
                  ("iou:" + t, sw["iou"] / (ssw * len(tags)), C.band(1.0 - r["iou"], B["one_minus_iou"]))]
        feat += [("extent:%s:%s" % (t, "xyz"[i]), fw["extents"] / (3 * len(tags)), C.band(r["ext"][i], B["extent_mm"])) for i in range(3)]
    feats = [f for f in ref.gt.get("features", []) if f.get("tag") in tags]
    cuts = [f for f in feats if f.get("kind") == "cutout"]
    spacers = [f for f in feats if f.get("kind") == "spacer"]
    for n, f in enumerate(cuts):
        r = FT.cutout_item(pr[f["tag"]]["mesh"], ref.unit.parts[f["tag"]].mesh, f, B, R.get("probes"))
        feat.append(("cutout:%d:%s" % (n + 1, f["tag"]), fw["cutouts"] / len(cuts), r["score"]))
        det.append("cut-out %s %s: present %.0f, offset %s mm, area error %s -> %.2f" % (
            f.get("shape", ""), f["tag"], r["present"], r["offset_mm"], r["area_error"], r["score"]))
    for n, f in enumerate(spacers):
        r = FT.spacer_item(pr[f["tag"]]["mesh"], ref.unit.parts[f["tag"]].mesh, f, B, R.get("probes"))
        feat.append(("spacer:%d:%s" % (n + 1, f["tag"]), fw["spacers"] / len(spacers), r["score"]))
        det.append("spacer %s at %s: floor error %+.2f mm -> %.2f" % (f["tag"], f.get("joint", "?"), r["error_mm"], r["score"]))
    return shape, feat, det


# ----------------------------------------------------------------------------------------------- mass
def link_mass(unit, ref, ln):
    """(mass kg, CoM mm at Z) of a link: part volumes x the reference densities + the bought masses (0 kg for a link
    the unit does not have)."""
    m, c = 0.0, np.zeros(3)
    if ln not in unit.links:
        return m, c
    for t in unit.links[ln]["parts"]:
        if t not in unit.parts:
            continue
        b = unit.parts[t]
        mt = b.volume * 1e-9 * ref.density(t)
        m += mt
        c += mt * b.centroid
    for o in unit.links[ln]["bought"]:
        Kc = ref.classes[o["class"]]
        m += Kc.mass
        c += Kc.mass * C.apply(_m(o["placement"]), Kc.com)
    return m, (c / m if m > 0 else c)


def mass_items(sub, ref, arm):
    B, W = ref.rules["bands"], ref.rules["subweights"]["mass"]
    links = [n for n, l in ref.unit.links.items() if l.get("arm") == arm]
    items, m = [], {}
    sw = sum(W.values())
    for ln in sorted(links):
        mr, cr = link_mass(ref.unit, ref, ln)
        ms, cs = link_mass(sub, ref, ln)
        rel = (ms - mr) / mr if mr > 0 else float("inf")
        dc = float(np.linalg.norm(cs - cr)) if ms > 0 else float("inf")
        m[ln] = (rel, dc)
        items += [("mass:" + ln, W["mass"] / sw / len(links), C.band(rel, B["mass_rel"])),
                  ("com:" + ln, W["com"] / sw / len(links), C.band(dc, B["com_mm"]))]
    return items, m


# ----------------------------------------------------------------------------------------------- component fit
def _rings(face_point, normal, bolts, outboard, margin, n):
    """Probe rings (class frame) 1 mm outboard of a face: around the bolt circle (pcd) at -+ (d / 2 + margin), or around
    each listed bolt (xy) at d / 2 + margin."""
    q = np.asarray(face_point, float) + outboard * C.unit(normal)
    nz = C.unit(normal)
    ref = np.array([1.0, 0.0, 0.0]) if abs(nz[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = C.unit(ref - nz * float(ref @ nz))
    b = np.cross(nz, a)
    ts = np.linspace(0.0, 2.0 * math.pi, int(n), endpoint=False)
    d = float(bolts.get("d", 2.5))
    out = []
    if bolts.get("pcd"):
        r0 = 0.5 * float(bolts["pcd"])
        for r in (r0 - (0.5 * d + margin), r0 + (0.5 * d + margin)):
            if r > 0.5:
                out.append(np.array([q + r * (math.cos(t) * a + math.sin(t) * b) for t in ts]))
    for xy in bolts.get("xy") or []:
        c = np.asarray(list(xy) + [0.0] * (3 - len(xy)), float)
        c = c - nz * float((c - np.asarray(face_point, float)) @ nz) + outboard * nz
        r = 0.5 * d + margin
        out.append(np.array([c + r * (math.cos(t) * a + math.sin(t) * b) for t in ts]))
    return out


def _inside(unit, ln, pts):
    acc = np.zeros(len(pts), dtype=bool)
    for t in unit.links[ln]["parts"]:
        if t in unit.parts:
            b = unit.parts[t]
            lo, hi = b.bounds
            near = np.all((pts >= lo - 1e-6) & (pts <= hi + 1e-6), axis=1)
            if near.any():
                acc[near] |= b.contains(pts[near])
    return acc


def _keepout_volume(unit, oid, links):
    o = unit.occ[oid]
    Kc = unit.classes[o["class"]]
    ko = C.manifold_transform(Kc.keepout.manifold, o["P"])
    box = C.aabb_of(o["P"], Kc.keepout.bounds)
    v = 0.0
    for ln in links:
        for t in unit.links[ln]["parts"]:
            if t in unit.parts:
                b = unit.parts[t]
                if C.aabb_overlap(box, np.concatenate(b.bounds)):
                    v += (ko ^ b.manifold).volume()
    return v


def fit_items(sub, ref, arm):
    """Per motor of the arm (the reference occurrences that drive its motor joints): keep-out x the case side x the
    output side (rules.json fit)."""
    F = ref.rules["fit"]
    U = ref.unit
    motors = {}
    for n, j in sorted(U.joints.items()):
        if j.get("arm") == arm and j["kind"] == "motor" and j.get("driven_by") and j["driven_by"] not in motors:
            case = U.occ[j["driven_by"]]["link"]
            motors[j["driven_by"]] = (case, j["child"] if case == j["parent"] else j["parent"])
    out, det = {}, []
    for oid, (case, driven) in sorted(motors.items()):
        Kc = ref.classes[U.occ[oid]["class"]]
        if oid not in sub.occ or any(ln not in sub.links for ln in (case, driven)):
            out[oid] = 0.0
            det.append("%s: not in the submitted unit -> 0" % oid)
            continue
        if sub.occ[oid]["class"] != U.occ[oid]["class"]:
            out[oid] = 0.0
            det.append("%s: a %s where the unit has a %s -> 0" % (oid, sub.occ[oid]["class"], U.occ[oid]["class"]))
            continue
        vr = _keepout_volume(U, oid, (case, driven))
        vs = _keepout_volume(sub, oid, (case, driven))
        ko = C.score_ratio(max(0.0, vs - vr) / max(Kc.volume, 1e-9), float(F["keepout_ratio"]))
        sides = []
        faces = [("case", case, f["point"], f["normal"], f.get("bolts", {})) for f in (Kc.meta.get("mount_faces") or [])]
        if Kc.meta.get("output_bolts") and Kc.output_face_z is not None:
            faces.append(("output", driven, [0.0, 0.0, float(Kc.output_face_z)], [0.0, 0.0, 1.0], Kc.meta["output_bolts"]))
        per = collections.defaultdict(list)
        for side, ln, pt, nrm, bolts in faces:
            for ring in _rings(pt, nrm, bolts, float(F["probe_outboard_mm"]), float(F["ring_margin_mm"]), int(F["ring_points"])):
                fr = float(_inside(U, ln, C.apply(U.occ[oid]["P"], ring)).mean())
                if fr < float(F["ref_min_fraction"]):
                    continue
                fsub = float(_inside(sub, ln, C.apply(sub.occ[oid]["P"], ring)).mean())
                per[side].append(C.ring_score(fsub / fr, *F["relative_ring"]))
        score = ko
        parts_txt = ["keep-out +%.1f mm3 -> %.2f" % (max(0.0, vs - vr), ko)]
        for side in ("case", "output"):
            if per[side]:
                s = float(np.mean(per[side]))
                score *= s
                parts_txt.append("%s %.2f" % (side, s))
                sides.append(side)
        out[oid] = score
        det.append("%s: %s -> %.2f" % (oid, ", ".join(parts_txt), score))
    return out, det


# ----------------------------------------------------------------------------------------------- interference
def _placed(unit, T, links):
    out = {}
    its = unit.items()
    for ln in links:
        Tl = T[ln]
        out[ln] = [(C.manifold_transform(man, Tl), C.aabb_of(Tl, bnd), vol) for man, bnd, vol in its[ln]]
    return out


def _pair_ratio(pa, pb):
    vol = 0.0
    for ma, ba, _va in pa:
        for mb, bb, _vb in pb:
            if C.aabb_overlap(ba, bb):
                vol += (ma ^ mb).volume()
    tot = min(sum(v for _m, _b, v in pa), sum(v for _m, _b, v in pb))
    return vol / max(tot, 1e-9), vol


def _adjacent(unit):
    adj = set()
    for j in unit.joints.values():
        adj.add((j["parent"], j["child"]))
        adj.add((j["child"], j["parent"]))
    return adj


def pair_scores(unit, q, pairs, rules):
    """{pair: (score_ratio of the overlap over the smaller link's volume, overlap mm3)} at the joint angles q; None when
    a loop cannot close."""
    links = sorted(set(l for p in pairs for l in p))
    if any(l not in unit.links for l in links):
        return None
    try:
        T = unit.mech.poses(q)
    except K.LoopError:
        return None
    pl = _placed(unit, T, links)
    adj = _adjacent(unit)
    cl = rules["clearances"]
    out = {}
    for a, b in pairs:
        ratio, vol = _pair_ratio(pl[a], pl[b])
        thr = float(cl["adjacent_ratio"] if (a, b) in adj else cl["nonadjacent_ratio"])
        out[(a, b)] = (C.score_ratio(ratio, thr), vol)
    return out


def _term(sub, ref, draws, to_config, pairs, n_want, key0):
    """(mean score, kept, examined, clean, worst text) of one interference term: per draw the submission's worst pair,
    over the draws the reference passes (rules.json interference.reference: "config" = the reference has no colliding
    pair at the draw; "pair" = every draw, the pairs that collide in the reference left out)."""
    mode = ref.rules.get("interference", {}).get("reference", "config")
    scores, worst, examined = [], (1.0, ""), 0
    for i, u in enumerate(draws):
        if len(scores) >= n_want:
            break
        examined += 1
        q = to_config(u)
        key = key0 + (i,)
        if key not in ref.cache:
            ref.cache[key] = pair_scores(ref.unit, q, pairs, ref.rules)
        rp = ref.cache[key]
        if rp is None:
            continue
        if mode == "pair":
            allowed = [p for p in pairs if rp[p][0] >= 1.0]
        else:
            allowed = pairs if all(v[0] >= 1.0 for v in rp.values()) else []
        if not allowed:
            continue
        sp = pair_scores(sub, q, allowed, ref.rules)
        if sp is None:
            s, txt = 0.0, "the linkage cannot close (or a link is missing)"
        else:
            p = min(allowed, key=lambda x: sp[x][0])
            s, txt = sp[p][0], "%s-%s %.1f mm3" % (p[0], p[1], sp[p][1])
        scores.append(s)
        if s < worst[0]:
            worst = (s, txt)
    clean = sum(1 for s in scores if s >= 1.0)
    return (float(np.mean(scores)) if scores else None), len(scores), examined, clean, worst[1]


def interference(sub, ref, arm, cases, both_valid):
    """(score, detail): the self term over the arm's draws (the other arm at Z) and, when both arms are valid, the
    inter-arm term over draws of both arms; a term no reference draw passes is not measured (1.0, said so)."""
    R = ref.rules
    S_ = R["samples"]
    W = R["subweights"]["interference"]
    U = ref.unit
    arm_links = [n for n, l in U.links.items() if l.get("arm") == arm]
    pairs = list(itertools.combinations(sorted([U.mech.root] + arm_links), 2))
    names = arm_tree_joints(ref, arm)
    s_self, kept, ex, clean, worst = _term(sub, ref, cases["self"][arm], lambda u: to_q(u, names, ref.box), pairs,
                                           int(S_["interference_configs"]), ("self", arm, cases["key"]))
    det = "self: %d reference draws used of %d, %d clean%s" % (kept, ex, clean, ", worst %s" % worst if worst else "")
    if s_self is None:
        s_self = 1.0
        det += " (not measured: no draw of the reference box passes)"
    if not both_valid or len(ref.arms) < 2:
        return s_self, det + ("; inter-arm not measured (the other arm is not valid)" if len(ref.arms) > 1 else "")
    left, right = ref.arms[0], ref.arms[1]
    ll = sorted(n for n, l in U.links.items() if l.get("arm") == left)
    rl = sorted(n for n, l in U.links.items() if l.get("arm") == right)
    cross = [(a, b) for a in ll for b in rl]
    nl, nr = arm_tree_joints(ref, left), arm_tree_joints(ref, right)

    def both(u):
        q = to_q(u[:len(nl)], nl, ref.box)
        q.update(to_q(u[len(nl):], nr, ref.box))
        return q
    s_int, kept, ex, clean, worst = _term(sub, ref, cases["inter"], both, cross, int(S_["inter_arm_configs"]), ("inter", cases["key"]))
    det += "; inter-arm: %d reference draws used of %d, %d clean%s" % (kept, ex, clean, ", worst %s" % worst if worst else "")
    if s_int is None:
        s_int = 1.0
        det += " (not measured)"
    return (W["self"] * s_self + W["inter_arm"] * s_int) / (W["self"] + W["inter_arm"]), det


# ----------------------------------------------------------------------------------------------- public-model reuse
def load_blocklist(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path) as fh:
        doc = json.load(fh)
    return {"sha256": set(doc.get("sha256", [])), "vertex_fingerprints": set(doc.get("vertex_fingerprints", [])),
            "invariants": doc.get("invariants", [])}


def public_reuse(sub, blocklist, submission_dir=None):
    """(gate, details, n pose-invariant matches): 0 when a part's vertex fingerprint (in B, as converted) or a loose
    STEP / STL / BREP file of the submission is in the blocklist."""
    if blocklist is None:
        return 1.0, ["no public blocklist given"], 0
    hits, det, inv = [], [], 0
    for t, b in sorted(sub.parts.items()):
        if C.vertex_fingerprint(b.mesh) in blocklist["vertex_fingerprints"]:
            hits.append(t)
            det.append("%s: vertex fingerprint of a public model -> gate 0" % t)
        g = C.geometry_invariants(b.mesh)
        if any(C.invariants_match(g, r) for r in blocklist["invariants"]):
            inv += 1
    if submission_dir and os.path.isdir(submission_dir):
        for root, _d, files in os.walk(submission_dir):
            for fn in files:
                if fn.lower().endswith((".step", ".stp", ".stl", ".brep")):
                    p = os.path.join(root, fn)
                    if not os.path.islink(p) and C.sha256_file(p) in blocklist["sha256"]:
                        hits.append(fn)
                        det.append("%s: a public model file -> gate 0" % fn)
    return (0.0 if hits else 1.0), (det or ["no part matches the public blocklist"]), inv


# ----------------------------------------------------------------------------------------------- the grade
def grade(conv_dir, ref, ss, conversion=None, blocklist=None, submission_dir=None, log=None):
    """Grade the converted submission in `conv_dir` against `ref` with the draws of the SeedSequence `ss`. Returns
    (rewards, stats, details); rewards hold the checks, the six criteria and design_score."""
    log = log or (lambda m: None)
    t0 = time.time()
    R = ref.rules
    rewards = {k: 0.0 for k in CHECKS + CRITERIA + ["design_score"]}
    stats = {"grader_version_" + GRADER_VERSION.split("/")[1].replace(".", "_"): 1.0}
    details = {"arms": {}}
    if conversion is None:
        p = os.path.join(conv_dir, "convert.json")
        conversion = json.load(open(p)) if os.path.isfile(p) else {"ok": False, "errors": ["no conversion record"], "arms": {}}
    if ref.provisional:
        details["operating_box"] = ["provisional box (rules.json operating_box) for %d joints without reference limits: %s" % (
            len(ref.provisional), ", ".join(ref.provisional[:6]) + (" ..." if len(ref.provisional) > 6 else ""))]
        stats["operating_box_provisional"] = float(len(ref.provisional))
    if not conversion.get("ok"):
        details["cad_valid"] = [str(e)[:300] for e in conversion.get("errors", [])[:12]] or ["the document was not converted"]
        log("[check] cad_valid FAILED: " + " | ".join(details["cad_valid"][:4]))
        return rewards, stats, details
    sub = Unit(conv_dir, ref.classes, tags=list(ref.unit.parts))
    shared = [t for t in sorted(ref.unit.parts) if t not in sum((arm_tags(ref, a) for a in ref.arms), []) or
              all(t in arm_tags(ref, a) for a in ref.arms)]
    bad_shared = [t for t in shared if t in sub.part_errors]
    if bad_shared:
        details["cad_valid"] = ["part %s: %s" % (t, sub.part_errors[t][:160]) for t in bad_shared]
        return rewards, stats, details
    rewards["cad_valid"] = 1.0
    details["cad_valid"] = ["the unit is found; %d parts load as valid watertight solids" % len(sub.parts)]
    reuse, rdet, inv = public_reuse(sub, blocklist, submission_dir)
    rewards["public_model_reuse"] = reuse
    details["public_model_reuse"] = rdet[:6]
    stats["source_geometry_matches"] = float(inv)
    cases = make_cases(ss, ref)
    seed = cases["seed"]
    arm_ok, arm_cons = {}, {}
    for arm in ref.arms:
        a = conversion.get("arms", {}).get(arm, {})
        reasons = list(a.get("errors", []))
        bad = [t for t in arm_tags(ref, arm) if t in sub.part_errors and t not in shared]
        reasons += ["part %s: %s" % (t, sub.part_errors[t][:120]) for t in bad]
        arm_ok[arm] = bool(a.get("valid")) and not bad
        arm_cons[arm] = bool(a.get("consistent")) and arm_ok[arm]
        details["arms"][arm] = {"valid": arm_ok[arm], "reasons": reasons[:8], "consistent": arm_cons[arm],
                                "consistency": list(a.get("consistency", []))[:6]}
        for k in ("misplaced", "case_flipped"):        # rules.json link_membership "majority", case_side "reported"
            if a.get(k):
                details["arms"][arm][k] = list(a[k])[:8]
    valid_arms = [a for a in ref.arms if arm_ok[a]]
    rewards["cad_json_consistent"] = 1.0 if valid_arms and all(arm_cons[a] for a in valid_arms) else 0.0
    details["cad_json_consistent"] = ["%s: %s" % (a, "consistent" if arm_cons[a] else ("; ".join(details["arms"][a]["consistency"][:3])
                                                                                    or "not valid")) for a in ref.arms]
    unit_gate = 1.0
    for g in R["gates"]["unit"]:
        unit_gate *= rewards[g]
    t_parts = time.time()
    pr = {}
    for t in sorted(set(t for a in valid_arms for t in arm_tags(ref, a))):
        pr[t] = part_result(sub, ref, t, seed)
    stats["time_parts_s"] = time.time() - t_parts
    per_arm = {}
    W = R["weights"]
    B = R["bands"]
    for arm in ref.arms:
        crit = {c: 0.0 for c in CRITERIA}
        if not arm_ok[arm]:
            per_arm[arm] = crit
            stats["%s_valid" % arm] = 0.0
            continue
        ta = time.time()
        tags = arm_tags(ref, arm)
        kin_items, km = kinematic_items(sub, ref, arm, cases)
        shape_items, feat_items, fdet = shape_feature_items(pr, ref, tags)
        m_items, mm = mass_items(sub, ref, arm)
        fits, fit_det = fit_items(sub, ref, arm)
        tb = time.time()
        inter, idet = interference(sub, ref, arm, cases, len(valid_arms) == len(ref.arms))
        stats["time_interference_%s_s" % arm] = time.time() - tb
        kin, feat, shape, mass = C.weighted(kin_items), C.weighted(feat_items), C.weighted(shape_items), C.weighted(m_items)
        fit = float(np.mean(list(fits.values()))) if fits else 1.0
        lo, hi = B["floor_iou"]
        F_s = float(np.mean([C.clamp01((pr[t]["iou"] - lo) / (hi - lo)) for t in tags]))
        F_k = C.band(km["tcp_median"], B["floor_fk_mm"])
        gate = 1.0
        for g in R["gates"]["arm"]:
            gate *= 1.0 if (g != "cad_json_consistent" or arm_cons[arm]) else 0.0
        raw = {"kinematic_fidelity": kin, "feature_fidelity": feat, "shape_fidelity": shape, "mass_properties": mass,
               "component_fit": fit, "interference_free": inter}
        floor = {"kinematic_fidelity": F_s, "feature_fidelity": F_s, "shape_fidelity": 1.0, "mass_properties": F_s * F_k,
                 "component_fit": F_s * F_k, "interference_free": F_s * F_k}
        for c in CRITERIA:
            crit[c] = float(max(0.0, min(1.0, gate * raw[c] * floor[c])))
            stats["%s_raw_%s" % (arm, c)] = raw[c]
            stats["%s_%s" % (arm, c)] = crit[c]
        per_arm[arm] = crit
        kg, fg = C.group_means(kin_items), C.group_means(feat_items)
        stats.update({"%s_valid" % arm: 1.0, "%s_floor_shape" % arm: F_s, "%s_floor_kinematic" % arm: F_k,
                      "%s_tcp_median_mm" % arm: km["tcp_median"], "%s_tcp_p90_mm" % arm: km["tcp_p90"],
                      "%s_iou_mean" % arm: float(np.mean([pr[t]["iou"] for t in tags])),
                      "%s_surface_mm_mean" % arm: float(np.mean([pr[t]["surf"] for t in tags])),
                      "%s_time_s" % arm: time.time() - ta})
        for k, v in kg.items():
            stats["%s_kinematic_%s" % (arm, k)] = v
        for k, v in fg.items():
            stats["%s_feature_%s" % (arm, k)] = v
        for n, v in km["axis"].items():
            stats["axis_err_mm_" + n] = v
            stats["axis_angle_deg_" + n] = km["angle"][n]
        for n, v in km["length"].items():
            stats["length_err_mm_%s_%s" % (arm, n)] = v
        for n, v in km["tips"].items():
            stats["tip_median_mm_" + n] = v
        for ln, (rel, dc) in mm.items():
            stats["mass_rel_err_" + ln], stats["com_err_mm_" + ln] = rel, dc
        for oid, v in fits.items():
            stats["fit_" + oid] = v
        d = details["arms"][arm]
        d["kinematic_fidelity"] = "axes %.3f, lengths %.3f, FK %.3f (tool median %.2f / p90 %.2f mm, tips %s), range %.3f%s" % (
            kg.get("axis", 0.0), kg.get("length", 0.0), kg.get("fk", 0.0), km["tcp_median"], km["tcp_p90"],
            "/".join("%.2f" % v for v in km["tips"].values()), kg.get("range", 0.0),
            ("; " + "; ".join(km["notes"][:3])) if km["notes"] else "")
        d["feature_fidelity"] = "cut-outs %s, spacers %s, extents %.3f" % (
            "%.3f" % fg["cutout"] if "cutout" in fg else "none", "%.3f" % fg["spacer"] if "spacer" in fg else "none", fg.get("extent", 0.0))
        d["features"] = fdet[:8]
        worst = sorted(tags, key=lambda t: pr[t]["iou"])[:5]
        d["shape_fidelity"] = "mean IoU %.3f, surface %.2f mm; lowest (surface mm / IoU): %s" % (
            stats["%s_iou_mean" % arm], stats["%s_surface_mm_mean" % arm], ", ".join("%s %.2f/%.3f" % (t, pr[t]["surf"], pr[t]["iou"]) for t in worst))
        wl = sorted(mm, key=lambda ln: -(abs(mm[ln][0]) / B["mass_rel"][1] + mm[ln][1] / B["com_mm"][1]))[:5]
        d["mass_properties"] = "worst links (mass rel. error / CoM mm): " + ", ".join("%s %+.3f/%.1f" % (ln, mm[ln][0], mm[ln][1]) for ln in wl)
        d["component_fit"] = fit_det[:8]
        d["interference_free"] = idet
        d["floors"] = "F_s %.3f, F_k %.3f" % (F_s, F_k)
        d["criteria"] = {c: round(crit[c], 4) for c in CRITERIA}
        log("[arm %s] %s" % (arm, ", ".join("%s %.3f" % (c.split("_")[0], crit[c]) for c in CRITERIA)))
    for c in CRITERIA:
        rewards[c] = float(np.mean([per_arm[a][c] for a in ref.arms])) * unit_gate
    arm_scores = {a: sum(W[c] * per_arm[a][c] for c in CRITERIA) / sum(W[c] for c in CRITERIA) for a in ref.arms}
    for a, v in arm_scores.items():
        stats["%s_arm_score" % a] = v
    rewards["design_score"] = float(np.mean(list(arm_scores.values()))) * unit_gate
    for k, v in (conversion.get("stats") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            stats["conv_" + k] = float(v)
    stats["grade_s"] = time.time() - t0
    for c in CRITERIA:
        details[c] = ["%s %.3f" % (a, per_arm[a][c]) for a in ref.arms]
    return rewards, stats, details


def main(argv=None):
    """Dev entry: grade a converted folder. python -m <pkg>.cad_tree.grade CONV_DIR REF_DIR BOUGHT_DIR [HI:LO] [OUT_JSON]"""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    conv_dir, ref_dir, bought_dir = argv[:3]
    hi, lo = [int(x) for x in (argv[3] if len(argv) > 3 else "2026:1027").split(":")]
    classes = C.load_classes(bought_dir)
    ref = load_reference(ref_dir, classes)
    rewards, stats, details = grade(conv_dir, ref, np.random.SeedSequence([hi, lo]), log=print)
    print(json.dumps(rewards, indent=1))
    if len(argv) > 4:
        with open(argv[4], "w") as fh:
            json.dump({"rewards": rewards, "stats": stats, "details": details}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
