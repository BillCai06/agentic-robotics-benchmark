"""Turn an extracted FreeCAD document (cad_tree.extract, format bimanual-fcstd-extract/1) into a submitted unit in the
reference's own format (bimanual-cad-unit/1, CONTRACTS 6), so that cad_tree.kinematics poses the submission and the
reference by one rule.

    res = convert(extract_dir, ref_dir, bought_dir, out_dir)      writes out_dir/unit.json, out_dir/parts/<tag>.brep

The agent is given no topology, so the mechanism is inferred from the document (CONTRACTS 7):

  parts    leaves labelled with a tag of the reference (T01, L01 .. L14, R01 .. R14), one valid solid each; a label
           used twice is resolved by visibility (the one visible body) or is an error;
  bought   leaves the extractor matched to a class (volume, sorted extents and local centroid); a leaf that matches by
           volume and extents alone (its placement baked into the geometry, or one of two mirrored gripper classes) is
           registered on each candidate class's geometry, which gives its class and its class-frame placement;
  links    groups of components held together by Fixed joints and the Grounded joint. A link is the reference link
           with the same tag set (rules.json tolerances.link_membership "strict"; "majority": the group holding the
           largest share of the reference link's part volume, a part fixed elsewhere graded where it is and reported,
           not an error); a part-less link (the gripper, its fingers) is the group that holds the reference
           link's bought classes and hangs by a revolute joint from the link matched to its reference parent;
  joints   each reference joint is the one Revolute joint that connects its two matched links (either order); Gears,
           RackPinion, Belt and Screw joints are ignored, other joint types are noted and play no part;
  motors   each motor joint's driver: a motor on the joint's parent or child link whose axis lies on the joint axis
           (motor_match_deg / _mm), the one nearest the driven link, the reference driver's class first; its case
           must ride on the reference's side (rules.json tolerances.case_side "strict"; "reported": a case on the
           other side is reported, not an error, and the joint's sense taken from the side it rides on). A motor of
           another type is matched all the same (noted; the grader's component_fit scores it 0). A linkage joint
           takes the motor of the motor joint with its driver;
  sense    the motor's +z (case on the parent) or -z (case on the child: the child-relative-to-parent axis); for a
           finger the reference's hinge axis carried by the gripper module; pins follow the crank. Zero = the saved
           pose; FreeCAD's angle (Reference2's part relative to Reference1's about jcs1 z) gives the limits relative to
           it, turned into the reference sense (a joint built child-first or with its frames flipped reads the same);
  loops    the reference's loops on the submitted joints (crank and coupler lengths from the submitted axes);
  tool     the tool point and the fingertips carried by the gripper module and finger pieces (their reference points
           in those pieces' frames);
  frame    the submitted torso (its made parts, the column, the J1 motors) aligned rigidly onto the reference torso:
           the column frames matched, then point-to-plane ICP on surface samples with a fixed seed (rules.json
           torso_alignment); the residual is reported.

Per-arm validity: an arm is valid when every reference link and joint of it is found, its links hold the reference's
bought pieces (motors counted as motors, whatever their type) and its motors sit on their joints; else the reasons are
recorded and the grader scores the arm 0, the other arm unaffected. A part fixed to another link than the reference's
(link_membership "majority") and a motor case on the other side of its joint (case_side "reported"; the motors are then
left out of the per-link bought check) are reported in the arm's `misplaced` / `case_flipped`, not errors; a part on a
rigid link that matches none of the reference's links is an error (a link the unit does not have). Unit-wide errors (no readable document, a missing
tag, wrong bought counts (motors counted together), no grounded torso holding the
shared parts and the column) set ok = False (cad_valid 0). Consistency per arm (each joint's axis on its motor's axis,
a finger's on the module's hinge, the pins parallel to the crank, both joint frames coincident at the saved pose:
consistency_deg / _mm) is recorded for the grader's arm gate. Errors are strings; nothing raises.
"""
import collections
import json
import math
import os
import traceback

import numpy as np

from . import checks as C

EXTRACT_FORMAT = "bimanual-fcstd-extract/1"
UNIT_FORMAT = "bimanual-cad-unit/1"
COUPLING_TYPES = ("Gears", "RackPinion", "Belt", "Screw")
JOINT_ORDER = {"motor": 0, "linkage": 1, "finger": 2, "pin": 3}


def _m(a):
    return np.asarray(a, float).reshape(4, 4)


def _is_motor(K):
    return K.kind == "motor" or (K.kind is None and K.output_face_z is not None)


def _kinds(counter, classes):
    """A class counter with every motor class counted as "motor": a motor of another type is matched joint by joint and
    scored there (grade: component_fit), not a different mechanism."""
    out = collections.Counter()
    for c, n in counter.items():
        out["motor" if c in classes and _is_motor(classes[c]) else c] += n
    return out


class _UF(object):
    def __init__(self, names=()):
        self.p = {n: n for n in names}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)        # the smaller name is the root: deterministic


def _fmt_counts(c):
    return ", ".join("%s x%d" % (k, v) if v > 1 else k for k, v in sorted(c.items())) or "no bought piece"


def tag_arm(tag, arms, rules, fallback=None):
    """The arm a tag belongs to by its prefix (rules.json arms.*.tag_prefix), else `fallback` (shared parts: None)."""
    for arm in arms:
        pre = rules.get("arms", {}).get(arm, {}).get("tag_prefix")
        if pre and tag.startswith(pre):
            return arm
    return fallback


def ref_structure(unit, rules):
    """The reference mechanism in the terms the converter and the grader use."""
    links = {l["name"]: l for l in unit["links"]}
    joints = {j["name"]: j for j in unit["joints"]}
    tree = [j for j in unit["joints"] if j.get("tree", True)]
    children = set(j["child"] for j in tree)
    loop_bodies = set()
    for lp in unit.get("loops", []):
        loop_bodies.update(joints[n]["child"] for n in lp["joints"][:2])
    roots = [n for n in links if n not in children and n not in loop_bodies]
    root = roots[0] if roots else unit["links"][0]["name"]
    arms = sorted(set(l["arm"] for l in unit["links"] if l.get("arm")))
    occ = {}
    for l in unit["links"]:
        for b in l["bought"]:
            occ[b["id"]] = {"class": b["class"], "link": l["name"], "placement": _m(b["placement"])}
    tags = [t for l in unit["links"] for t in l["parts"]]
    t_arm = {t: tag_arm(t, arms, rules) for t in tags}
    drivers = {}
    for j in unit["joints"]:
        if j.get("driven_by"):
            drivers.setdefault(j["driven_by"], []).append(j["name"])
    s = {"unit": unit, "links": links, "joints": joints, "root": root, "arms": arms, "occ": occ, "tags": set(tags),
         "tag_arm": t_arm, "drivers": drivers,
         "shared_tags": sorted(t for t in tags if t_arm[t] is None),
         "arm_links": {a: [n for n, l in links.items() if l.get("arm") == a] for a in arms},
         "arm_joints": {a: [n for n, j in joints.items() if j.get("arm") == a] for a in arms},
         "arm_tags": {a: sorted(t for t in tags if t_arm[t] == a) for a in arms},
         "arm_torso_tags": {a: sorted(t for t in links[root]["parts"] if t_arm[t] == a) for a in arms}}
    s["shared_occ"] = sorted(o for o, r in occ.items() if r["link"] == root and o not in drivers)
    return s


# ---------------------------------------------------------------------------------------------- the document's pieces
def _leaf_problem(leaf, extract_dir):
    if leaf.get("error"):
        return str(leaf["error"])[:160]
    if not leaf.get("valid"):
        return "not a valid solid (FreeCAD's check)"
    if leaf.get("n_solids") != 1:
        return "%s solids (one expected)" % leaf.get("n_solids")
    f = leaf.get("brep") or leaf.get("step")
    if not f or not os.path.isfile(os.path.join(extract_dir, f)):
        return "no shape file"
    return None


def _leaf_shape(leaf, extract_dir):
    f = leaf.get("brep") or leaf.get("step")
    if not f:
        return None, "no shape file"
    return C.load_shape(os.path.join(extract_dir, f))


def resolve_bought(leaf, model, classes, extract_dir, rules, seed):
    """{class, P (class frame in the document), how} of a bought leaf, or {class: None, why}."""
    tol = rules["tolerances"]
    cls = leaf.get("bought_class")
    if cls and cls in classes and not leaf.get("error"):
        top = (model.get("bought_classes") or {}).get(cls, {}).get("placement")
        P = _m(leaf["placement"])
        if top is not None:
            P = P @ C.inv(_m(top))
        return {"class": cls, "P": C.orthonormal(P), "how": "placement"}
    cands = [c for c in (leaf.get("bought_candidates") or []) if c in classes]
    if cls and cls in classes and cls not in cands:
        cands.append(cls)
    if not cands:
        return {"class": None, "why": "matches no bought class of the task"}
    shape, err = _leaf_shape(leaf, extract_dir)
    if err:
        return {"class": None, "why": err}
    lb = C.Body("leaf", shape, strict=False)
    if lb.mesh is None:
        return {"class": None, "why": "the leaf does not tessellate"}
    from scipy.spatial import cKDTree
    n_leaf, n_class = tol.get("registration_samples", [30000, 6000])
    dst, dn = C.sample_surface(lb.mesh, int(n_leaf), seed)
    tree = cKDTree(dst)
    best = None
    for c in sorted(cands):
        K = classes[c]
        src, _n = C.sample_surface(K.body.mesh, int(n_class), seed + 1)
        starts = C.pca_starts(src, dst) + [_m(leaf["placement"])]
        T, _rms = C.register(src, dst, dn, starts)
        resid = float(np.percentile(C.surface_distances(C.apply(T, src), dst, dn, tree), 90))
        if best is None or resid < best[2]:
            best = (c, T, resid)
    c, T, resid = best
    if resid > tol["registration_rms_mm"]:
        return {"class": None, "why": "its shape registers on no candidate class (%s; best %s at %.2f mm)" % (", ".join(cands), c, resid)}
    return {"class": c, "P": C.orthonormal(T), "how": "registered on the class geometry (p90 %.3f mm)" % resid}


# ---------------------------------------------------------------------------------------------- joints
def joint_axis(j, parent_group, uf):
    """(point, child-relative-to-parent axis, lo, hi, frame gap mm, frame tilt deg) of a Revolute joint record at the
    saved pose; lo / hi relative to it (None when not enabled)."""
    J1, J2 = _m(j["jcs1"]), _m(j["jcs2"])
    p, z1 = J1[:3, 3].copy(), C.unit(J1[:3, 2])
    theta0 = C.signed_angle(J1[:3, 0], J2[:3, 0], z1)
    lo = math.radians(float(j["angle_min_deg"])) - theta0 if j.get("enable_angle_min") else None
    hi = math.radians(float(j["angle_max_deg"])) - theta0 if j.get("enable_angle_max") else None
    a = z1 if uf.find(j["part1"]) == parent_group else -z1     # Reference1 is the child: the parent turns about +z1
    gap = float(np.linalg.norm(J2[:3, 3] - J1[:3, 3]))
    tilt = C.line_angle_deg(J1[:3, 2], J2[:3, 2])
    return p, a, lo, hi, gap, tilt


def orient(a, lo, hi, d):
    """The axis and limits turned so that the axis points along the reference-convention direction d."""
    if d is None or float(a @ d) >= 0:
        return a, lo, hi
    return -a, (-hi if hi is not None else None), (-lo if lo is not None else None)


def _aabb_distance(p, box):
    lo, hi = box[:3], box[3:]
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - p, p - hi))))


# ---------------------------------------------------------------------------------------------- torso alignment
def _samples_by_area(meshes, n_total, seed):
    """Area-proportional surface samples (and normals) over a list of meshes, seeded per mesh."""
    areas = np.array([max(float(m.area), 1e-9) for m in meshes])
    counts = np.maximum(200, np.round(n_total * areas / areas.sum()).astype(int))
    P, N = [], []
    for k, (m, n) in enumerate(zip(meshes, counts)):
        p, nn = C.sample_surface(m, int(n), seed + 7 * k)
        P.append(p)
        N.append(nn)
    return np.vstack(P), np.vstack(N)


def align_torso(ref_meshes, sub_meshes, T0, rules):
    """(T submission -> reference, stats): ICP of the submitted torso's samples onto the reference's from T0 (the column
    frames matched) and from the identity; the 24 principal-axis starts when the best trimmed RMS exceeds 3 mm; then
    point-to-plane on the flat regions only (checks.refine_planar), which has no curvature bias."""
    from scipy.spatial import cKDTree
    ra = rules["torso_alignment"]
    seed = int(ra["seed"])
    dst, dn = _samples_by_area(ref_meshes, int(ra["reference_samples"]), seed)
    src, src_n = _samples_by_area(sub_meshes, int(ra["samples"]), seed + 1)
    tree = cKDTree(dst)
    best = None
    for name, Ts in (("column frames", T0), ("identity", np.eye(4))):
        if Ts is None:
            continue
        T, rms = C.icp(src, dst, dn, tree, Ts, iters=60, restrict=False)
        if best is None or rms < best[1]:
            best = (T, rms, name)
    if best is None or best[1] > C.MULTISTART_RMS_MM:
        T, rms = C.register(src, dst, dn, C.pca_starts(src, dst), iters=60)
        if best is None or rms < best[1]:
            best = (T, rms, "principal axes")
    T, rms, start = best
    T, n_flat = C.refine_planar(src, src_n, dst, dn, tree, C.orthonormal(T), agree_deg=float(ra.get("flat_agree_deg", 0.05)))
    P = C.apply(T, src)
    _d, idx = tree.query(P)
    pp = np.abs(np.einsum("ij,ij->i", P - dst[idx], dn[idx]))          # point-to-plane: ~0 on a matched surface
    st = {"start": start, "trimmed_rms_mm": float(rms), "residual_median_mm": float(np.median(pp)),
          "residual_p90_mm": float(np.percentile(pp, 90)), "rotation_deg": float(C.rotation_angle_deg(T[:3, :3])),
          "translation_mm": float(np.linalg.norm(T[:3, 3])), "flat_correspondences": int(n_flat)}
    return T, st


# ---------------------------------------------------------------------------------------------- the conversion
def convert(extract_dir, ref_dir, bought_dir, out_dir, rules=None, classes=None):
    """Returns the conversion record {ok, errors, arms: {arm: {valid, errors, consistent, consistency}}, notes, stats,
    alignment}; on ok, out_dir holds unit.json and parts/<tag>.brep (also written: convert.json, the record)."""
    R = rules or C.load_rules()
    res = {"format": "cad-tree-convert/1", "ok": False, "errors": [], "arms": {}, "notes": [], "stats": {}, "alignment": {}}
    try:
        if classes is None:
            classes = C.load_classes(bought_dir)
        _convert(extract_dir, ref_dir, out_dir, R, classes, res)
    except Exception:  # noqa: BLE001 - a document that crashes the conversion earns cad_valid 0, with the trace
        res["ok"] = False
        res["errors"].append("conversion failed: " + traceback.format_exc()[-600:])
        res["exception"] = 1
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "convert.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    return res


def _convert(extract_dir, ref_dir, out_dir, R, classes, res):
    E, notes, st = res["errors"], res["notes"], res["stats"]
    tol = R["tolerances"]
    path = os.path.join(extract_dir, "model.json")
    if not os.path.isfile(path):
        E.append("the document could not be read (no extraction record)")
        return
    with open(path) as fh:
        model = json.load(fh)
    if model.get("format") != EXTRACT_FORMAT:
        notes.append("extraction format %s (expected %s)" % (model.get("format"), EXTRACT_FORMAT))
    if model.get("errors"):
        E.extend("document: %s" % str(e)[:300] for e in model["errors"][:10])
        return
    with open(os.path.join(ref_dir, "unit.json")) as fh:
        uref = json.load(fh)
    S = ref_structure(uref, R)
    arms = S["arms"]
    AE = {a: [] for a in arms}

    def owner(tag):
        a = S["tag_arm"].get(tag)
        return AE[a] if a in AE else E

    comps = {c["name"]: c for c in model.get("components", [])}
    by_tag, bought_leaves, other = collections.defaultdict(list), [], []
    for c in model.get("components", []):
        if c.get("error"):
            E.append("component %s: %s" % (c.get("label"), str(c["error"])[:160]))
        for leaf in c.get("leaves", []):
            leaf = dict(leaf, component=c["name"])
            label = (leaf.get("label") or "").strip()
            if label in S["tags"]:
                by_tag[label].append(leaf)
            elif leaf.get("bought_class") or leaf.get("bought_candidates"):
                bought_leaves.append(leaf)
            else:
                other.append(leaf)
    st["n_components"] = len(comps)
    st["n_leaves_other"] = len(other)
    if other:
        notes.append("%d leaves are neither a tagged part nor a bought part (not graded): %s" % (
            len(other), ", ".join(sorted(set((l.get("label") or "?") for l in other)))[:200]))
    parts = {}
    for tag, ls in sorted(by_tag.items()):
        if len(ls) > 1:
            vis = [l for l in ls if l.get("visible", True)]
            if len(vis) != 1:
                owner(tag).append("the label %s is used by %d bodies" % (tag, len(ls)))
                continue
            notes.append("the label %s is used by %d bodies: the visible one is graded" % (tag, len(ls)))
            ls = vis
        why = _leaf_problem(ls[0], extract_dir)
        if why:
            owner(tag).append("part %s: %s" % (tag, why))
            continue
        parts[tag] = ls[0]
    missing = sorted(t for t in S["tags"] if t not in by_tag)
    if missing:
        E.append("the Assembly has no body labelled %s" % ", ".join(missing))
    st["n_parts"] = len(parts)

    resolved = []
    for k, leaf in enumerate(bought_leaves):
        r = resolve_bought(leaf, model, classes, extract_dir, R, int(R["torso_alignment"]["seed"]) + 31 * k)
        if r["class"] is None:
            notes.append("leaf %s (%s): %s" % (leaf.get("label"), leaf["component"], r["why"]))
            continue
        r["leaf"] = leaf
        if r["how"] != "placement":
            notes.append("leaf %s: %s %s" % (leaf.get("label"), r["class"], r["how"]))
        resolved.append(r)
    have = collections.Counter(r["class"] for r in resolved)
    want = collections.Counter(o["class"] for o in S["occ"].values())
    hk, wk = _kinds(have, classes), _kinds(want, classes)
    for k in sorted(set(hk) | set(wk)):
        if hk[k] != wk[k]:
            E.append("the Assembly holds %d %s piece(s) (the unit has %d)" % (hk[k], k, wk[k]))
    for cls in sorted(set(have) | set(want)):
        if cls in classes and _is_motor(classes[cls]) and have[cls] != want[cls]:
            notes.append("the Assembly holds %d %s motor(s) (the unit has %d): matched joint by joint" % (have[cls], cls, want[cls]))
    st["n_bought"] = len(resolved)

    # ---- links (Fixed + Grounded), revolute joints
    names = set(comps)
    uf = _UF(names)
    revs, n_fixed, ignored = [], 0, collections.Counter()
    top_asm = model.get("assembly")
    for j in model.get("joints", []):
        label = j.get("label") or j.get("name")
        t = j.get("type")
        if j.get("suppressed"):
            ignored["suppressed"] += 1
            continue
        if j.get("assembly") and top_asm and j["assembly"] != top_asm:
            notes.append("joint %s belongs to the sub-assembly %s (a rigid component here): ignored" % (label, j["assembly"]))
            continue
        if t not in ("Fixed", "Revolute"):
            ignored[t] += 1
            continue
        if j.get("error"):
            E.append("joint %s cannot be read: %s" % (label, str(j["error"])[:160]))
            continue
        p1, p2 = j.get("part1"), j.get("part2")
        if p1 not in names or p2 not in names:
            E.append("joint %s does not join two Assembly components (%s, %s)" % (label, p1, p2))
            continue
        if p1 == p2:
            notes.append("joint %s joins a component to itself: ignored" % label)
            continue
        if t == "Fixed":
            uf.union(p1, p2)
            n_fixed += 1
        else:
            revs.append(j)
    for t, n in sorted(ignored.items()):
        notes.append("%d %s joint(s) ignored%s" % (n, t, " (coupling joints play no part)" if t in COUPLING_TYPES else
                                                    "" if t == "suppressed" else " (not Fixed or Revolute)"))
    st.update(n_fixed=n_fixed, n_revolute=len(revs))
    grounded = [g for g in model.get("grounded", []) if g]
    if not grounded:
        E.append("no Grounded joint (the torso must be grounded)")
    for g in grounded:
        if g not in names:
            E.append("the grounded object %s is not an Assembly component" % g)
    if E:
        return
    for g in grounded[1:]:
        uf.union(grounded[0], g)
    g0 = uf.find(grounded[0])
    groups = collections.defaultdict(lambda: {"components": [], "tags": set(), "bought": []})
    for n in sorted(names):
        groups[uf.find(n)]["components"].append(n)
    for tag, leaf in parts.items():
        groups[uf.find(leaf["component"])]["tags"].add(tag)
    for r in resolved:
        groups[uf.find(r["leaf"]["component"])]["bought"].append(r)

    def desc(g):
        gg = groups[g]
        return "the link of %s" % (", ".join(sorted(gg["tags"])) or _fmt_counts(collections.Counter(r["class"] for r in gg["bought"])))

    for t in S["shared_tags"]:
        if t in parts and uf.find(parts[t]["component"]) != g0:
            E.append("the grounded link does not hold %s (no grounded torso; %s is in %s)" % (t, t, desc(uf.find(parts[t]["component"]))))
    for oid in S["shared_occ"]:
        cls = S["occ"][oid]["class"]
        n_ref = sum(1 for o in S["shared_occ"] if S["occ"][o]["class"] == cls)
        if sum(1 for r in groups[g0]["bought"] if r["class"] == cls) < n_ref:
            E.append("the grounded torso does not hold the %s piece" % cls)
    if E:
        return

    rev_by_pair = collections.defaultdict(list)
    for j in revs:
        a, b = uf.find(j["part1"]), uf.find(j["part2"])
        if a == b:
            notes.append("revolute joint %s joins two components of one rigid link: ignored" % (j.get("label") or j["name"]))
            continue
        rev_by_pair[frozenset((a, b))].append(j)

    # ---- per arm: links, joints, motors, senses, limits, consistency
    leaf_box = {}

    def part_box(tag):
        if tag not in leaf_box:
            shp, err = _leaf_shape(parts[tag], extract_dir)
            leaf_box[tag] = None if err else _bbox(shp)
        return leaf_box[tag]

    def group_box(g):
        boxes = [part_box(t) for t in groups[g]["tags"]]
        boxes += [C.aabb_of(r["P"], classes[r["class"]].body.bounds) for r in groups[g]["bought"]]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return None
        B = np.array(boxes)
        return np.concatenate([B[:, :3].min(axis=0), B[:, 3:].max(axis=0)])

    taken = {g0}
    out_arms = {arm: {"links": {S["root"]: g0}, "joints": {}} for arm in arms}
    membership = tol.get("link_membership", "strict")
    case_side = tol.get("case_side", "strict")

    def held(counter):
        """The bought kinds a link is checked on; with case_side "reported" the motors are left out (each is matched to
        its joint on either side of it, and the unit-wide count holds them)."""
        k = _kinds(counter, classes)
        if case_side == "reported":
            k.pop("motor", None)
        return k
    misplaced = {arm: [] for arm in arms}
    for arm in arms:                                   # links by tag set
        errs, m = AE[arm], out_arms[arm]["links"]
        if membership == "majority":
            partless = set(S["occ"][b["id"]]["class"] for ln in S["arm_links"][arm] if not S["links"][ln]["parts"]
                           for b in S["links"][ln]["bought"])
            modules = set(g for g in groups if any(r["class"] in partless for r in groups[g]["bought"]))
            _links_by_share(arm, S, parts, uf, groups, g0, m, taken, errs, desc, modules)
            continue
        for t in S["arm_torso_tags"][arm]:
            if t in parts and uf.find(parts[t]["component"]) != g0:
                errs.append("%s is not fixed to the grounded torso (it is in %s)" % (t, desc(uf.find(parts[t]["component"]))))
        for ln in S["arm_links"][arm]:
            L = S["links"][ln]
            if not L["parts"] or any(t not in parts for t in L["parts"]):
                continue
            roots = sorted(set(uf.find(parts[t]["component"]) for t in L["parts"]))
            if len(roots) > 1:
                errs.append("the parts of %s (%s) are not one rigid link: %s" % (ln, ", ".join(L["parts"]), "; ".join(desc(g) for g in roots)))
                continue
            g = roots[0]
            if g == g0:
                errs.append("%s (%s) is fixed to the torso" % (ln, ", ".join(L["parts"])))
                continue
            extra = sorted(groups[g]["tags"] - set(L["parts"]))
            if extra:
                errs.append("the link holding %s also holds %s (the unit's %s holds %s only)" % (
                    ", ".join(L["parts"]), ", ".join(extra), ln, ", ".join(L["parts"])))
                continue
            m[ln] = g
            taken.add(g)
    for arm in arms:                                   # part-less links: the bought classes, hanging from the parent
        errs, m = AE[arm], out_arms[arm]["links"]
        progress = True
        while progress:
            progress = False
            for jn in S["arm_joints"][arm]:
                J = S["joints"][jn]
                if J["child"] in m or J["parent"] not in m or S["links"][J["child"]]["parts"]:
                    continue
                want_c = held(collections.Counter(S["occ"][b["id"]]["class"] for b in S["links"][J["child"]]["bought"]))
                gp = m[J["parent"]]
                cands = set()
                for key in rev_by_pair:
                    if gp in key and len(key) == 2:
                        (oth,) = tuple(key - {gp})
                        if oth in taken or (groups[oth]["tags"] and membership != "majority"):
                            continue
                        if held(collections.Counter(r["class"] for r in groups[oth]["bought"])) == want_c:
                            cands.add(oth)
                if len(cands) == 1:
                    m[J["child"]] = cands.pop()
                    taken.add(m[J["child"]])
                    progress = True
        for ln in S["arm_links"][arm]:
            L = S["links"][ln]
            if ln in m or L["parts"]:
                continue
            par = [S["joints"][jn]["parent"] for jn in S["arm_joints"][arm] if S["joints"][jn]["child"] == ln]
            want_c = collections.Counter(S["occ"][b["id"]]["class"] for b in L["bought"])
            errs.append("no link holding %s hangs from %s by a revolute joint (the unit's %s)" % (
                _fmt_counts(want_c), desc(m[par[0]]) if par and par[0] in m else (par[0] if par else "?"), ln))
    if membership == "majority":                       # every link matched: where each part ended up
        linked = set(g for arm in arms for g in out_arms[arm]["links"].values())
        for arm in arms:
            at = {g: ln for ln, g in out_arms[arm]["links"].items()}
            home = {t: S["root"] for t in S["arm_torso_tags"][arm]}
            home.update({t: ln for ln in S["arm_links"][arm] for t in S["links"][ln]["parts"]})
            for t in sorted(home):
                if t not in parts:
                    continue
                g = uf.find(parts[t]["component"])
                if g not in linked:
                    AE[arm].append("%s is on a link the unit does not have (%s)" % (t, desc(g)))
                elif at.get(g) != home[t]:
                    misplaced[arm].append("%s (the unit's %s) is in %s" % (t, home[t], at.get(g) or desc(g)))
    arm_of_group = {g: arm for arm in arms for ln, g in out_arms[arm]["links"].items() if ln != S["root"]}
    used_j = set()
    for arm in arms:                                   # bought classes per link, joints
        errs, m, jm = AE[arm], out_arms[arm]["links"], out_arms[arm]["joints"]
        for ln, g in m.items():
            if ln == S["root"]:
                continue
            want_c = held(collections.Counter(S["occ"][b["id"]]["class"] for b in S["links"][ln]["bought"]))
            have_c = held(collections.Counter(r["class"] for r in groups[g]["bought"]))
            if have_c != want_c:
                errs.append("%s holds %s; the unit's %s holds %s" % (desc(g), _fmt_counts(have_c), ln, _fmt_counts(want_c)))
        for jn in S["arm_joints"][arm]:
            J = S["joints"][jn]
            if J["parent"] not in m or J["child"] not in m:
                continue
            js = rev_by_pair.get(frozenset((m[J["parent"]], m[J["child"]])), [])
            if not js:
                errs.append("no revolute joint connects %s and %s (the unit's %s)" % (desc(m[J["parent"]]), desc(m[J["child"]]), jn))
            elif len(js) > 1:
                errs.append("%d revolute joints connect %s and %s (the unit's %s)" % (len(js), desc(m[J["parent"]]), desc(m[J["child"]]), jn))
            else:
                jm[jn] = js[0]
                used_j.add(id(js[0]))
    for key, js in sorted(rev_by_pair.items(), key=lambda kv: sorted(kv[0])):
        for j in js:
            if id(j) in used_j:
                continue
            label = j.get("label") or j["name"]
            touched = sorted(set(arm_of_group[g] for g in key if g in arm_of_group))
            unknown = [g for g in key if g not in arm_of_group and g != g0]
            if not touched or unknown:
                notes.append("revolute joint %s joins %s, a link the unit does not have: not graded" % (label, " and ".join(desc(g) for g in sorted(key))))
                continue
            for a in touched:
                AE[a].append("revolute joint %s connects %s: the unit has no joint there" % (label, " and ".join(desc(g) for g in sorted(key))))

    # ---- motors, senses, limits, consistency (per arm)
    motor_of = {}
    flipped = {arm: [] for arm in arms}
    for arm in arms:
        errs = AE[arm]
        m, jm = out_arms[arm]["links"], out_arms[arm]["joints"]
        conv, cons = {}, []
        names_sorted = sorted(jm, key=lambda n: (JOINT_ORDER.get(S["joints"][n]["kind"], 9), n))
        for jn in names_sorted:
            J, j = S["joints"][jn], jm[jn]
            gp, gc = m[J["parent"]], m[J["child"]]
            p, a, lo, hi, gap, tilt = joint_axis(j, gp, uf)
            rec = {"name": jn, "arm": arm, "kind": J["kind"], "parent": J["parent"], "child": J["child"], "tree": bool(J.get("tree", True)),
                   "driven_by": J.get("driven_by"), "case_on": None, "source": j.get("label") or j["name"]}
            if gap > tol["consistency_mm"] or tilt > tol["consistency_deg"]:
                cons.append("%s: its two joint frames are %.2f mm / %.2f deg apart at the saved pose" % (jn, gap, tilt))
            d = None
            if J["kind"] == "motor":
                oid = J["driven_by"]
                cls = S["occ"][oid]["class"]
                cands, near = [], None
                for side, ln in (("parent", J["parent"]), ("child", J["child"])):
                    for r in groups[m[ln]]["bought"]:
                        if not _is_motor(classes[r["class"]]) or id(r) in [id(x) for x in motor_of.values()]:
                            continue
                        c, z = classes[r["class"]].output_centre(r["P"])
                        ang, dist = C.line_angle_deg(z, a), C.point_line_distance(c, p, a)
                        if near is None or (dist, ang) < near:
                            near = (dist, ang)
                        if ang > tol["motor_match_deg"] or dist > tol["motor_match_mm"]:
                            continue
                        box = group_box(m[J["child"] if side == "parent" else J["parent"]])
                        prox = _aabb_distance(c, box) if box is not None else 0.0
                        cands.append((prox > float(tol.get("motor_near_driven_mm", 2.0)), r["class"] != cls, prox, dist, ang, side, r))
                if not cands:
                    errs.append("%s: no motor on %s or %s whose axis lies on the joint's axis%s" % (
                        jn, J["parent"], J["child"], " (nearest %.1f mm / %.1f deg off)" % near if near else ""))
                    continue
                cands.sort(key=lambda x: x[:5])
                _f, _other, _prox, dist, ang, side, r = cands[0]
                K = classes[r["class"]]
                if r["class"] != cls:
                    rec["motor_class"] = r["class"]
                    notes.append("%s: driven by a %s motor where the unit has a %s (scored in component_fit)" % (jn, r["class"], cls))
                if side != J["case_on"]:
                    msg = "%s: the %s motor's case rides on the %s (%s); the unit's on the %s" % (jn, cls, side, J[side], J["case_on"])
                    if case_side != "reported":
                        errs.append(msg)
                        continue
                    flipped[arm].append(msg)
                motor_of[oid] = r
                c, z = K.output_centre(r["P"])
                d = z if side == "parent" else -z
                rec["case_on"] = side
                if ang > tol["consistency_deg"] or dist > tol["consistency_mm"]:
                    cons.append("%s: the joint axis is %.2f mm / %.2f deg off its motor's axis" % (jn, dist, ang))
            elif J["kind"] == "linkage":
                oid = J["driven_by"]
                r = motor_of.get(oid)
                if r is None:
                    errs.append("%s: its driver %s drives no matched motor joint" % (jn, oid))
                    continue
                K = classes[r["class"]]
                _c, z = K.output_centre(r["P"])
                ang = C.line_angle_deg(z, a)
                if ang > tol["motor_match_deg"]:
                    errs.append("%s: the joint axis is %.1f deg off its driving motor's axis" % (jn, ang))
                    continue
                case_link = S["occ"][oid]["link"]
                if case_side == "reported":            # the link its motor's case rides on in the document
                    mj = next((n for n, x in conv.items() if x["kind"] == "motor" and x.get("driven_by") == oid), None)
                    if mj is not None:
                        case_link = conv[mj][conv[mj]["case_on"]]
                d = z if case_link == J["parent"] else -z
                rec["case_on"] = "parent" if case_link == J["parent"] else "child"
                if ang > tol["consistency_deg"]:
                    cons.append("%s: the joint axis is %.2f deg off its driving motor's axis" % (jn, ang))
            elif J["kind"] == "finger":
                piece = _piece_on(groups[gp], S, J["parent"])
                if piece is None:
                    errs.append("%s: no gripper module on %s" % (jn, J["parent"]))
                    continue
                Tm = piece["P"] @ C.inv(S["occ"][piece["ref_id"]]["placement"])      # reference module pose -> submitted
                hp, hd = C.apply(Tm, J["axis_point"]), Tm[:3, :3] @ np.asarray(J["axis_dir"], float)
                ang, dist = C.line_angle_deg(hd, a), C.point_line_distance(hp, p, a)
                if ang > tol["finger_match_deg"] or dist > tol["finger_match_mm"]:
                    errs.append("%s: the joint axis is %.1f mm / %.1f deg off the module's hinge" % (jn, dist, ang))
                    continue
                d = hd
                if ang > tol["consistency_deg"] or dist > tol["consistency_mm"]:
                    cons.append("%s: the joint axis is %.2f mm / %.2f deg off the module's hinge" % (jn, dist, ang))
            elif J["kind"] == "pin":
                crank = [n for lp in uref.get("loops", []) if jn in lp["joints"] for n in lp["joints"][:1]]
                if crank and crank[0] in conv:
                    d = np.asarray(conv[crank[0]]["axis_dir"], float)
                    ang = C.line_angle_deg(d, a)
                    if ang > tol["consistency_deg"]:
                        cons.append("%s: the pin is %.2f deg off parallel to the crank" % (jn, ang))
            a, lo, hi = orient(a, lo, hi, d)
            rec.update(axis_point=p, axis_dir=a, limits=[lo, hi] if (lo is not None or hi is not None) else None)
            conv[jn] = rec
        # a parallelogram turns its driven joint by the crank's angle: limits set on the crank's motor joint (and none on
        # the driven one) are the driven joint's limits (either joint of the loop is a fair place to set them)
        for lp in uref.get("loops", []):
            jc, jd = lp["joints"][0], lp["joints"][-1]
            if lp.get("kind") == "parallelogram" and jc in conv and jd in conv and conv[jd].get("limits") is None \
                    and conv[jc].get("limits") is not None:
                conv[jd]["limits"] = list(conv[jc]["limits"])
                notes.append("%s: no limits; those of %s taken (the parallelogram turns both alike)" % (jd, jc))
        out_arms[arm]["conv"] = conv
        out_arms[arm]["consistency"] = cons

    for arm in arms:
        valid = not AE[arm] and len(out_arms[arm]["conv"]) == len(S["arm_joints"][arm])
        if not AE[arm] and not valid:
            AE[arm].append("%d of the arm's %d joints were found" % (len(out_arms[arm]["conv"]), len(S["arm_joints"][arm])))
        res["arms"][arm] = {"valid": bool(valid), "errors": AE[arm][:12], "consistent": bool(valid and not out_arms[arm]["consistency"]),
                            "consistency": out_arms[arm]["consistency"][:12]}
        if membership == "majority":
            res["arms"][arm]["misplaced"] = misplaced[arm][:12]
            st["n_misplaced_%s" % arm] = len(misplaced[arm])
        if case_side == "reported":
            res["arms"][arm]["case_flipped"] = flipped[arm][:12]
            st["n_case_flipped_%s" % arm] = len(flipped[arm])
    if E:
        return

    # ---- occurrence ids of the bought pieces (motors by the joints they drive, the rest by class within the link)
    occ_id = {}
    for oid, r in motor_of.items():
        occ_id[id(r)] = oid
    link_of_group = {}
    for arm in arms:
        if res["arms"][arm]["valid"]:
            for ln, g in out_arms[arm]["links"].items():
                link_of_group[g] = ln
    link_of_group[g0] = S["root"]
    for g, ln in link_of_group.items():
        refs = [b["id"] for b in S["links"][ln]["bought"] if b["id"] not in motor_of]
        for r in groups[g]["bought"]:
            if id(r) in occ_id:
                continue
            same = [o for o in refs if S["occ"][o]["class"] == r["class"] and o not in occ_id.values()]
            if same:
                occ_id[id(r)] = same[0]
    k_extra = collections.Counter()
    for g in groups:
        for r in groups[g]["bought"]:
            if id(r) not in occ_id:
                k_extra[r["class"]] += 1
                occ_id[id(r)] = "x_%s_%d" % (r["class"], k_extra[r["class"]])

    # ---- the frame: the submitted torso onto the reference's
    ref_meshes, sub_meshes = [], []
    ref_parts = {}
    for t in S["links"][S["root"]]["parts"]:
        if t not in parts or uf.find(parts[t]["component"]) != g0:
            continue
        shp, err = C.load_shape(os.path.join(ref_dir, "parts", t + ".brep"))
        if err:
            shp, err = C.load_shape(os.path.join(ref_dir, "parts", t + ".step"))
        if err:
            raise RuntimeError("reference part %s: %s" % (t, err))
        sshp, serr = _leaf_shape(parts[t], extract_dir)
        if serr:
            continue
        rb, sb = C.Body(t, shp, strict=False), C.Body(t, sshp, strict=False)
        if rb.mesh is None or sb.mesh is None:
            continue
        ref_parts[t] = rb
        ref_meshes.append(rb.mesh)
        sub_meshes.append(sb.mesh)
    T0 = None
    for r in groups[g0]["bought"]:
        oid = occ_id[id(r)]
        if oid not in S["occ"]:
            continue
        K = classes[r["class"]]
        mr, ms = K.body.mesh.copy(), K.body.mesh.copy()
        mr.apply_transform(S["occ"][oid]["placement"])
        ms.apply_transform(r["P"])
        ref_meshes.append(mr)
        sub_meshes.append(ms)
        if oid in S["shared_occ"] and T0 is None:
            T0 = S["occ"][oid]["placement"] @ C.inv(r["P"])
    T, ast = align_torso(ref_meshes, sub_meshes, T0, R)
    res["alignment"] = ast
    st["torso_alignment_p90_mm"] = ast["residual_p90_mm"]

    # ---- write: parts (aligned), unit.json
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    for tag, leaf in sorted(parts.items()):
        shp, err = _leaf_shape(leaf, extract_dir)
        if err:
            owner(tag).append("part %s: %s" % (tag, err))
            continue
        C.write_brep(C.transform_shape(shp, T), os.path.join(out_dir, "parts", tag + ".brep"))
    links_out, joints_out, loops_out, tcp_out, tips_out = [], [], [], {}, {}
    for g, ln in sorted(link_of_group.items(), key=lambda kv: kv[1]):
        L = S["links"][ln]
        tags = sorted(groups[g]["tags"]) if ln == S["root"] or membership == "majority" else list(L["parts"])
        bought = [{"id": occ_id[id(r)], "class": r["class"], "placement": (T @ r["P"]).tolist()} for r in groups[g]["bought"]]
        links_out.append({"name": ln, "arm": L.get("arm"), "parts": tags, "bought": sorted(bought, key=lambda b: b["id"])})
    for arm in arms:
        if not res["arms"][arm]["valid"]:
            continue
        for jn, rec in sorted(out_arms[arm]["conv"].items()):
            r2 = dict(rec)
            r2["axis_point"] = C.apply(T, rec["axis_point"]).tolist()
            r2["axis_dir"] = C.unit(T[:3, :3] @ np.asarray(rec["axis_dir"], float)).tolist()
            joints_out.append(r2)
        for lp in uref.get("loops", []):
            if lp.get("arm") != arm:
                continue
            conv = out_arms[arm]["conv"]
            jc, j1, j2 = (conv[n] for n in lp["joints"][:3])
            loops_out.append({"arm": arm, "joints": list(lp["joints"]), "kind": lp.get("kind"), "ratio": float(lp.get("ratio", 1.0)),
                              "crank_mm": C.line_line_distance(jc["axis_point"], jc["axis_dir"], j1["axis_point"], j1["axis_dir"]),
                              "coupler_mm": C.line_line_distance(j1["axis_point"], j1["axis_dir"], j2["axis_point"], j2["axis_dir"])})
        tr = uref.get("tcp", {}).get(arm)
        if tr:
            piece = _piece_on(groups[out_arms[arm]["links"][tr["link"]]], S, tr["link"])
            if piece is not None:
                Tm = T @ piece["P"] @ C.inv(S["occ"][piece["ref_id"]]["placement"])
                tcp_out[arm] = {"link": tr["link"], "point": C.apply(Tm, tr["point"]).tolist()}
        for key, pt in uref.get("fingertips", {}).items():
            if key in out_arms[arm]["links"]:
                piece = _piece_on(groups[out_arms[arm]["links"][key]], S, key)
                if piece is not None:
                    Tm = T @ piece["P"] @ C.inv(S["occ"][piece["ref_id"]]["placement"])
                    tips_out[key] = C.apply(Tm, pt).tolist()
    unit = {"format": UNIT_FORMAT, "units": {"length": "mm", "angle": "rad"},
            "frame": "B at configuration Z (the submission, its torso aligned on the reference torso)",
            "links": links_out, "joints": [_jsonable(j) for j in joints_out], "loops": loops_out, "tcp": tcp_out,
            "fingertips": tips_out,
            "submission": {"arms": res["arms"], "alignment": ast, "T_document_to_B": T.tolist()}}
    with open(os.path.join(out_dir, "unit.json"), "w") as fh:
        json.dump(unit, fh, indent=1)
    res["ok"] = not E


def _links_by_share(arm, S, parts, uf, groups, g0, m, taken, errs, desc, modules=()):
    """rules.json tolerances.link_membership = "majority": each of the arm's reference links with parts is the rigid link
    holding the largest share of its parts' volume (every rigid link taken once, the largest shares first; never the
    torso, nor a rigid link holding a gripper piece of the arm: `modules`, matched afterwards as the part-less links).
    A reference link whose parts give it no rigid link of its own is an error; where each part ended up is reported
    by the caller once every link is matched."""
    share = []
    for ln in S["arm_links"][arm]:
        L = S["links"][ln]
        if not L["parts"] or any(t not in parts for t in L["parts"]):
            continue
        vol = collections.defaultdict(float)
        for t in L["parts"]:
            vol[uf.find(parts[t]["component"])] += float(parts[t].get("volume") or 0.0)
        share += [(-v, ln, g) for g, v in vol.items() if g != g0 and g not in modules]
    for _v, ln, g in sorted(share):
        if ln not in m and g not in taken:
            m[ln] = g
            taken.add(g)
    for ln in S["arm_links"][arm]:
        L = S["links"][ln]
        if L["parts"] and ln not in m and all(t in parts for t in L["parts"]):
            errs.append("no rigid link of its own holds the parts of %s (%s): %s" % (
                ln, ", ".join(L["parts"]), "; ".join(desc(g) for g in sorted(set(uf.find(parts[t]["component"]) for t in L["parts"])))))


def _piece_on(group, S, link):
    """The bought piece of a matched link that carries the link's reference occurrence (by class): {P, class, ref_id}."""
    refs = S["links"][link]["bought"]
    for b in refs:
        for r in group["bought"]:
            if r["class"] == b["class"]:
                return {"P": r["P"], "class": r["class"], "ref_id": b["id"]}
    return None


def _bbox(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box()
    BRepBndLib.Add_s(shape, b)
    return np.array(b.Get(), float)


def _jsonable(rec):
    out = {}
    for k, v in rec.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            out[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
        else:
            out[k] = v
    return out
