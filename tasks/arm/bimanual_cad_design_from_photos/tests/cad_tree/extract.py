"""Read a submitted FreeCAD document of the bimanual unit into plain data (the grader's only contact with FreeCAD).

Runs under FreeCAD 1.1.4's headless interpreter as an unprivileged account (the verify stage starts it with a time
limit); it imports FreeCAD's own modules and the standard library only:

    freecadcmd -c "import sys; sys.argv = ['extract.py', FCSTD, BOUGHT_DIR, OUT_DIR]; exec(open(PATH).read())"

BOUGHT_DIR holds <class>.step for every bought class, each in its class frame (CONTRACTS 5). The document is opened and
recomputed with the Assembly's solve-on-recompute turned off: the placements are read as saved (configuration Z is the
saved pose; FreeCAD 1.1.4 would otherwise re-solve, and move the parts of, a document saved unsolved, which then shows
here as joints whose two frames do not coincide). Its top-level Assembly is found, and OUT_DIR/model.json (format
"bimanual-fcstd-extract/1": the worked example's "fcstd-extract/1" plus the per-leaf class fields) is written with
shapes/NNN.step and .brep for every shape-bearing leaf of every assembly component, in world coordinates (the BREP beside
the STEP because a STEP round trip can turn a valid solid invalid: skill pitfalls 1):

  bought_classes  per class: its STEP's volume, sorted extents and centroid (of the shape with its top Placement reset,
                  the same terms as a leaf's) and that top placement (identity for a single-body STEP), the reference
                  of the match.
  components      the Assembly's members (App::Link, Body, feature, rigid App::Part, nested assembly; the members of
                  document folders count, as in FreeCAD's getComponentReference) and their leaves: label, type, path,
                  visible, valid, n_solids, volume, placement (the global 4x4), extents_local and centroid_local (of the
                  leaf's shape with its top Placement reset: the sorted extents of its exact bounding box, the
                  volume-weighted centroid of its solids), bought_class (the one class whose STEP matches: volume
                  within 1 %, sorted extents within 0.5 mm and local centroid within 1.0 mm; null if none; two or more
                  matches are the leaf's error) and bought_candidates (the classes matching by volume and extents alone,
                  CONTRACTS 7's rule; the centroid is what tells the mirrored left / right gripper classes apart).
  joints          every Assembly joint object (Fixed, Revolute, Gears, RackPinion, Belt, Slider, ...): type (the
                  JointType string), suppressed, part1 / part2 = the assembly components the two references attach to
                  (references rooted at the component, or at the Assembly with the component in the subname: skill
                  pitfalls 2), jcs1 / jcs2 = the global joint frames (UtilsAssembly.getJcsGlobalPlc), the angle limits as
                  stored (degrees; FreeCAD's angle is Reference2's part relative to Reference1's about jcs1's z), and
                  the assembly that holds the joint.
  grounded        the objects of the Grounded joints.

Nothing in the document is executed beyond what FreeCAD runs to open and recompute it. Errors are recorded in model.json,
never raised; a model.json saying the extraction did not finish is written first, so a run killed by the time limit
(a document whose dependency graph has a cycle makes FreeCAD 1.1.4 spin while opening it) leaves a reason.
"""
import json
import os
import sys
import time
import traceback

import FreeCAD as App
import Part

FORMAT = "bimanual-fcstd-extract/1"
BOUGHT_VOLUME_REL_TOL = 0.01
BOUGHT_EXTENT_TOL_MM = 0.5
BOUGHT_CENTROID_TOL_MM = 1.0
MAX_DEPTH = 12                      # nesting of folders / parts / links followed (a cycle cannot recurse forever)
JOINT_CONTAINERS = ("Assembly::JointGroup", "Assembly::ViewGroup", "Assembly::BomGroup", "Assembly::SimulationGroup")


def matrix(placement):
    m = placement.toMatrix()
    return [[m.A11, m.A12, m.A13, m.A14], [m.A21, m.A22, m.A23, m.A24], [m.A31, m.A32, m.A33, m.A34], [0.0, 0.0, 0.0, 1.0]]


def extents(shape):
    """Sorted extents of the exact bounding box: BRepBndLib::AddOptimal on the geometry, without a triangulation (the
    BoundBox property is loose on B-spline faces by up to mm, and tight where a triangulation was saved with the shape,
    so it would compare a GUI document's leaf and the class STEP on different terms)."""
    try:
        b = shape.optimalBoundingBox(False, False)
    except Exception:  # noqa: BLE001 - an empty or odd shape: the plain box
        b = shape.BoundBox
    return sorted([float(b.XLength), float(b.YLength), float(b.ZLength)])


def centroid(shape):
    """Volume-weighted centre of the shape's solids, in the shape's frame (None when it has no volume)."""
    tot, acc = 0.0, App.Vector()
    for s in shape.Solids:
        v = s.Volume
        tot += v
        acc = acc + s.CenterOfMass * v
    return [acc.x / tot, acc.y / tot, acc.z / tot] if tot > 1e-9 else None


def describe(shape):
    """(volume, sorted extents, centroid) of a shape in its own frame: the terms of the class match."""
    return float(shape.Volume), extents(shape), centroid(shape)


def match_bought(vol, ext, cen, classes):
    """(classes matching volume + extents + centroid, classes matching volume + extents)."""
    hits, cands = [], []
    for cls in sorted(classes):
        ref = classes[cls]
        if "error" in ref or abs(vol - ref["volume"]) > BOUGHT_VOLUME_REL_TOL * abs(ref["volume"]):
            continue
        if any(abs(a - b) > BOUGHT_EXTENT_TOL_MM for a, b in zip(ext, ref["extents"])):
            continue
        cands.append(cls)
        d2 = sum((a - b) ** 2 for a, b in zip(cen, ref["centroid"])) if cen is not None and ref["centroid"] is not None else None
        if d2 is not None and d2 <= BOUGHT_CENTROID_TOL_MM ** 2:
            hits.append(cls)
    return hits, cands


def read_classes(bought_dir):
    classes = {}
    names = sorted(f for f in os.listdir(bought_dir) if f.lower().endswith((".step", ".stp")))
    for f in names:
        cls = os.path.splitext(f)[0]
        try:
            shape = Part.read(os.path.join(bought_dir, f))
            top = matrix(shape.Placement)       # identity for a single-body STEP; a leaf's placement includes it
            shape.Placement = App.Placement()
            vol, ext, cen = describe(shape)
            classes[cls] = {"file": f, "volume": vol, "extents": ext, "centroid": cen, "n_solids": len(shape.Solids),
                            "placement": top}
        except Exception as exc:  # noqa: BLE001
            classes[cls] = {"file": f, "error": "the class STEP does not read: %s" % exc}
    return classes


def leaf_label(obj):
    target = obj.getLinkedObject(True) if hasattr(obj, "getLinkedObject") else obj
    return (target.Label or "").strip(), target


def is_joint_container(obj):
    return obj.TypeId in JOINT_CONTAINERS


def is_joint(obj):
    return obj.TypeId == "App::FeaturePython" and (hasattr(obj, "JointType") or hasattr(obj, "ObjectToGround"))


def skipped(obj):
    return is_joint_container(obj) or is_joint(obj) or obj.TypeId.startswith("App::Origin")


def components_of(asm):
    """(folder prefix, object) of every Assembly component: the Assembly's members, document folders
    (App::DocumentObjectGroup) descended into; joints, joint containers and origins skipped. A folder's members are
    normally members of the Assembly as well (a geometry group owns everything below it; their subname path skips the
    folder), so each object is listed once, at the shallowest level it appears."""
    out, seen = [], set()

    def walk(group, prefix, depth):
        folders = []
        for child in group.Group:
            if skipped(child) or child.Name in seen:
                continue
            if child.TypeId == "App::DocumentObjectGroup":
                folders.append(child)
                continue
            seen.add(child.Name)
            out.append((prefix, child))
        for f in folders:
            if depth < MAX_DEPTH and f.Name not in seen:
                seen.add(f.Name)
                walk(f, prefix + f.Name + ".", depth + 1)
    walk(asm, "", 0)
    return out


def leaves(obj, prefix, depth=0):
    """(subname path, leaf object) of every shape-bearing leaf under an assembly component: the component itself when it
    is a body / feature / link to one, else the shape-bearing members of a rigid sub-assembly (App::Part, a nested
    Assembly)."""
    target = obj.getLinkedObject(True) if hasattr(obj, "getLinkedObject") else obj
    if target.TypeId in ("App::Part", "Assembly::AssemblyObject"):
        out = []
        if depth < MAX_DEPTH:
            for child in target.Group:
                if not skipped(child):
                    out += leaves(child, prefix + child.Name + ".", depth + 1)
        return out
    if hasattr(target, "Shape") and not target.Shape.isNull():
        return [(prefix, obj)]
    return []


def moving_part(UtilsAssembly, asm, components, ref):
    """The assembly component a joint reference attaches to. FreeCAD 1.1's getMovingPart returns the reference's object,
    which is the component only for references rooted at it ((L05_link, ["Face1"]), the form FreeCAD 1.1.4's joint tools
    write); a reference rooted at the Assembly with the component in its subname ((Assembly, ["L05_link.Face1"]), the
    older form) is resolved through getComponentReference. Its global joint frame is the same either way
    (getJcsGlobalPlc)."""
    part = UtilsAssembly.getMovingPart(ref)
    if part is not None and part.Name in components:
        return part
    sub = ref[1][0] if ref is not None and len(ref) == 2 and ref[1] else ""
    if ref is None or len(ref) != 2:
        return part
    comp, _rel = UtilsAssembly.getComponentReference(asm, ref[0], sub)
    return comp if comp is not None else part


def owning_assembly(obj):
    """The Assembly that holds a joint (its InList chain, a few levels up), or None."""
    seen, level = set(), [obj]
    for _ in range(4):
        nxt = []
        for o in level:
            for p in o.InList:
                if p.TypeId == "Assembly::AssemblyObject":
                    return p.Name
                if p.Name not in seen:
                    seen.add(p.Name)
                    nxt.append(p)
        level = nxt
    return None


def write_model(model, out_dir):
    with open(os.path.join(out_dir, "model.json"), "w") as fh:
        json.dump(model, fh, indent=1, sort_keys=True)


def extract(fcstd, bought_dir, out_dir):
    """The model of the document (see the module doc). The saved pose is configuration Z, so the recompute must not run
    the Assembly solver: FreeCAD 1.1.4 solves on recompute by default, which moves the parts of a document saved unsolved
    and can run long on a badly constrained one; the preference is turned off for the extraction and restored after."""
    grp = App.ParamGet("User parameter:BaseApp/Preferences/Mod/Assembly")
    had = grp.GetBool("SolveOnRecompute", True)
    grp.SetBool("SolveOnRecompute", False)
    try:
        return _extract(fcstd, bought_dir, out_dir)
    finally:
        grp.SetBool("SolveOnRecompute", had)


def _extract(fcstd, bought_dir, out_dir):
    t0 = time.time()
    model = {"format": FORMAT, "errors": [], "components": [], "joints": [], "grounded": [], "bought_classes": {}}
    os.makedirs(os.path.join(out_dir, "shapes"), exist_ok=True)
    try:
        model["bought_classes"] = read_classes(bought_dir)
    except Exception as exc:  # noqa: BLE001
        model["errors"].append("the bought classes cannot be read from %s: %s" % (bought_dir, exc))
    classes = model["bought_classes"]
    try:
        doc = App.openDocument(fcstd)
    except Exception as exc:  # noqa: BLE001
        model["errors"].append("the document does not open: %s" % exc)
        return model
    model["freecad"] = ".".join(App.Version()[:3])
    try:
        doc.recompute()
    except Exception as exc:  # noqa: BLE001
        model["errors"].append("the document does not recompute: %s" % exc)
    assemblies = [o for o in doc.Objects if o.TypeId == "Assembly::AssemblyObject"]
    model["n_assemblies"] = len(assemblies)
    if not assemblies:
        model["errors"].append("no Assembly in the document")
        return model
    asm = assemblies[0]
    if len(assemblies) > 1:  # the top-level one: not contained in another assembly
        tops = [a for a in assemblies if not any(p[0].TypeId == "Assembly::AssemblyObject" for p in (a.Parents or []))]
        asm = tops[0] if tops else asm
        model["assemblies"] = [a.Name for a in assemblies]
    root, root_sub = asm, ""
    parents = asm.Parents or []
    if parents:
        root, root_sub = parents[0][0], parents[0][1]
    model["assembly"] = asm.Name

    import UtilsAssembly  # FreeCAD's Assembly helpers (headless-safe)

    shape_id = 0
    comps = components_of(asm)
    for prefix, comp in comps:
        entry = {"name": comp.Name, "label": comp.Label, "type": comp.TypeId, "leaves": []}
        if prefix:
            entry["folder"] = prefix[:-1]
        try:
            found = leaves(comp, "")
        except Exception as exc:  # noqa: BLE001
            entry["error"] = "the component's leaves cannot be listed: %s" % exc
            found = []
        for sub, leaf in found:
            label, target = leaf_label(leaf)
            path = root_sub + prefix + comp.Name + "." + sub
            rec = {"label": label, "type": target.TypeId, "path": path,
                   "visible": bool(getattr(leaf, "Visibility", True))}
            try:
                shp = Part.getShape(root, path, needSubElement=False, transform=True).copy()
                rec["valid"] = bool(shp.isValid())
                rec["n_solids"] = len(shp.Solids)
                rec["volume"] = float(shp.Volume)
                rec["placement"] = matrix(shp.Placement)
                local = shp.copy()
                local.Placement = App.Placement()
                vol, ext, cen = describe(local)
                rec["extents_local"], rec["centroid_local"] = ext, cen
                hits, cands = match_bought(vol, ext, cen, classes)
                rec["bought_class"] = hits[0] if len(hits) == 1 else None
                rec["bought_candidates"] = cands
                if len(hits) > 1:
                    rec["error"] = "ambiguous bought class: the leaf matches %s" % ", ".join(hits)
                shape_id += 1
                rec["step"] = "shapes/%03d.step" % shape_id
                shp.exportStep(os.path.join(out_dir, rec["step"]))
                rec["brep"] = "shapes/%03d.brep" % shape_id
                shp.exportBrep(os.path.join(out_dir, rec["brep"]))
            except Exception as exc:  # noqa: BLE001
                rec["error"] = "%s%s" % (rec["error"] + "; " if rec.get("error") else "", exc)
            entry["leaves"].append(rec)
        model["components"].append(entry)

    components = set(c.Name for _p, c in comps)
    for obj in doc.Objects:
        if not is_joint(obj):
            continue
        if hasattr(obj, "ObjectToGround"):
            tgt = obj.ObjectToGround
            model["grounded"].append(tgt.Name if tgt is not None else None)
            continue
        rec = {"name": obj.Name, "label": obj.Label, "type": str(obj.JointType),
               "suppressed": bool(getattr(obj, "Suppressed", False)), "assembly": owning_assembly(obj)}
        try:
            for k in ("1", "2"):
                ref = getattr(obj, "Reference" + k)
                part = moving_part(UtilsAssembly, asm, components, ref)
                rec["part" + k] = part.Name if part is not None else None
                rec["jcs" + k] = matrix(UtilsAssembly.getJcsGlobalPlc(getattr(obj, "Placement" + k), ref))
            for k in ("Min", "Max"):
                rec["enable_angle_" + k.lower()] = bool(getattr(obj, "EnableAngle" + k, False))
                q = getattr(obj, "Angle" + k, 0.0)
                rec["angle_" + k.lower() + "_deg"] = float(getattr(q, "Value", q))
        except Exception as exc:  # noqa: BLE001
            rec["error"] = "%s" % exc
        model["joints"].append(rec)
    model["n_leaves"] = sum(len(c["leaves"]) for c in model["components"])
    model["n_shapes"] = shape_id
    model["elapsed_s"] = time.time() - t0
    return model


def main(argv):
    fcstd, bought_dir, out_dir = argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    write_model({"format": FORMAT, "errors": ["the extraction did not finish (time limit?)"]}, out_dir)
    try:
        model = extract(fcstd, bought_dir, out_dir)
    except Exception:  # noqa: BLE001
        model = {"format": FORMAT, "errors": ["extraction failed: " + traceback.format_exc()[-1500:]]}
    write_model(model, out_dir)
    print("BIMANUAL-EXTRACT done", len(model.get("components", [])), "components", model.get("n_leaves", 0), "leaves",
          len(model.get("joints", [])), "joints", len(model.get("errors", [])), "errors")


if __name__ == "__main__":
    main(sys.argv)
