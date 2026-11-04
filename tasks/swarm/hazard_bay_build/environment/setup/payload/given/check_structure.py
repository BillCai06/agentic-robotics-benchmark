"""Static check of a hazard-bay scene against the interface (`interface.md`): files, stage settings, required prims,
joints, the ropes, declarations, OmniGraph rules. Reads the USD as data (pxr only, no Kit, nothing executed) and scores
nothing. (The controller, `run.sh` + its Python, is the submission's other half; this check only notes whether it is
there.)

    python3 scan.py /home/user/submission/scene.usda [--json]

The grader runs this same file before it measures anything; a scene with errors scores zero.
"""

import json
import math
import os
import sys

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

ASSET_ROOTS = ("/opt/assets/isaac", "/opt/assets/arb")
CORE_MDL = {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl", "OmniSurfaceLite.mdl", "OmniEmissive.mdl",
            "OmniHair.mdl", "OmniUe4Base.mdl", "OmniUe4Function.mdl", "OmniUe4Translucent.mdl", "OmniUe4FunctionExtension17.mdl",
            "OmniImage.mdl", "OmniPBRBase.mdl", "OmniPBR_ClearCoat.mdl", "OmniPBR_Opacity.mdl", "OmniSurfaceBase.mdl"}
# OmniGraph: the node types a scene may use (events, reads, arithmetic, logic, variables, drive / arb:* writes)
ALLOWED_NODES = {
    "isaacsim.core.nodes.OnPhysicsStep", "isaacsim.core.nodes.IsaacArticulationState", "isaacsim.core.nodes.IsaacReadSimulationTime",
    "isaacsim.sensors.physics.IsaacReadJointState",
    "omni.graph.action.OnPlaybackTick", "omni.graph.action.OnTick", "omni.graph.action.Branch", "omni.graph.action.FlipFlop",
    "omni.graph.action.Once", "omni.graph.action.Sequence", "omni.graph.action.Switch", "omni.graph.action.Counter",
    "omni.graph.action.Delay", "omni.graph.action.OnVariableChange", "omni.graph.action.SyncGate", "omni.graph.action.Gate",
    "omni.graph.action.ForLoop", "omni.graph.action.ForEach", "omni.graph.action.MultiGate", "omni.graph.action.RationalTimeSyncGate",
    "omni.graph.core.ReadVariable", "omni.graph.core.WriteVariable",
    "omni.graph.nodes.Add", "omni.graph.nodes.Subtract", "omni.graph.nodes.Multiply", "omni.graph.nodes.Divide", "omni.graph.nodes.Negate",
    "omni.graph.nodes.Clamp", "omni.graph.nodes.Magnitude", "omni.graph.nodes.Normalize", "omni.graph.nodes.DotProduct",
    "omni.graph.nodes.CrossProduct", "omni.graph.nodes.Distance3D", "omni.graph.nodes.Compare", "omni.graph.nodes.SelectIf",
    "omni.graph.nodes.BooleanAnd", "omni.graph.nodes.BooleanOr", "omni.graph.nodes.BooleanNot", "omni.graph.nodes.BooleanNand",
    "omni.graph.nodes.BooleanNor", "omni.graph.nodes.BooleanXor", "omni.graph.nodes.ArrayIndex", "omni.graph.nodes.ArrayLength",
    "omni.graph.nodes.ReadPrimAttribute", "omni.graph.nodes.ReadPrimAttributes", "omni.graph.nodes.ReadPrimLocalTransform",
    "omni.graph.nodes.GetPrimLocalToWorldTransform", "omni.graph.nodes.GetMatrix4Translation", "omni.graph.nodes.GetMatrix4Quaternion",
    "omni.graph.nodes.GetMatrix4Rotation", "omni.graph.nodes.RotateVector", "omni.graph.nodes.TransformVector",
    "omni.graph.nodes.BreakVector2", "omni.graph.nodes.BreakVector3", "omni.graph.nodes.BreakVector4", "omni.graph.nodes.MakeVector2",
    "omni.graph.nodes.MakeVector3", "omni.graph.nodes.MakeVector4", "omni.graph.nodes.ToDouble", "omni.graph.nodes.ToFloat",
    "omni.graph.nodes.ToInt", "omni.graph.nodes.ToToken", "omni.graph.nodes.ToString", "omni.graph.nodes.Modulo", "omni.graph.nodes.Sin",
    "omni.graph.nodes.Cos", "omni.graph.nodes.Tan", "omni.graph.nodes.Acos", "omni.graph.nodes.Asin", "omni.graph.nodes.Atan",
    "omni.graph.nodes.Ease", "omni.graph.nodes.Increment", "omni.graph.nodes.Decrement", "omni.graph.nodes.Floor", "omni.graph.nodes.Ceil",
    "omni.graph.nodes.Round", "omni.graph.nodes.Exponent", "omni.graph.nodes.Length", "omni.graph.nodes.Min", "omni.graph.nodes.Max",
    "omni.graph.nodes.Abs", "omni.graph.nodes.Sqrt", "omni.graph.nodes.Noop", "omni.graph.nodes.GetGraphTargetPrim",
    "omni.graph.nodes.WritePrimAttribute", "omni.graph.nodes.ExtractAttribute", "omni.graph.nodes.ExtractPrim",
    "omni.graph.nodes.BundleConstructor", "omni.graph.nodes.AppendArray", "omni.graph.nodes.ArrayFill", "omni.graph.nodes.ArrayRemoveIndex",
    "omni.graph.nodes.ArrayInsertValue", "omni.graph.nodes.ArraySetIndex", "omni.graph.nodes.ArrayResize",
}
REQUIRED = ("/World/Facility/Floor", "/World/Facility/Tables/Table_A", "/World/Facility/Tables/Table_B", "/World/Facility/Crate",
            "/World/Facility/Items", "/World/Facility/SteamLine", "/World/Facility/SteamLine/Handle", "/World/Facility/SteamLine/Plume",
            "/World/Facility/SteamLine/Gauge/Dial", "/World/Facility/FirePan", "/World/Facility/Tank", "/World/Facility/Bucket/Body",
            "/World/Facility/Pads/Pad_1", "/World/Facility/Pads/Pad_2", "/World/Facility/Pads/Pad_3", "/World/Facility/Pads/UGV_Bay",
            "/World/Facility/Pads/Valve_Square", "/World/Fleet/UAV_1", "/World/Fleet/UAV_2", "/World/Fleet/UAV_3", "/World/Fleet/UGV",
            "/World/Fleet/UAV_1/Rope", "/World/Fleet/UAV_2/Rope", "/World/Fleet/UAV_3/Rope")
BODIES = ("/World/Facility/SteamLine/Handle", "/World/Facility/Bucket/Body")
STATE_ATTRS = (("/World/Facility/SteamLine", "arb:steam_active"), ("/World/Facility/FirePan", "arb:fire_active"),
               ("/World/Fleet/UAV_1", "arb:disabled"), ("/World/Fleet/UAV_2", "arb:disabled"), ("/World/Fleet/UAV_3", "arb:disabled"),
               ("/World/Fleet/UGV", "arb:disabled"))
ROPE_LINKS_MIN = 8
ROBOTS = ("UAV_1", "UAV_2", "UAV_3", "UGV")


def _inside(path, roots):
    p = os.path.realpath(path)
    return any(p == r or p.startswith(r + os.sep) for r in roots)


def _referenced_assets(prim):
    """Asset paths of the references / payloads that compose `prim` (its own arcs, not its descendants')."""
    out = []
    for spec in prim.GetPrimStack():
        for lst in (spec.referenceList, spec.payloadList):
            for item in list(lst.prependedItems) + list(lst.appendedItems) + list(lst.explicitItems) + list(lst.addedItems):
                if item.assetPath:
                    out.append(item.assetPath)
    return out


def _finite(value):
    try:
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if hasattr(value, "__len__") and not isinstance(value, str):
            return all(_finite(v) for v in value)
        if hasattr(value, "GetReal"):
            return math.isfinite(value.GetReal()) and _finite(value.GetImaginary())
    except Exception:  # noqa: BLE001
        return False
    return True


def _joint_ends(j):
    jj = UsdPhysics.Joint(j)
    b0 = [t.pathString for t in jj.GetBody0Rel().GetTargets()]
    b1 = [t.pathString for t in jj.GetBody1Rel().GetTargets()]
    return b0, b1


def _asset_arcs(prim, asset_roots):
    """The references and payloads on `prim` into the asset library, authored in a layer outside it (the robot's
    own arcs to a library model, not the library's arcs among its own files): (absolute asset path, target prim
    path or None for the asset's default prim)."""
    out = []
    for spec in prim.GetPrimStack():
        layer = spec.layer
        if layer.realPath and _inside(layer.realPath, asset_roots):
            continue
        for lst in (spec.referenceList, spec.payloadList):
            for item in list(lst.prependedItems) + list(lst.appendedItems) + list(lst.explicitItems) + list(lst.addedItems):
                if not item.assetPath:
                    continue
                path = layer.ComputeAbsolutePath(item.assetPath) if not os.path.isabs(item.assetPath) else item.assetPath
                if _inside(path, asset_roots):
                    out.append((path, item.primPath.pathString if item.primPath and not item.primPath.isEmpty else None))
    return out


_ASSET_COLLIDERS = {}


def _asset_colliders(path, prim_path):
    """The colliders a library model defines below the prim a reference takes (paths relative to it)."""
    key = (path, prim_path)
    if key not in _ASSET_COLLIDERS:
        rel = []
        try:
            st = Usd.Stage.Open(path, Usd.Stage.LoadAll)
            top = st.GetPrimAtPath(prim_path) if prim_path else st.GetDefaultPrim()
            if top and top.IsValid():
                for p in Usd.PrimRange(top, Usd.TraverseInstanceProxies()):
                    if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute("physics:collisionEnabled").Get() is not False:
                        rel.append(p.GetPath().MakeRelativePath(top.GetPath()).pathString)
        except Exception:  # noqa: BLE001 - an asset that does not open has nothing to compare against
            rel = []
        _ASSET_COLLIDERS[key] = rel
    return _ASSET_COLLIDERS[key]


def colliders_kept(stage, root, asset_roots):
    """Does the robot keep every collider of the library models it is built from? A collider of a referenced model
    that is gone from the composed stage (its prim removed or deactivated, its CollisionAPI deleted) or switched off
    (`physics:collisionEnabled` false) counts against it; colliders added beside them are the robot's own business."""
    checked, missing, disabled = 0, [], []
    rp = stage.GetPrimAtPath(root)
    for p in Usd.PrimRange(rp, Usd.TraverseInstanceProxies()):
        for path, prim_path in _asset_arcs(p, asset_roots):
            for rel in _asset_colliders(path, prim_path):
                checked += 1
                tgt = p.GetPath() if rel in ("", ".") else p.GetPath().AppendPath(Sdf.Path(rel))
                q = stage.GetPrimAtPath(tgt)
                if not q or not q.IsValid() or not q.IsActive() or not q.HasAPI(UsdPhysics.CollisionAPI):
                    missing.append(tgt.pathString)
                elif q.GetAttribute("physics:collisionEnabled").Get() is False:
                    disabled.append(tgt.pathString)
    return dict(ok=not missing and not disabled, checked=checked, missing=missing[:8], n_missing=len(missing),
                disabled=disabled[:8], n_disabled=len(disabled))


def scan(scene):
    errors, notes, facts = [], [], {}
    sub = os.path.realpath(os.path.dirname(scene))
    roots = (sub,) + tuple(os.path.realpath(r) for r in ASSET_ROOTS)
    asset_roots = tuple(os.path.realpath(r) for r in ASSET_ROOTS)
    if not os.path.isfile(scene):
        return dict(ok=False, errors=[f"no scene file at {scene}"], notes=notes, facts=facts)
    if not os.path.isfile(os.path.join(sub, "run.sh")):
        notes.append("no run.sh in the submission: the hoist mission will have no controller")
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(scene)
    except Exception as exc:  # noqa: BLE001
        return dict(ok=False, errors=[f"the scene cannot be read: {exc}"], notes=notes, facts=facts)
    for layer in layers:
        if layer.realPath and not _inside(layer.realPath, roots):
            errors.append(f"layer outside the submission and the asset library: {layer.realPath}")
    for a in assets:
        if not _inside(str(a), roots):
            errors.append(f"file outside the submission and the asset library: {a}")
    for u in unresolved:
        if os.path.basename(str(u)) not in CORE_MDL:
            errors.append(f"unresolved file: {u}")
    # every reference into the submission is relative (an absolute one breaks when the submission is copied)
    for layer in layers:
        if not layer.realPath or not _inside(layer.realPath, (sub,)):
            continue
        for ref in layer.GetExternalReferences():
            if ref and os.path.isabs(ref) and not _inside(ref, asset_roots) and os.path.basename(ref) not in CORE_MDL:
                errors.append(f"{os.path.basename(layer.realPath)} references {ref} by absolute path; only the asset library may be")
    stage = Usd.Stage.Open(scene)
    if stage is None:
        return dict(ok=False, errors=errors + ["the scene does not open"], notes=notes, facts=facts)
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        errors.append("upAxis is not Z")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
        errors.append("metersPerUnit is not 1")
    if abs(UsdPhysics.GetStageKilogramsPerUnit(stage) - 1.0) > 1e-9:
        errors.append("kilogramsPerUnit is not 1")
    if not stage.GetDefaultPrim() or stage.GetDefaultPrim().GetPath() != Sdf.Path("/World"):
        errors.append("the default prim is not /World")
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if [str(p.GetPath()) for p in scenes] != ["/World/PhysicsScene"]:
        errors.append(f"expected exactly one physics scene at /World/PhysicsScene, found {[str(p.GetPath()) for p in scenes]}")
    for path in REQUIRED:
        if not stage.GetPrimAtPath(path).IsValid():
            errors.append(f"missing {path}")
    if errors:
        return dict(ok=False, errors=errors, notes=notes, facts=facts)

    def has(path, api):
        return stage.GetPrimAtPath(path).HasAPI(api)

    for path in BODIES:
        if not has(path, UsdPhysics.RigidBodyAPI):
            errors.append(f"{path} is not a rigid body")
    for path, name in STATE_ATTRS:
        a = stage.GetPrimAtPath(path).GetAttribute(name)
        if not a or not a.IsValid() or a.GetTypeName() != Sdf.ValueTypeNames.Bool:
            errors.append(f"{path} has no bool attribute {name}")
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint) and p.IsActive()]
    handle, bucket = "/World/Facility/SteamLine/Handle", "/World/Facility/Bucket/Body"
    if not any(j.IsA(UsdPhysics.RevoluteJoint) and handle in sum(_joint_ends(j), []) for j in joints):
        errors.append(f"no revolute joint on {handle}")
    floor = stage.GetPrimAtPath("/World/Facility/Floor")
    if not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(floor, Usd.TraverseInstanceProxies())):
        errors.append("/World/Facility/Floor has no collider")
    items = [c for c in stage.GetPrimAtPath("/World/Facility/Items").GetChildren() if c.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not items:
        errors.append("no rigid body under /World/Facility/Items")
    facts["items"] = {c.GetName(): sorted({a for p in Usd.PrimRange(c) for a in _referenced_assets(p)}) for c in items}
    # the fleet: each robot one floating articulation, declared models, tagged sensors, nothing fixing it to the world
    facts["robots"] = {}
    ropes, all_links = {}, set()
    for name in ROBOTS:
        root = f"/World/Fleet/{name}"
        rp = stage.GetPrimAtPath(root)
        art = [p for p in Usd.PrimRange(rp) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(art) != 1:
            errors.append(f"{root} holds {len(art)} articulation roots, expected 1")
        for j in joints:
            b0, b1 = _joint_ends(j)
            ends = b0 + b1
            if (not b0 or not b1) and any(e.startswith(root + "/") or e == root for e in ends):
                errors.append(f"{j.GetPath()} fixes {root} to the world")
            elif (any(e.startswith(root + "/") for e in ends) and any(not e.startswith(root + "/") for e in ends)
                  and not any(e.startswith(root + "/Rope/") for e in ends)):
                notes.append(f"{j.GetPath()} joins {root} to {[e for e in ends if not e.startswith(root + '/')]}: the bay mission drops it")
        refs = sorted({a for p in Usd.PrimRange(rp) for a in _referenced_assets(p)})
        decl = {}
        for key in ("airframe_model", "arm_model", "base_model"):
            vals = sorted({str(p.GetAttribute(f"arb:{key}").Get()) for p in Usd.PrimRange(rp)
                           if p.GetAttribute(f"arb:{key}") and p.GetAttribute(f"arb:{key}").Get()})
            decl[key] = vals[0] if len(vals) == 1 else None
            if len(vals) > 1:
                errors.append(f"{root} declares several arb:{key} values: {vals}")
        if name.startswith("UAV") and not decl["airframe_model"]:
            errors.append(f"{root} declares no arb:airframe_model")
        if name == "UGV" and not decl["base_model"]:
            errors.append(f"{root} declares no arb:base_model")
        if not decl["arm_model"]:
            errors.append(f"{root} declares no arb:arm_model")
        cams = [p.GetPath().pathString for p in Usd.PrimRange(rp) if p.IsA(UsdGeom.Camera) and p.IsActive()]
        tagged = [c for c in cams if stage.GetPrimAtPath(c).GetAttribute("arb:sensor_model")
                  and stage.GetPrimAtPath(c).GetAttribute("arb:sensor_model").Get()]
        kept = colliders_kept(stage, root, asset_roots)
        facts["robots"][name] = dict(articulation_root=str(art[0].GetPath()) if art else None, assets=refs, cameras=cams,
                                     tagged_cameras=tagged, declared=decl, colliders=kept)
        if not tagged:
            notes.append(f"{root} carries no camera with arb:sensor_model")
        if not kept["ok"]:
            notes.append(f"{root}: {kept['n_missing']} collider(s) of its library models removed and {kept['n_disabled']} switched off "
                         f"(e.g. {(kept['missing'] + kept['disabled'])[:2]}): in the bay mission it scores 0 on no collisions")
    # the ropes: under each flying robot a chain of rigid-body links joined by joints, hooked to a body of the robot by a
    # joint outside the articulation, joined to the bucket at its ring; the bucket joined to nothing else; no distance
    # joints in the rigging
    for name in ("UAV_1", "UAV_2", "UAV_3"):
        root = f"/World/Fleet/{name}"
        rope = stage.GetPrimAtPath(f"{root}/Rope")
        links = [p.GetPath().pathString for p in Usd.PrimRange(rope) if p.HasAPI(UsdPhysics.RigidBodyAPI)] if rope.IsValid() else []
        linkset = set(links)
        all_links |= linkset
        hook, ring, chain, bad = [], [], 0, []
        for j in joints:
            b0, b1 = _joint_ends(j)
            ends = b0 + b1
            if not any(e in linkset for e in ends):
                continue
            if j.IsA(UsdPhysics.DistanceJoint):
                bad.append(f"{j.GetPath()} is a distance joint")
                continue
            others = [e for e in ends if e not in linkset]
            if not others:
                chain += 1
            elif bucket in others:
                ring.append(j)
            elif any(e.startswith(root + "/") for e in others):
                hook.append(j)
            else:
                bad.append(f"{j.GetPath()} joins the rope to {others}")
        facts.setdefault("ropes", {})[name] = dict(links=len(links), chain_joints=chain, hook_joints=len(hook), ring_joints=len(ring))
        if len(links) < ROPE_LINKS_MIN:
            errors.append(f"{root}/Rope holds {len(links)} rigid-body links; at least {ROPE_LINKS_MIN} are required")
        if len(hook) != 1:
            errors.append(f"{root}/Rope: expected exactly one joint from a body of the robot to a link (the hook), found {len(hook)}")
        elif not UsdPhysics.Joint(hook[0]).GetExcludeFromArticulationAttr().Get():
            errors.append(f"{hook[0].GetPath()}: the hook joint must set physics:excludeFromArticulation (the rope is not part of the articulation)")
        if len(ring) != 1:
            errors.append(f"{root}/Rope: expected exactly one joint from a link to {bucket} (the ring), found {len(ring)}")
        if links and chain < len(links) - 1:
            errors.append(f"{root}/Rope: {len(links)} links but only {chain} joints between them: not one chain")
        errors += bad
    for j in joints:
        b0, b1 = _joint_ends(j)
        ends = b0 + b1
        if bucket in ends and not any(e in all_links for e in ends):
            errors.append(f"{j.GetPath()} joins the bucket to something other than a rope")
    # no behaviour scripts anywhere, no non-finite numbers where they would break the measurement
    n_nodes = 0
    for p in stage.Traverse():
        schemas = p.GetPrimTypeInfo().GetAppliedAPISchemas()
        if any("Scripting" in s for s in schemas) or p.GetAttribute("omni:scripting:scripts"):
            errors.append(f"{p.GetPath()}: behaviour scripts are not allowed")
        for a in p.GetAttributes():
            n = a.GetName()
            if n.startswith(("physics:mass", "xformOp:", "physics:localPos", "physics:maxDistance")) or n in (
                    "focalLength", "horizontalAperture", "verticalAperture", "physics:centerOfMass"):
                v = a.Get() if a.HasAuthoredValue() else None  # (unauthored schema fallbacks can be -inf by design)
                if v is not None and not _finite(v):
                    errors.append(f"{p.GetPath()}.{n} is not a finite number")
        if p.GetTypeName() != "OmniGraphNode":
            continue
        n_nodes += 1
        t = p.GetAttribute("node:type").Get() if p.GetAttribute("node:type") else None
        where = str(p.GetPath())
        if not t or "script" in t.lower() or not (t in ALLOWED_NODES or t.startswith("omni.graph.nodes.Constant")):
            errors.append(f"{where}: node type {t} is not allowed")
            continue
        if t == "omni.graph.nodes.WritePrimAttribute":
            na = p.GetAttribute("inputs:name")
            if na.HasAuthoredConnections() if na else True:
                errors.append(f"{where}: the written attribute's name must be a constant")
                continue
            name = na.Get() or ""
            if not (name.startswith("drive:") or name.startswith("arb:")):
                errors.append(f"{where}: writes {name!r}; only drive:* and arb:* attributes may be written")
    facts["graph_nodes"] = n_nodes
    return dict(ok=not errors, errors=errors, notes=notes, facts=facts)


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/home/user/submission/scene.usda"
    if not os.path.isfile(scene) and scene.endswith(".usda") and os.path.isfile(scene[:-1]):
        scene = scene[:-1]
    r = scan(scene)
    if "--json" in sys.argv:
        print(json.dumps(r))
        return
    print("OK" if r["ok"] else "FAILED")
    for e in r["errors"]:
        print("  error:", e)
    for n in r["notes"]:
        print("  note:", n)


if __name__ == "__main__":
    main()
