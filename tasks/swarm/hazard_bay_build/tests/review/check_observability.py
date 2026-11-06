"""Maintainers' check that a candidate day's targets can be observed (round 4, spec section 4.1: the pre-registered
observer-rig check). It runs on every candidate of a batch before the reference does; a day that fails is dropped and
recorded (`make_fixed_set.py select` reads `<results>/<id>.obs.json`). Never part of the grade, never named to the
agent.

    # the grader's renderer (in the task image, one GPU; the real bay dressed as for the mission, no robots):
    /opt/venv/bin/python3 -I -s tests/review/check_observability.py --set FILE --out DIR [--ids a,b] [--bay-rows]
    # the pxr / numpy cross-check (the ray caster of the observability study, `obs_raycast/`):
    python3 -I tests/review/check_observability.py --raycast --set FILE --out DIR [--ids a,b] [--bay-rows]

The rows (THRESHOLDS, pre-registered; frozen with the reference: `make_fixed_set.py` hashes this file):

 1  each part on Table A: a CAM-RGB-M at 0.92 m, aimed, from any floor point within 2.5 m with a 0.35 m circle clear of
    the furniture and outside the burning pan's 2 m ring: best view >= 0.9 of its camera-facing surface and >= 10,000 px^2
 2  each part on Table B: the same without the ring (the value with the ring recorded)
 3  each part from above: a CAM-RGB-W looking down from 2.6 m over its table's centre: >= 0.9 of its top
 4  the lever: a CAM-RGB-M at 0.92 m over the white square, aimed: the grip >= 0.5 and >= 1,000 px^2 at the day's angle,
    85 and 90 degrees
 5  the dial's reading: the same camera aimed at the dial: the face >= 0.9, >= 100 px across, the needle >= 3 px wide
 6  the dial from the start: the reference's mast camera at the day's start pose, or after a turn on the spot of up to
    90 degrees: >= 0.5 of the face, >= 40 px across by R8's measure
 7  the pan: the reference's THERM-C at the approach hover: >= 0.9 of its top, >= 25 px across
 8  the vehicle's start pose: the reference's mast camera at the day's start pose: >= 3 of {dial, valve body, lever grip,
    supports 1-3}, each >= 0.5 and >= 1,000 px^2
 9-11 (once per bay, --bay-rows): the bucket from the reference formation's down cameras >= 0.9; a flying robot's own place
    (its forward CAM-RGB-M on its pad and at the approach hover: >= 3 fixed landmarks, each >= 0.5 and >= 1,000 px^2);
    the crate's rim from a CAM-RGB-M at 0.92 m within 2.5 m >= 0.9

The renderer's measure: the observer is a grader camera (the mission's: the catalog's intrinsics, the pinned render
settings) with instance segmentation; a target's visible share is its pixels in the render over its pixels with every
other prim hidden (a second render). Its viewpoints for rows 1-2 are the ray caster's best three per part (its sweep of
floor points; the renderer then measures them), the fixed ones for the other rows. The robots' bodies are not drawn.
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(HERE, "obs_raycast"))

THRESHOLDS = dict(part_share=0.9, part_px2=10000.0, top_share=0.9, grip_share=0.5, grip_px2=1000.0, grip_angles=(85.0, 90.0),
                  dial_share=0.9, dial_px=100.0, needle_px=3.0, start_dial_share=0.5, start_dial_px=40.0, start_turns_deg=90.0,
                  pan_share=0.9, pan_px=25.0, landmarks=3, landmark_share=0.5, landmark_px2=1000.0, bucket_share=0.9,
                  rim_share=0.9, ring_m=2.0, reach_m=2.5, mast_z=0.92, square=(-1.24, 2.61), approach=(2.60, -0.40, 2.50))
LANDMARKS = ("dial", "valve_block", "lever_grip", "support_1", "support_2", "support_3")


def file_hash():
    """This check's code and thresholds (the freeze: make_fixed_set.py records it)."""
    h = hashlib.sha256(open(os.path.abspath(__file__), "rb").read())
    for f in ("extract_geom.py", "visibility.py", "views.py"):
        h.update(open(os.path.join(HERE, "obs_raycast", f), "rb").read())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- the ray caster's rows
def raycast_rows(day, bay, V, W):
    """Rows 1-8 by the ray caster (obs_raycast): per row the measure and pass."""
    th = THRESHOLDS
    occ, parts = V.day_scene(bay, day)
    rows = {}
    for n, b in parts.items():
        table = day["parts"][n]["table"]
        ring = V.best_ugv_view(b, occ, day, heights=(th["mast_z"],), fire=True, rmax=th["reach_m"])
        free = V.best_ugv_view(b, occ, day, heights=(th["mast_z"],), fire=False, rmax=th["reach_m"])
        best = ring if table == "Table_A" else free
        ok = bool(best and best["frac"] >= th["part_share"] and best["px2"] >= th["part_px2"])
        rows[f"{1 if table == 'Table_A' else 2}:{n}"] = dict(share=_r(best and best["frac"]), px2=_r(best and best["px2"]), at=best and [best["x"], best["y"]],
                                                             with_ring=dict(share=_r(ring and ring["frac"]), px2=_r(ring and ring["px2"])), ok=ok)
        t = V.TABLES[table]
        C = np.array([t["c"][0], t["c"][1], 2.6])
        top = V.view(C, np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), "CAM-RGB-W", b, occ)
        rows[f"3:{n}"] = dict(share=_r(top["frac"]), px2=_r(top["px2"]), ok=bool(top["frac"] >= th["top_share"]))
    sq = np.array([th["square"][0], th["square"][1], th["mast_z"]])
    for deg in (float(day["lever_deg"]),) + tuple(th["grip_angles"]):
        o2, _ = V.day_scene(bay, dict(day, lever_deg=deg))
        grip = W.find(o2, "Handle/grip")[0]
        d, u, r = V.look_cam(sq, grip.c)
        v = V.view(sq, d, u, r, "CAM-RGB-M", grip, o2)
        rows[f"4:grip@{deg:.1f}"] = dict(share=_r(v["frac"]), px2=_r(v["px2"]), ok=bool(v["frac"] >= th["grip_share"] and v["px2"] >= th["grip_px2"]))
    dial = W.find(occ, "Gauge/Dial")[0]
    d, u, r = V.look_cam(sq, dial.c)
    v = V.view(sq, d, u, r, "CAM-RGB-M", dial, occ)
    dp = W.dial_needle_px(sq)
    rows["5:dial"] = dict(share=_r(v["frac"]), dial_px=dp["dial_px"], needle_px=dp["needle_w_px"],
                          ok=bool(v["frac"] >= th["dial_share"] and dp["dial_px"] >= th["dial_px"] and dp["needle_w_px"] >= th["needle_px"]))
    dx, dy, dpsi = day["ugv"]
    root = W.UGV_ROOT + np.array([dx, dy, 0.0])
    best6 = None
    for turn in np.arange(-th["start_turns_deg"], th["start_turns_deg"] + 0.1, 15.0):
        C, d, u, r = W.cam_world(root, W.ugv_frame(root[:2], dpsi + turn), W.UGV_MAST)
        v = V.view(C, d, u, r, "CAM-RGB-M", dial, occ)
        dist = float(np.linalg.norm(dial.c - C))
        px = 1920 * math.degrees(2 * math.atan(0.05 / dist)) / 70.3 if v["frac"] > 0 else 0.0  # (R8's measure, cs:870)
        key = (v["frac"] >= th["start_dial_share"] and px >= th["start_dial_px"], v["frac"])
        if best6 is None or key > best6[0]:
            best6 = (key, dict(turn_deg=float(turn), share=_r(v["frac"]), px=round(px, 1)))
    rows["6:dial_from_start"] = dict(best6[1], ok=bool(best6[0][0]))
    pv = W.pan_views(day, occ)["approach_hover/Thermal"]
    rows["7:pan"] = dict(share=pv["frac"], px=pv["w"], ok=bool(pv["frac"] >= th["pan_share"] and (pv["w"] or 0) >= th["pan_px"]))
    sv = W.start_views(day, occ)["views"]
    good = [k for k in LANDMARKS if k in sv and sv[k]["frac"] >= th["landmark_share"] and sv[k]["px2"] >= th["landmark_px2"]]
    rows["8:start_landmarks"] = dict(seen=good, n=len(good), ok=len(good) >= th["landmarks"])
    return rows


def raycast_bay_rows(bay, V, W):
    th = THRESHOLDS
    rows = {}
    bv = W.bucket_views(bay)
    for n in ("UAV_1", "UAV_2", "UAV_3"):
        rows[f"9:bucket:{n}"] = dict(share=bv[n]["frac"], ok=bool(bv[n]["frac"] >= th["bucket_share"]))
    day0 = dict(lever_deg=0.0, pan=[0.0, 0.0], parts={})
    occ, _ = V.day_scene(bay, day0)
    marks = {k: s for k, s in W.landmarks(occ).items()}
    for k in ("Pad_1", "Pad_2", "Pad_3", "CalibMat"):
        f = [s for s in occ if f"/{k}/" in s.name]
        if f:
            marks[k] = f[0]
    for n, (px, py) in {"UAV_1": (0.0, -1.5), "UAV_2": (-0.95, -3.15), "UAV_3": (0.95, -3.15), "approach": th["approach"][:2]}.items():
        H = np.array([px, py, 0.479 if n != "approach" else th["approach"][2]])
        C, d, u, r = W.cam_world(H, np.eye(3), W.UAV_CAMS["Camera"])
        seen = []
        for k, s in marks.items():
            v = V.view(C, d, u, r, "CAM-RGB-M", s, occ)
            if v["frac"] >= th["landmark_share"] and v["px2"] >= th["landmark_px2"]:
                seen.append(k)
        rows[f"10:own_place:{n}"] = dict(seen=seen, n=len(seen), ok=len(seen) >= th["landmarks"])
    walls = [s for s in occ if "/Crate/" in s.name]
    best = 0.0
    for s in walls:
        b = V.best_ugv_view(s, occ, day0, heights=(th["mast_z"],), fire=False, rmax=th["reach_m"])
        best = max(best, b["frac"] if b else 0.0)
    rows["11:crate_rim"] = dict(share=_r(best), ok=bool(best >= th["rim_share"]))
    return rows


def _r(x):
    return None if x is None else round(float(x), 4)


# --------------------------------------------------------------------------------------------- the renderer
class Render:
    """The real bay of a day, dressed as for the mission (no robots), in Kit; observer cameras with instance
    segmentation through the mission's own render products."""

    def __init__(self, cfg, work):
        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": True})
        self.cfg, self.work = cfg, work

    def open_day(self, day, bay_layer):
        import control_sim
        import mission_stage
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom
        path = os.path.join(self.work, "obs_day.usda")
        if os.path.exists(path):
            os.remove(path)
        layer = Sdf.Layer.CreateNew(path)
        layer.subLayerPaths.append(bay_layer)
        layer.defaultPrim = "World"
        layer.Save()
        mission_stage.dress(path, {}, self.cfg["sensors"]["look"])
        st = Usd.Stage.Open(path)
        for n, p in day["parts"].items():  # the parts at the day's poses
            prim = st.GetPrimAtPath(f"/World/Facility/Items/{n}")
            if not prim.IsValid():
                continue
            m = Gf.Matrix4d(1.0)
            m.SetRotateOnly(Gf.Quatd(float(p["quat"][0]), Gf.Vec3d(*[float(v) for v in p["quat"][1:]])))
            m.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in p["pos"]]))
            xf = UsdGeom.Xformable(prim)
            parent = UsdGeom.XformCache().GetParentToWorldTransform(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "arb_day").Set(m * parent.GetInverse())
        stem = Gf.Vec3d(-1.2, 3.2, 0.95)  # the lever turned toward +x (closing) about the stem (+y), the pan moved
        rot = Gf.Matrix4d(1.0).SetTranslate(-stem) * Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), float(day["lever_deg"]))) * \
            Gf.Matrix4d(1.0).SetTranslate(stem)
        control_sim.move_in_world(st, "/World/Facility/SteamLine/Handle", rot, "arb_day")
        control_sim.move_in_world(st, "/World/Facility/FirePan", Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(day["pan"][0], day["pan"][1], 0.0)), "arb_day")
        st.GetRootLayer().Save()
        ctx = omni.usd.get_context()
        ctx.open_stage(path)
        for _ in range(3):
            self.app.update()
        self.stage = ctx.get_stage()
        self.products = {}

    def product(self, model):
        import sensor_rig
        if model not in self.products:
            sm = sensor_rig.stream_of(model, self.cfg["catalog"]["sensors"][model])
            spec = dict(name=f"Obs_{model.replace('-', '_')}", kind="thermal", width=sm["width"], height=sm["height"], focal=sm["focal"], hap=sm["hap"],
                        clip=self.cfg["sensors"]["clip_m"])
            self.products[model] = (sensor_rig.Products(self.stage, [spec], self.cfg["sensors"]["render"]), sm)
        return self.products[model]

    def ids(self, model, eye, target):
        import sensor_rig
        P, sm = self.product(model)
        P.set_pose(0, eye, sensor_rig.look_at(eye, target))
        for _ in range(3):  # (a read shows the render before)
            self.app.update()
        arr, info = P.read()[0]
        return np.asarray(arr).reshape(sm["height"], sm["width"]), (info or {}).get("idToLabels") or {}

    def measure(self, model, eye, target_xyz, target_prim):
        """(visible share, visible px, px across) of a prim's subtree from `eye` aimed at `target_xyz`."""
        from pxr import UsdGeom
        ids, lab = self.ids(model, eye, target_xyz)
        hit = [int(k) for k, v in lab.items() if isinstance(v, str) and (v == target_prim or v.startswith(target_prim + "/"))]
        a = int(np.isin(ids, hit).sum()) if hit else 0
        hidden = []  # every other prim hidden: the siblings of the target and of each of its ancestors
        p = self.stage.GetPrimAtPath(target_prim)
        while p and p.GetPath().pathString != "/":
            for sib in p.GetParent().GetChildren() if p.GetParent() else []:
                if sib.GetPath() != p.GetPath() and sib.IsA(UsdGeom.Imageable) and not sib.GetPath().pathString.startswith("/World/Sensors") \
                        and not sib.GetPath().pathString.startswith("/Render"):
                    UsdGeom.Imageable(sib).MakeInvisible()
                    hidden.append(sib)
            p = p.GetParent()
        ids2, lab2 = self.ids(model, eye, target_xyz)
        hit2 = [int(k) for k, v in lab2.items() if isinstance(v, str) and (v == target_prim or v.startswith(target_prim + "/"))]
        mask = np.isin(ids2, hit2) if hit2 else np.zeros_like(ids2, dtype=bool)
        b = int(mask.sum())
        for sib in hidden:
            UsdGeom.Imageable(sib).MakeVisible()
        cols = np.where(mask.any(axis=0))[0]
        across = int(cols.max() - cols.min() + 1) if cols.size else 0
        return (a / b if b else 0.0), a, across


def render_rows(day, R, ray_rows, V, W, bay):
    """Rows 1-8 with the renderer: the viewpoints of the ray caster's sweep (rows 1-2: its best view per part and two
    runners-up), the fixed ones for the others."""
    th = THRESHOLDS
    occ, parts = V.day_scene(bay, day)
    rows = {}
    for n, b in parts.items():
        table = day["parts"][n]["table"]
        fire = table == "Table_A"
        rects, discs = V.footprints(day, fire)
        cands = []
        for x in np.arange(-3.5, 3.5001, 0.1):
            for y in np.arange(-3.6, 3.2501, 0.1):
                if math.hypot(x - b.c[0], y - b.c[1]) > th["reach_m"] or not V.free((x, y), rects, discs):
                    continue
                C = np.array([x, y, th["mast_z"]])
                d, u, r = V.look_cam(C, b.c)
                v = V.view(C, d, u, r, "CAM-RGB-M", b, occ)
                cands.append((v["frac"] >= 0.5, v["px2"], C))
        cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
        best = None
        for _, _, C in cands[:3]:
            share, px, _ = R.measure("CAM-RGB-M", C, b.c, f"/World/Facility/Items/{n}")
            if best is None or (share >= th["part_share"], px) > (best[0] >= th["part_share"], best[1]):
                best = (share, px, C)
        rows[f"{1 if fire else 2}:{n}"] = dict(share=_r(best and best[0]), px2=best and best[1], at=best and [round(float(v), 2) for v in best[2][:2]],
                                                ok=bool(best and best[0] >= th["part_share"] and best[1] >= th["part_px2"]),
                                                raycast=ray_rows.get(f"{1 if fire else 2}:{n}"))
        t = V.TABLES[table]
        share, px, _ = R.measure("CAM-RGB-W", np.array([t["c"][0], t["c"][1], 2.6]), b.c + np.array([0, 0, -0.001]), f"/World/Facility/Items/{n}")
        rows[f"3:{n}"] = dict(share=_r(share), px2=px, ok=bool(share >= th["top_share"]))
    sq = np.array([th["square"][0], th["square"][1], th["mast_z"]])
    grip = W.find(occ, "Handle/grip")[0]
    share, px, _ = R.measure("CAM-RGB-M", sq, grip.c, "/World/Facility/SteamLine/Handle")
    rows["4:grip@day"] = dict(share=_r(share), px2=px, ok=bool(share >= th["grip_share"] and px >= th["grip_px2"]),
                              raycast=[ray_rows.get(k) for k in ray_rows if k.startswith("4:")])
    dial = W.find(occ, "Gauge/Dial")[0]
    share, px, across = R.measure("CAM-RGB-M", sq, dial.c, "/World/Facility/SteamLine/Gauge/Dial")
    _, npx, _ = R.measure("CAM-RGB-M", sq, dial.c, "/World/Facility/SteamLine/Gauge/needle")
    needle_w = npx / max(1.0, W.dial_needle_px(sq)["needle_len_px"])
    rows["5:dial"] = dict(share=_r(share), dial_px=across, needle_px=round(needle_w, 2),
                          ok=bool(share >= th["dial_share"] and across >= th["dial_px"] and needle_w >= th["needle_px"]))
    dx, dy, dpsi = day["ugv"]
    root = W.UGV_ROOT + np.array([dx, dy, 0.0])
    best6 = None
    for turn in np.arange(-th["start_turns_deg"], th["start_turns_deg"] + 0.1, 15.0):
        C, d, u, r = W.cam_world(root, W.ugv_frame(root[:2], dpsi + turn), W.UGV_MAST)
        share, px, _ = R.measure("CAM-RGB-M", C, C + d, "/World/Facility/SteamLine/Gauge/Dial")
        dist = float(np.linalg.norm(dial.c - C))
        r8 = 1920 * math.degrees(2 * math.atan(0.05 / dist)) / 70.3 if px else 0.0
        key = (share >= th["start_dial_share"] and r8 >= th["start_dial_px"], share)
        if best6 is None or key > best6[0]:
            best6 = (key, dict(turn_deg=float(turn), share=_r(share), px=round(r8, 1)))
    rows["6:dial_from_start"] = dict(best6[1], ok=bool(best6[0][0]))
    H = np.array(th["approach"])
    C, d, u, r = W.cam_world(H, np.eye(3), W.UAV_CAMS["Thermal"])
    share, px, across = R.measure("THERM-C", C, C + d, "/World/Facility/FirePan")
    rows["7:pan"] = dict(share=_r(share), px=across, ok=bool(share >= th["pan_share"] and across >= th["pan_px"]))
    C, d, u, r = W.cam_world(root, W.ugv_frame(root[:2], dpsi), W.UGV_MAST)
    prims = {"dial": "/World/Facility/SteamLine/Gauge/Dial", "valve_block": "/World/Facility/SteamLine/ValveBody",
             "lever_grip": "/World/Facility/SteamLine/Handle", "support_1": "/World/Facility/SteamLine/Pipe/support_1",
             "support_2": "/World/Facility/SteamLine/Pipe/support_2", "support_3": "/World/Facility/SteamLine/Pipe/support_3"}
    seen = []
    for k, pr in prims.items():
        share, px, _ = R.measure("CAM-RGB-M", C, C + d, pr)
        if share >= th["landmark_share"] and px >= th["landmark_px2"]:
            seen.append(k)
    rows["8:start_landmarks"] = dict(seen=seen, n=len(seen), ok=len(seen) >= th["landmarks"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, dest="set_file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--raycast", action="store_true")
    ap.add_argument("--bay-rows", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(os.path.join(TESTS, "grader_config.json")))
    missions = json.load(open(a.set_file))["missions"]
    ids = [x for x in a.ids.split(",") if x]
    import extract_geom_lib
    bay = extract_geom_lib.bay_shapes(os.path.join(TESTS, "hidden", "bay_true.usda"))
    import visibility as V
    import views as W
    R = None if a.raycast else Render(cfg, a.out)
    h = file_hash()
    for m in missions:
        if ids and m["id"] not in ids:
            continue
        day = m["day"]
        rows = raycast_rows(day, bay, V, W)
        mode = "raycast"
        if R is not None:
            R.open_day(day, os.path.join(TESTS, "hidden", "bay_true.usda"))
            rows = render_rows(day, R, rows, V, W, bay)
            mode = "renderer"
        res = dict(id=m["id"], mode=mode, check_sha256=h, check_file_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                   thresholds=THRESHOLDS, rows=rows, ok=all(r["ok"] for r in rows.values()),
                   failed=[k for k, r in rows.items() if not r["ok"]])
        json.dump(res, open(os.path.join(a.out, f"{m['id']}.obs.json"), "w"), indent=1)
        print(m["id"], mode, "PASS" if res["ok"] else f"FAIL {res['failed']}", flush=True)
    if a.bay_rows:
        rows = raycast_bay_rows(bay, V, W)
        res = dict(mode="raycast", check_sha256=h, rows=rows, ok=all(r["ok"] for r in rows.values()), failed=[k for k, r in rows.items() if not r["ok"]])
        json.dump(res, open(os.path.join(a.out, "bay.obs.json"), "w"), indent=1)
        print("bay rows", "PASS" if res["ok"] else f"FAIL {res['failed']}", flush=True)
    if R is not None:
        R.app.close()


if __name__ == "__main__":
    main()
