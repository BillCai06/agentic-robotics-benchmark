"""The agent's cameras of the OpenArm pair cell: MuJoCo's Filament renderer (physically based, Vulkan) in the baked
factory hall, as in robotic_arm_full_assembly_so101_panda (`arm/assembly_full/agent_render.py`, whose session class this
module reuses), on the pair's cell.

One session per episode. The render twin is the episode's physics MjSpec (same bodies and joint names) without the
collision geoms and the floor plane, plus a steel workbench frame under the 1.6 x 1.4 m bench top, the datum boards
(a checkerboard on the bench, one on the gallows' foot plate, a plate on each Panda's hand: physical facts the cell
manual documents), the factory room bundle (baked lightmaps, its lamps lighting the live objects, its HDR image light)
and the look (`agent_look.json` + the material classes below). RGB from Filament as an exact pinhole, depth from the
MuJoCo backend on the same geometry; lens distortion, noise, blur, vignetting, colour jitter and the depth-sensor model
in render/postprocess.py. `filament=False` (no Filament module, or the config asks for MuJoCo): the same twin through
MuJoCo's classic renderer, lit by two lamps.
"""
import json
import math
import os

import numpy as np

from ..assembly_full import agent_render as FA

__all__ = ["open_session", "available", "render_spec", "BENCH_BOARD", "FIXTURE_BOARD"]

LOOK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_look.json")
TOP_HALF = (0.8, 0.7)
# the datum boards (the integrator's survey, exact by construction): the bench board on the bench top beyond the gallows'
# foot; the fixture board on the foot plate (its top at 10 mm), in the base's frame; the hand plates (the information
# tier's: on each Panda hand's -x flank)
BENCH_BOARD = {"cols": 9, "rows": 6, "square_mm": 20.0, "pos": [0.46, 0.0, 0.0], "yaw_deg": 90.0}
FIXTURE_BOARD = {"cols": 5, "rows": 4, "square_mm": 10.0, "margin_mm": 4.0, "thickness_mm": 2.0, "pos_world_nominal": [0.235, 0.0, 0.010]}
_SPECS = {}


def available():
    return FA.available()


def make_look(exposure_factor=1.0):
    from render_filament.backend import Look
    look = Look(name="factory-agent-openarm", version="1")
    look.materials = [
        (None, "mat_alu", dict(roughness=0.38, metallic=1.0)),          # machined / anodised aluminium plates
        (None, "mat_steel", dict(roughness=0.32, metallic=1.0)),        # steel shafts, the gallows' column
        (None, "mat_dark", dict(roughness=0.5, metallic=0.0)),          # the motors' black cases, the gripper's prints
        (None, "mat_rotor", dict(roughness=0.3, metallic=1.0)),         # the motors' output flanges
        (None, "mat_mark", dict(roughness=0.5, metallic=0.0)),          # the flanges' index ticks
        (None, "mat_riser", dict(roughness=0.4, metallic=1.0)),         # the aluminium riser blocks
        (None, "pl/*", dict(roughness=0.38, metallic=0.0)),             # the Pandas' painted shells
        (None, "pr/*", dict(roughness=0.38, metallic=0.0)),
        (None, "mat_bench", dict(roughness=1.0, metallic=0.0)),         # x the oak's roughness map
        (None, "bench_steel", dict(roughness=0.4, metallic=1.0)),
        (None, "bench_shelf", dict(roughness=0.55, metallic=1.0)),
        (None, "*_board__*", dict(roughness=0.8, metallic=0.0)),
        (None, "hand_plate*", dict(roughness=0.8, metallic=0.0)),
        (None, None, dict(roughness=0.6, metallic=0.0)),
    ]
    look.surface_material, look.surface_geom, look.surface_map_m = "mat_bench", "table_top", 1.83
    look.surface_albedo_gain = 1.0
    look.override({k: v for k, v in json.load(open(LOOK_FILE)).items() if not k.startswith("_")})
    oak = os.path.join(FA._render_dir(), "oak")
    look.surface_dir = oak if os.path.isdir(oak) else None
    ibl = os.path.join(FA._room_dir(), "ibl", "ibl.ktx")
    look.room_ibl_ktx = ibl if os.path.isfile(ibl) else None
    look.exposure_scale = float(look.exposure_scale) * float(exposure_factor)
    return look


def _frame(z, x):
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(x, float) - z * float(np.dot(x, z))
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, np.cross(z, x), z
    return T


def board_poses(server):
    """{name: (parent body name, T_in_parent, ref)} of the episode's datum boards"""
    from ..assembly_info.cell_info import HAND_PLATE
    from ..assembly_info.cell_render import board_ref
    out = {}
    b = BENCH_BOARD
    T = _frame([0, 0, 1], [math.cos(math.radians(b["yaw_deg"])), math.sin(math.radians(b["yaw_deg"])), 0])
    T[:3, 3] = b["pos"]
    out["bench_board"] = ("world", T, board_ref("bench_board", b["cols"], b["rows"], b["square_mm"]))
    f = FIXTURE_BOARD
    Tb = server.fa.body_T("base")
    Tw = _frame([0, 0, 1], [0, 1, 0])
    fx = server.layout["fixture"]["xy"]
    Tw[:3, 3] = np.asarray(f["pos_world_nominal"], float) + np.array([fx[0], fx[1], 0.0])
    out["fixture_board"] = ("base", np.linalg.inv(Tb) @ Tw, board_ref("fixture_board", f["cols"], f["rows"], f["square_mm"], f["margin_mm"], f["thickness_mm"]))
    h = HAND_PLATE
    J = np.asarray(server.episode["hand_plate_jitter"], float)
    for api, pn in (("left", "pl"), ("right", "pr")):
        ref = board_ref("hand_plate_" + api, h["cols"], h["rows"], h["square_mm"], h["margin_mm"], h["thickness_mm"])
        T = _frame(h["normal_in_hand"], h["cols_along_in_hand"])
        T[:3, 3] = np.asarray(h["pos_in_hand_m"], float)
        out["hand_plate_" + api] = (pn + "/hand", T @ J, ref)
    return out


def render_spec(server):
    """the render twin (no room): the physics spec's visual geoms, the workbench, the datum boards"""
    from ..assembly_info.cell_render import attach_ref
    spec = server.fa.spec.copy()
    for g in list(spec.geoms):
        nm = g.name or ""
        if nm == "floor" or int(g.group) > 2:
            spec.delete(g)
    FA.TOP_HALF = TOP_HALF            # the pair's bench: 1.6 x 1.4 m
    FA._workbench(spec)
    for name, (parent, T, ref) in board_poses(server).items():
        body = spec.worldbody if parent == "world" else [b for b in spec.bodies if b.name == parent][0]
        attach_ref(spec, body, ref, T)
    return spec


def depth_scene(key):
    return _SPECS[key].copy()


def _mjrf():
    """the Filament module, with MuJoCo 3.14's names on 3.17 (ModelObjects -> ModelResources; a texture's
    `set_data` -> `upload`, in the room's image light)"""
    import mujoco._render_filament as mjrf
    if not hasattr(mjrf, "ModelObjects") and hasattr(mjrf, "ModelResources"):
        mjrf.ModelObjects = mjrf.ModelResources
    if not hasattr(mjrf.Texture, "set_data"):
        from render_filament import backend as FB

        def _ktx_ibl(self, ctx, scn, path=None, intensity=None):
            raw = open(path, "rb").read() if path else self._ibl_bytes()
            tc = mjrf.TextureConfig()
            tc.width, tc.height, tc.format = len(raw), 0, mjrf.PixelFormat.PIXEL_FORMAT_KTX
            tc.sampler_type, tc.color_space = mjrf.SamplerType.TEXTURE_CUBE, mjrf.ColorSpace.COLORSPACE_LINEAR
            tex = ctx.create_texture(tc)
            td = mjrf.TextureData()
            td.set_image(raw)
            tex.upload(td)
            lp = mjrf.LightParams()
            lp.type, lp.texture, lp.color = mjrf.LightType.LIGHT_IMAGE, tex, [1.0, 1.0, 1.0]
            lp.intensity = float(self.look.ibl_intensity if intensity is None else intensity)
            ibl = ctx.create_light(lp)
            scn.add_light(ibl)
            return ibl, tex, td
        FB.FilamentBackend._ktx_ibl = _ktx_ibl
    return mjrf


class PairSession(FA.FilamentAgentSession):
    """the full-arm task's Filament session; MuJoCo 3.17's readback (ReadRequest + a caller's buffer) where 3.14 had
    ReadPixelsRequest (alloc / buffer)"""

    def _target(self, w, h):
        mjrf = self._mjrf
        if hasattr(mjrf, "ReadPixelsRequest"):
            return FA.FilamentAgentSession._target(self, w, h)
        if (w, h) not in self._targets:
            t = self.ctx.create_render_target(mjrf.RenderTargetConfig(color_format=mjrf.PixelFormat.PIXEL_FORMAT_RGB8))
            t.resize(w, h)
            rd = mjrf.ReadRequest()
            rd.target = t
            buf = np.zeros(w * h * 3, np.uint8)
            rd.set_buffer(buf)
            self._targets[(w, h)] = (t, rd, buf)
        return self._targets[(w, h)]

    def _rgb(self, cam):
        if hasattr(self._mjrf, "ReadPixelsRequest"):
            return FA.FilamentAgentSession._rgb(self, cam)
        from render_filament.backend import gl_camera
        mjrf = self._mjrf
        target, rd, buf = self._target(cam.width, cam.height)
        rq = mjrf.RenderRequest(camera=gl_camera(cam, self.near, self.far), draw_mode=mjrf.DrawMode.DRAW_MODE_DEFAULT)
        rq.scene, rq.target = self.scn, target
        rq.viewport.left, rq.viewport.bottom, rq.viewport.width, rq.viewport.height = 0, 0, cam.width, cam.height
        self.ctx.wait_for_frame(self.ctx.render([rq], [rd]))
        rgb = buf.reshape(cam.height, cam.width, 3)
        if self.lut is None:
            return rgb.copy()
        idx = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
        return self.lut[idx]


def open_session(server, filament=True):
    from render.spec import SceneSpec
    key = "oa-episode-%d-%d" % (id(server), int(server.ep_seed) & 0xFFFFFFFF)
    spec = render_spec(server)
    ph = server.cfg["photo"]
    content = [os.path.join(server.kit_dir, "meshes", "visual")]
    if filament:
        FA._room_geometry(spec, FA._room_dir())
        content.append(os.path.join(FA._room_dir(), "room.json"))
    _SPECS.clear()
    _SPECS[key] = spec
    if not filament:
        from render.backends import get_backend
        from render.spec import BackgroundSpec, HeadlightSpec, LightSpec
        be = get_backend("mujoco")
        L = server.episode["lights"]
        lc = server.episode["config"]["lights"]
        lights = []
        for k in ("key", "fill"):
            pos = np.asarray(lc[k]["pos"], float)
            d = np.asarray(lc[k]["target"], float) - pos
            d /= np.linalg.norm(d)
            lights.append(LightSpec(name="light_" + k, pos=pos.tolist(), dir=d.tolist(), diffuse=[L[k]] * 3, specular=[0.2 * L[k]] * 3,
                                    castshadow=(k == "key"), cutoff_deg=70.0, exponent=2.0))
        sc = SceneSpec(builder="arm.assembly_openarm.agent_render:depth_scene", builder_args={"key": key}, content_files=content,
                       lights=lights, replace_lights=True, headlight=HeadlightSpec(ambient=[L["dome"]] * 3, diffuse=[0.08] * 3, specular=[0.02] * 3),
                       background=BackgroundSpec(rgb1=[0.78, 0.78, 0.79], rgb2=[0.42, 0.42, 0.43]), shadows=True,
                       shadowsize=int(ph.get("shadowsize", 2048)), lod=str(ph.get("lod", "full")), near_m=0.01)
        return be.open(sc)
    sc = SceneSpec(builder="arm.assembly_openarm.agent_render:depth_scene", builder_args={"key": key}, content_files=content,
                   shadows=False, lod=str(ph.get("lod", "full")), near_m=0.01)
    # the full-arm session builds its Filament twin with `_base_spec(server)` and its look with `make_look`: the pair's
    old_base, old_look = FA._base_spec, FA.make_look
    FA._base_spec, FA.make_look = (lambda _s: render_spec(server)), make_look
    try:
        _mjrf()
        be = FA.FilamentAgentBackend()
        f = float(server.episode["lights"]["key"]) / FA.NOMINAL_KEY
        return PairSession(be, server, sc, f)
    finally:
        FA._base_spec, FA.make_look = old_base, old_look
