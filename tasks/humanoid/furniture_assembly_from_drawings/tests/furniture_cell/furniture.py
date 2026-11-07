"""The B930 bookcase: parts, features and joints, in millimetres, in the furniture frame.

Furniture frame: origin on the floor at the middle of the bookcase's front bottom edge; x to the right when facing
the front, y towards the back, z up. Every part's design orientation is the identity (its local axes are the
furniture axes), so a part's design pose is its centre. This module is the single description of the furniture:
the physics cell builds its bodies, collision boxes, dowels and holes from it, the drawings are rendered from it,
and the grader scores against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PARTICLEBOARD_KG_M3 = 650.0
HARDBOARD_KG_M3 = 900.0

WIDTH, DEPTH, HEIGHT = 800.0, 300.0, 930.0
BOARD = 18.0                       # particleboard thickness
BACK = 6.0                         # hardboard back thickness
PLINTH = 0.0                       # floor to the underside of the bottom panel (flush with the sides' bottom edges)
RABBET_W, RABBET_D = 8.0, 6.0      # rabbet along the sides' inner back edge: width (along y) x depth (into x)
INNER = WIDTH - 2 * BOARD          # 764: length of the horizontal panels
H_DEPTH = DEPTH - RABBET_W         # 292: depth of top / bottom (front-flush, stop at the rabbet)
SHELF_DEPTH = 264.0                # fixed shelf, recessed at the front
SHELF_SETBACK = H_DEPTH - SHELF_DEPTH  # 28
SHELF_Z = 470.0                    # underside of the fixed shelf
DOWEL_D, DOWEL_OUT, DOWEL_IN = 8.0, 10.0, 20.0
DOWEL_INSET = 40.0                 # dowel centres from the front and back edges of each horizontal panel
BACK_CLEAR = 4.0                   # side clearance of the back panel in the rabbets
BACK_W = INNER + 2 * RABBET_D - 2 * BACK_CLEAR   # 768
BACK_H = HEIGHT - PLINTH           # 930: from the bottom panel's underside to the top


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in the part's local frame (centre, half sizes), mm."""

    center: tuple[float, float, float]
    half: tuple[float, float, float]


@dataclass(frozen=True)
class Peg:
    """A dowel (on a guest part) or a blind hole (on a host part): base centre on the face, unit axis.

    A dowel's axis points out of its part (the direction it enters the host); a hole's axis points into its part.
    """

    pos: tuple[float, float, float]
    axis: tuple[float, float, float]
    group: str                      # the joint this peg belongs to, e.g. "L.top" or "T.left"


@dataclass(frozen=True)
class Part:
    key: str                        # stable id: L, R, T, C, D, K
    letter: str                     # the drawings' part letter
    name: str
    size: tuple[float, float, float]
    design_center: tuple[float, float, float]
    material: str
    boxes: tuple[Box, ...]
    dowels: tuple[Peg, ...] = ()
    holes: tuple[Peg, ...] = ()
    banded_face: str | None = None  # face carrying the white edge banding ("-y" = front in the design pose)
    finished_face: str | None = None  # the white face of the back panel ("-y")

    @property
    def half(self) -> tuple[float, float, float]:
        return (self.size[0] / 2, self.size[1] / 2, self.size[2] / 2)

    @property
    def volume_m3(self) -> float:
        return sum(8 * b.half[0] * b.half[1] * b.half[2] for b in self.boxes) * 1e-9

    @property
    def mass_kg(self) -> float:
        rho = HARDBOARD_KG_M3 if self.material == "hardboard" else PARTICLEBOARD_KG_M3
        return round(self.volume_m3 * rho, 3)


@dataclass(frozen=True)
class Joint:
    """A designed connection: the guest's dowel group enters the host's hole group, or the back panel seats."""

    key: str
    host: str
    guest: str
    host_group: str
    guest_group: str
    kind: str                       # "dowel" or "rabbet"
    weight: float
    insert_dir: tuple[float, float, float]  # the guest's motion that seats it, furniture frame


def _side(key: str) -> Part:
    left = key == "L"
    sx = -1.0 if left else 1.0                       # which side of the bookcase
    cx = sx * (WIDTH / 2 - BOARD / 2)
    center = (cx, DEPTH / 2, HEIGHT / 2)
    inner = -sx                                      # local x direction of the inner face
    hx, hy, hz = BOARD / 2, DEPTH / 2, HEIGHT / 2
    # main body up to the rabbet, then the thinner strip behind it (the rabbet is cut from the inner face)
    main = Box((0.0, -RABBET_W / 2, 0.0), (hx, hy - RABBET_W / 2, hz))
    strip_half_x = (BOARD - RABBET_D) / 2
    strip = Box((-inner * (RABBET_D / 2), hy - RABBET_W / 2, 0.0), (strip_half_x, RABBET_W / 2, hz))
    holes = []
    for name, (y0, depth, zc) in {
        "top": (0.0, H_DEPTH, HEIGHT - BOARD / 2),
        "shelf": (SHELF_SETBACK, SHELF_DEPTH, SHELF_Z + BOARD / 2),
        "bottom": (0.0, H_DEPTH, PLINTH + BOARD / 2),
    }.items():
        for yf in (y0 + DOWEL_INSET, y0 + depth - DOWEL_INSET):
            holes.append(Peg((inner * hx, yf - center[1], zc - center[2]), (-inner, 0.0, 0.0), f"{key}.{name}"))
    return Part(
        key=key, letter="A" if left else "B", name=("left" if left else "right") + " side panel",
        size=(BOARD, DEPTH, HEIGHT), design_center=center, material="particleboard",
        boxes=(main, strip), holes=tuple(holes), banded_face="-y",
    )


def _horizontal(key: str, letter: str, name: str, depth: float, y0: float, z_under: float) -> Part:
    center = (0.0, y0 + depth / 2, z_under + BOARD / 2)
    hx, hy, hz = INNER / 2, depth / 2, BOARD / 2
    dowels = []
    for end, sx in (("left", -1.0), ("right", 1.0)):
        for yf in (y0 + DOWEL_INSET, y0 + depth - DOWEL_INSET):
            dowels.append(Peg((sx * hx, yf - center[1], 0.0), (sx, 0.0, 0.0), f"{key}.{end}"))
    return Part(
        key=key, letter=letter, name=name, size=(INNER, depth, BOARD), design_center=center,
        material="particleboard", boxes=(Box((0.0, 0.0, 0.0), (hx, hy, hz)),), dowels=tuple(dowels),
        banded_face="-y",
    )


def _back() -> Part:
    center = (0.0, H_DEPTH + BACK / 2 + 1.0, PLINTH + BACK_H / 2)
    half = (BACK_W / 2, BACK / 2, BACK_H / 2)
    return Part(
        key="K", letter="E", name="back panel", size=(BACK_W, BACK, BACK_H), design_center=center,
        material="hardboard", boxes=(Box((0.0, 0.0, 0.0), half),), finished_face="-y",
    )


PARTS: dict[str, Part] = {
    "L": _side("L"),
    "R": _side("R"),
    "T": _horizontal("T", "C", "top panel", H_DEPTH, 0.0, HEIGHT - BOARD),
    "C": _horizontal("C", "C", "bottom panel", H_DEPTH, 0.0, PLINTH),
    "D": _horizontal("D", "D", "fixed shelf", SHELF_DEPTH, SHELF_SETBACK, SHELF_Z),
    "K": _back(),
}

JOINTS: tuple[Joint, ...] = tuple(
    Joint(f"{side}-{g}", side, g, f"{side}.{grp}", f"{g}.{'left' if side == 'L' else 'right'}", "dowel", 1.0,
          (-1.0, 0.0, 0.0) if side == "L" else (1.0, 0.0, 0.0))
    for side in ("L", "R") for g, grp in (("T", "top"), ("D", "shelf"), ("C", "bottom"))
) + (Joint("K-seat", "L", "K", "L.rabbet", "K.panel", "rabbet", 1.5, (0.0, -1.0, 0.0)),)

PLACEMENT_WEIGHT = 1.5
TOTAL_WEIGHT = sum(j.weight for j in JOINTS) + PLACEMENT_WEIGHT

# the parts the drawings list, by letter: quantity and which part keys share the letter
LETTERS: dict[str, tuple[str, ...]] = {}
for _p in PARTS.values():
    LETTERS.setdefault(_p.letter, ())
    LETTERS[_p.letter] += (_p.key,)


def summary() -> str:
    lines = []
    for p in PARTS.values():
        lines.append(f"{p.key} ({p.letter}, {p.name}): size {p.size} mm, centre {p.design_center}, "
                     f"{p.mass_kg} kg, {len(p.dowels)} dowels, {len(p.holes)} holes")
    lines.append(f"joints: {[j.key for j in JOINTS]}; total weight {TOTAL_WEIGHT}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
