"""The customer's room: fixed geometry in metres (the survey the cell manual publishes exactly).

Room frame: origin at the south-west floor corner, x east, y north, z up. The bookcase is built on the customer's base
cabinet (a low cabinet standing free in the room, as wide and as deep as the bookcase); the drawings' floor plan shows
the cabinet and the bookcase's place on it. The flat-pack arrives unpacked: the tall panels stand on a panel cart, the horizontal panels lie
on two battens on a long kitting table. The customer's couch, plant and rug stay where they are, in the north-east and
north-west corners.
"""

from __future__ import annotations

import math

ROOM_X, ROOM_Y, WALL_H = 6.0, 5.0, 2.6
WALL_T = 0.12
DOOR_X = (4.80, 5.70)            # opening in the south wall
WINDOW_Y = (1.60, 3.00)          # window in the east wall (sill 0.90 m, head 2.10 m)

# the base cabinet (static, the customer's): centre of its footprint, size (x, y, z); the bookcase stands on it,
# flush with its sides and its back
SIDEBOARD = {"center": (3.40, 3.30), "size": (0.80, 0.36, 0.45)}
SIDEBOARD_TOP = SIDEBOARD["size"][2]

# the bookcase's place: the furniture frame's origin (middle of the bookcase's front bottom edge) in the room, and its
# yaw (0 = the bookcase's front faces south, furniture +y = room +y). Back flush with the sideboard's back.
TARGET = (SIDEBOARD["center"][0], SIDEBOARD["center"][1] + SIDEBOARD["size"][1] / 2 - 0.300, SIDEBOARD_TOP, 0.0)

# the tall panels (the two side panels and the back panel) stand upright on a panel cart: three grooves running
# east-west, each on its own 0.45 m bed block, each groove formed by two rails; a board stands on its bottom edge in a groove
PANEL_CART = {"x": 1.05, "slots_y": (1.15, 2.15, 3.15), "bed_h": 0.45, "bed_w": 0.16, "rail_len": 0.90,
              "rail_w": 0.03, "rail_h": 0.05, "play": 0.004}     # each groove is its board's thickness + play

# kitting table: long axis along x; the horizontal panels lie in a row along it, long axis along x, on two battens
TABLE = {"center": (4.20, 1.60), "size": (2.50, 0.50), "height": 0.72, "batten_h": 0.050, "batten_w": 0.04,
         "batten_dy": 0.035}
TABLE_SLOTS_X = (-0.82, 0.0, 0.82)       # panel centres along the table, relative to its centre

# the customer's own furniture, out of the assembly's way (static): a two-seat couch against the east wall in the
# north-east corner, seat towards the west, and a potted plant in the north-west corner (both obstacles), and a rug
# in front of the couch (flat on the floor, drawn only: not an obstacle). Boxes are (x0, y0, z0, x1, y1, z1), metres.
COUCH = {
    "footprint": (5.09, 3.36, 5.95, 4.94),
    "boxes": {                                       # the upholstered body on four legs
        "seat": (5.09, 3.49, 0.16, 5.75, 4.81, 0.48),
        "back": (5.75, 3.49, 0.16, 5.95, 4.81, 0.94),
        "arm_s": (5.09, 3.36, 0.16, 5.95, 3.49, 0.72),
        "arm_n": (5.09, 4.81, 0.16, 5.95, 4.94, 0.72),
    },
    "legs": {"centers": ((5.12, 3.39), (5.12, 4.91), (5.93, 3.39), (5.93, 4.91)), "side": 0.035, "height": 0.16},
}
# cylinders: centre (x, y), radius, (z0, z1)
PLANT = {"pot": {"center": (0.51, 4.51), "radius": 0.13, "z": (0.0, 0.25)},
         "foliage": {"center": (0.45, 4.55), "radius": 0.40, "z": (0.25, 1.62)}}
RUG = {"center": (4.85, 4.15), "size": (1.27, 1.52), "thickness": 0.01}

# robots' start poses (x, y, yaw) before the per-episode draw
ROBOT_HOME = {"a": (1.90, 0.55, math.pi / 2), "b": (3.20, 0.55, math.pi / 2)}

# fixed ceiling cameras (exact calibration is published): position, look-at point, vertical fov (deg), size
CEILING_CAMERAS = {
    "ceiling_west": {"pos": (1.90, 0.60, 2.50), "target": (1.45, 2.10, 0.40), "fovy": 72.0, "size": (960, 720)},
    "ceiling_east": {"pos": (3.70, 0.90, 2.50), "target": (3.60, 2.50, 0.50), "fovy": 72.0, "size": (960, 720)},
}
