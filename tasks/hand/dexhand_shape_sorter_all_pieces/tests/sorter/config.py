"""Fixed task constants. The grader uses exactly these values."""
import math

# ---- simulator (SuperDex 1.0.0, double precision, single-threaded: sp.initialize(0)) ----
SIM_DT = 0.002            # s per physics step
NEWTON_ITERS = 20         # Backward-Euler, non-linear solver iterations
ABS_TOL = 1e-6
GRAVITY = (0.0, 0.0, -9.81)

# ---- control ----
CONTROL_HZ = 25                       # plan waypoints per second
SUBSTEPS = 20                         # physics steps per waypoint (= 0.04 s / 0.002 s)
MAX_WAYPOINTS = 4500                  # 180 s of motion per instance
SETTLE_PRE_S = 0.5                    # hand holds the park pose before the first waypoint
SETTLE_POST_S = 2.0                   # last targets held after the last waypoint, then scoring
MAX_LIN_SPEED = 0.5                   # m/s between consecutive wrist waypoints
MAX_ANG_SPEED = 4.0                   # rad/s
MAX_FINGER_SPEED = 6.0                # rad/s per joint
WRIST_BOUNDS = ((-0.50, 0.50), (-0.40, 0.40), (0.0, 0.60))   # world metres
SETPOINT_CLAMP_POS = 0.025            # m: the internal wrist set-point never leads the wrist by more than this
SETPOINT_CLAMP_ROT = 0.40             # rad
# Wrist speed limit on the executed motion (r5). An arm-mounted hand cannot be flung: (1) per 2 ms physics step the
# internal wrist set-point moves at most SP_MAX_LIN_SPEED * SIM_DT (position) and SP_MAX_ANG_SPEED * SIM_DT
# (orientation) away from the previous set-point (1.2 x the plan limits, so a plan within its limits is never slowed
# down); (2) the set-point's allowed lead over the actual wrist falls linearly with the wrist's actual speed towards it,
# like a motor's force-speed curve: lead <= SETPOINT_CLAMP_POS * max(0, 1 - v / WRIST_V0_LIN), and for the orientation
# sin(lead angle / 2) <= sin(SETPOINT_CLAMP_ROT / 2) * max(0, 1 - w / WRIST_V0_ANG) (v, w = actual wrist speed towards
# the set-point over the last physics step). A blocked hand still pushes with the full clamp force; once it comes free
# it moves on at about the plan speeds instead of jumping to its target.
SP_MAX_LIN_SPEED = 0.6                # m/s
SP_MAX_ANG_SPEED = 4.8                # rad/s
WRIST_V0_LIN = 0.8                    # m/s: the wrist drive gives no push towards the set-point at this speed
WRIST_V0_ANG = 6.4                    # rad/s
COS_HALF_SP_ROT_STEP = 0.9999884800221184   # cos(0.5 * SP_MAX_ANG_SPEED * SIM_DT), written out (no libm call)
SIN_HALF_CLAMP_ROT = 0.19866933079506122    # sin(0.5 * SETPOINT_CLAMP_ROT)

# ---- hand: public SuperDex Wuji Hand 2 (beta1), right, floating wrist (gravity-compensated) ----
HAND_BOT = "bots/hands/wuji_hand2_beta1/right/wuji_hand2_beta1_right.superdex_bot"
WRIST_KP, WRIST_KD = 5000.0, 100.0    # N/m, N s/m (implicit PD on the root link position)
WRIST_KR, WRIST_KRD = 500.0, 10.0     # N m/rad, N m s/rad (root link orientation)
FINGER_KP, FINGER_KD = 30.0, 1.2      # per finger joint (N m/rad, N m s/rad)
PARK_POS = (-0.05, 0.0, 0.30)
PARK_QUAT = (-0.085497, 0.648759, -0.07424, 0.752523)   # xyzw, palm down, fingers towards +x
# Wrist orientation limit (an arm-mounted hand cannot turn palm-up at the table): the palm normal (out of the palm; in
# the wrist frame it is R_park^T (0, 0, -1), i.e. straight down at the park pose) must stay within WRIST_MAX_PALM_DEG
# of world down (0, 0, -1). 0 = palm down, 90 = palm sideways, 180 = palm up. See plan.palm_angle_deg.
WRIST_MAX_PALM_DEG = 105.0
COS_MAX_PALM = -0.25881904510252085   # cos(105 deg), written out so no libm call is needed
SIN_MAX_PALM = 0.9659258262890683     # sin(105 deg)
# Finger direction limit (r6; an arm-mounted hand at a tabletop does not point its fingers up with the wrist underneath):
# the finger axis (wrist -> fingertips; in the wrist frame it is R_park^T (1, 0, 0), i.e. horizontal towards +x at the
# park pose) may point at most FINGER_MAX_PITCH_DEG above the horizontal. Pointing down is not limited. See
# plan.finger_pitch_deg.
FINGER_MAX_PITCH_DEG = 35.0
SIN_MAX_PITCH = 0.573576436351046     # sin(35 deg), written out
COS_MAX_PITCH = 0.8191520442889918    # cos(35 deg)

# ---- scene ----
BOX_POS = (0.10, 0.0, 0.0)            # shape_box body frame origin (bottom centre) in world, no rotation
LID_OFFSET = (0.003, 0.0, 0.125025)   # lid frame relative to the body frame (from the public prefab)
TABLE_SIZE = (1.0, 0.8, 0.04)         # top face at z = 0
PIECE_DENSITY = 390.0                 # kg/m^3 (public prefab)
MU_PIECE, MU_TABLE, MU_BOX = 0.8, 0.6, 0.6
# Per-scene physics (world.physics_params). Nominal = the values above; the graded hidden scenes draw each parameter
# once, uniformly within PHYSICS_BOUNDS (values hidden).
PHYSICS_NOMINAL = {"mu_piece": MU_PIECE, "mu_cube": MU_BOX, "finger_mu_scale": 1.0, "mass_scale": 1.0, "finger_gain": 1.0}
PHYSICS_BOUNDS = {"mu_piece": (0.5, 1.1), "mu_cube": (0.35, 0.85), "finger_mu_scale": (0.6, 1.4), "mass_scale": (0.6, 1.6),
                  "finger_gain": (0.6, 1.4)}
# Per-scene hand-eye calibration error (world.calibration_error): each component of the translation dp uniform in
# [-CALIB_MAX_POS, CALIB_MAX_POS], each component of the rotation vector (world axes, about the wrist origin) uniform in
# [-CALIB_MAX_ROT_DEG, CALIB_MAX_ROT_DEG] degrees; drawn once per scene (values hidden).
CALIB_MAX_POS = 0.0015                # m
CALIB_MAX_ROT_DEG = 0.5               # deg
CALIB_BOUNDS = {"dp": (-CALIB_MAX_POS, CALIB_MAX_POS), "rotvec_deg": (-CALIB_MAX_ROT_DEG, CALIB_MAX_ROT_DEG)}
FRICTION_FALLOFF = 1e-3               # m/s (pieces, cube, table)
HAND_FRICTION_FALLOFF = 1e-4          # m/s (hand links; other contact settings are the asset's)
# Finger flexion joints may not hyperextend: the public asset allows -1.047 rad (60 deg) of backward bending on every
# flexion joint, which bends fingers dorsally. The task sets their lower limit to 0 (thumb CMC joints and all
# abduction joints keep the asset's ranges).
NO_HYPEREXTENSION_SUFFIXES = ("thumb_mcp", "thumb_ip", "_mcp_flex", "_pip", "_dip")
PREFAB_CONTACT = dict(penalty=999999995904.0, smooth=9.999999747378752e-05,
                      thresh=9.999999747378752e-05, pad=9.999999747378752e-05)
PIECES = ["cross", "triangle", "square", "trapezoid", "rectangle", "pentagon", "parallelogram",
          "octagon", "hexagon", "ellipse", "diamond"]   # the public star is larger than its hole; not used
LAYOUT_REGION = ((-0.42, -0.04), (-0.33, 0.33))        # x, y range of piece centroids in generated layouts
