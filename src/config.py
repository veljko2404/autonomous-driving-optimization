import math

WHEELBASE = 2.6
DT = 0.05
MAX_TIME = 40.0

A_MIN, A_MAX = -6.0, 3.0
V_MIN, V_MAX = 0.0, 30.0
DELTA_MIN, DELTA_MAX = -math.radians(35), math.radians(35)

TRACK_WIDTH = 6.0

BOUNDS = {
    "Ld": (2.0, 20.0),
    "Kp": (0.0, 6.0),
    "Ki": (0.0, 2.0),
    "Kd": (0.0, 2.0),
    "v_ref": (4.0, 24.0),
    "delta_max_deg": (10.0, 35.0),
}

DEFAULT_THETA = {
    "Ld": 8.0,
    "Kp": 1.8,
    "Ki": 0.2,
    "Kd": 0.4,
    "v_ref": 14.0,
    "delta_max_deg": 28.0,
}
