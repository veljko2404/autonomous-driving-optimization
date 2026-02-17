import math
from dataclasses import dataclass
from typing import Tuple
from .dynamics import State
from .track import distance_to_centerline, find_lookahead_goal

def wrap_angle(a: float) -> float:
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

@dataclass
class PID:
    # PID regulator za kontrolu brzine vozila.
    Kp: float
    Ki: float
    Kd: float
    integ: float = 0.0
    prev_err: float = 0.0

    def reset(self):
        self.integ = 0.0
        self.prev_err = 0.0

    def update(self, err: float, dt: float) -> float:
        self.integ += err * dt
        deriv = (err - self.prev_err) / dt if dt > 0 else 0.0
        self.prev_err = err
        return self.Kp*err + self.Ki*self.integ + self.Kd*deriv

def pure_pursuit_delta(state: State, goal_xy: Tuple[float,float], Ld: float, wheelbase: float) -> float:
    """
    Izracunava ugao volana koristeci Pure Pursuit algoritam.
    Vozilo se usmerava ka lookahead tacki.
    """
    gx, gy = goal_xy
    dx = gx - state.x
    dy = gy - state.y
    # alpha je ugao izmedju pravca vozila i pravca ka ciljnoj tacki
    alpha = wrap_angle(math.atan2(dy, dx) - state.yaw)
    return math.atan2(2.0 * wheelbase * math.sin(alpha), max(1e-6, Ld))

def control_step(state: State, centerline, width: float, theta: dict, pid: PID, dt: float, wheelbase: float):
    cte, seg_idx, _ = distance_to_centerline((state.x, state.y), centerline)
    goal = find_lookahead_goal((state.x, state.y), centerline, seg_idx, theta["Ld"])
    delta = pure_pursuit_delta(state, goal, theta["Ld"], wheelbase)

    delta_max = math.radians(theta["delta_max_deg"])
    delta = max(-delta_max, min(delta_max, delta))

    v_err = theta["v_ref"] - state.v
    a = pid.update(v_err, dt)

    return a, delta, cte, goal
