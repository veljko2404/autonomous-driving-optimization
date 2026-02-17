import math
from dataclasses import dataclass
from .config import WHEELBASE, DT, A_MIN, A_MAX, V_MIN, V_MAX, DELTA_MIN, DELTA_MAX

"""
Kinematicki bicycle model vozila. Model opisuje translaciju i rotaciju vozila
bez razmatranja sila i klizanja (dovoljno za pracenje putanje).
"""

@dataclass
class State: # Stanje vozila u 2D prostoru.
    x: float # x pozicija vozila
    y: float # y pozicija vozila
    yaw: float # orijentacija (ugao u odnosu na x osu)
    v: float # brzina vozila

def clip(x, lo, hi):
    """
    Ogranicava vrednost x na interval [lo, hi]. Koristi se za
    postovanje fizickih ogranicenja aktuatora (volan, brzina, ubrzanje).
    """
    return lo if x < lo else hi if x > hi else x

def step(state: State, a: float, delta: float) -> State:
    """
    Izvrsava jedan diskretni korak simulacije vozila.
    Na osnovu trenutnog stanja i komandi (a, delta)
    racuna novo stanje koristeci bicycle model.
    """
    a = clip(a, A_MIN, A_MAX)
    delta = clip(delta, DELTA_MIN, DELTA_MAX)

    x, y, yaw, v = state.x, state.y, state.yaw, state.v

    x += v * math.cos(yaw) * DT
    y += v * math.sin(yaw) * DT
    yaw += (v / WHEELBASE) * math.tan(delta) * DT
    v = clip(v + a * DT, V_MIN, V_MAX)

    if yaw > math.pi: yaw -= 2*math.pi
    if yaw < -math.pi: yaw += 2*math.pi

    return State(x, y, yaw, v)
