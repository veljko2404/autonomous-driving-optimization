import json
import math
from typing import Dict, Any, Tuple, List
from .config import DT, MAX_TIME, WHEELBASE
from .dynamics import State, step
from .controllers import PID, control_step
from .track import finish_line_point

def save_best(path: str, best_J: float, best_theta: Dict[str,float], best_info: Dict[str,Any], history=None):
    """
    Cuvа najbolje pronadjene parametre i rezultate simulacije u JSON fajl.
    Koristi se za kasniji replay putanje i analizu performansi.
    """
    payload = { # U payload se cuvaju:
        "best_J": best_J, # - najbolja vrednost cost funkcije
        "best_theta": best_theta, # - parametri kontrolera
        "best_info": {k: v for k, v in best_info.items() if k not in ("path","goals")}, # - osnovne metrike simulacije
        "path": best_info.get("path", []), # - kompletna putanja za vizuelizaciju
        "goals": best_info.get("goals", []),
        "history": history or []
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_best(path: str):
    """
    Ucitava sacuvano najbolje resenje iz JSON fajla.
    Koristi se za replay simulacije bez ponovnog treniranja.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def rollout(track, theta: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
    """
    Izvrsava jednu kompletnu simulaciju vozila po stazi.
    Za zadate parametre kontrolera racuna cost funkciju J.

    -----------------------------------------------------

    Pocetno stanje vozila se postavlja na pocetak staze
    Orijentacija se racuna na osnovu prvih tacaka centerline
    """
    start = track.centerline[0]
    nxt = track.centerline[3]
    yaw0 = math.atan2(nxt[1]-start[1], nxt[0]-start[0])
    state = State(x=start[0], y=start[1], yaw=yaw0, v=0.0)

    pid = PID(theta["Kp"], theta["Ki"], theta["Kd"]) # PID regulator za kontrolu brzine
    pid.reset()

    t = 0.0 # vreme van staze
    offroad_steps = 0
    cte_sum = 0.0
    cte_count = 0 # bocna greska (CTE)
    steer_jerk = 0.0 # glatkoca upravljanja
    prev_delta = 0.0

    path_xy: List[Tuple[float,float]] = [(state.x, state.y)]
    goals: List[Tuple[float,float]] = []

    finish = finish_line_point(track.centerline)
    reached = False
    min_time_before_finish = 5.0

    """
    Glavna simulaciona petlja. U svakom koraku se racunaju komande
    upravljanja, azurira stanje vozila i beleze metrike
    """
    while t < MAX_TIME:
        a, delta, cte, goal = control_step(state, track.centerline, track.width, theta, pid, DT, WHEELBASE)
        goals.append(goal)

        cte_sum += cte
        cte_count += 1
        steer_jerk += abs(delta - prev_delta)
        prev_delta = delta

        if cte > (track.width * 0.5):
            offroad_steps += 1

        state = step(state, a, delta)
        path_xy.append((state.x, state.y))

        if t > min_time_before_finish and math.hypot(state.x - finish[0], state.y - finish[1]) < 3.0:
            reached = True
            t += DT
            break

        t += DT

    mean_cte = cte_sum / max(1, cte_count)
    offroad_time = offroad_steps * DT
    time_penalty = t if reached else (MAX_TIME + 10.0)
    smooth_penalty = steer_jerk

    J = ( # Ukupna cost funkcija J
        4.0 * mean_cte +
        40.0 * offroad_time +
        0.6 * time_penalty +
        0.8 * smooth_penalty
    )

    info = { # informacije koje vracamo za analizu i vizualizaciju
        "reached": reached,
        "t": float(t),
        "mean_cte": float(mean_cte),
        "offroad_time": float(offroad_time),
        "steer_jerk": float(smooth_penalty),
        "path": path_xy,
        "goals": goals,
    }
    return float(J), info
