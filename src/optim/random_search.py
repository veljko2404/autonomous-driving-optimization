import random
from typing import Dict
from ..config import BOUNDS

def sample_theta(rng: random.Random) -> Dict[str, float]:
    theta = {}
    for k, (lo, hi) in BOUNDS.items():
        theta[k] = lo + (hi-lo) * rng.random()
    return theta

def optimize(objective_fn, iters: int = 200, seed: int = 0):
    rng = random.Random(seed)
    best_theta = None
    best_J = float("inf")
    best_info = None
    history = []

    for i in range(1, iters+1):
        theta = sample_theta(rng)
        J, info = objective_fn(theta)
        history.append({"iter": i, "J": J, "reached": bool(info.get("reached", False))})
        if J < best_J:
            best_J, best_theta, best_info = J, theta, info

    return best_J, best_theta, best_info, history
