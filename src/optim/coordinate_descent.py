from typing import Dict, Tuple, Any, Callable
from ..config import BOUNDS

def golden_section_minimize(f: Callable[[float], float], a: float, b: float, iters: int = 20):
    phi = (1 + 5**0.5) / 2
    invphi = 1 / phi

    c = b - (b - a) * invphi
    d = a + (b - a) * invphi
    fc = f(c)
    fd = f(d)

    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * invphi
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * invphi
            fd = f(d)

    xbest = (a + b) / 2
    fbest = f(xbest)
    return xbest, fbest

def optimize(objective_fn: Callable[[Dict[str,float]], Tuple[float, Any]],
             theta0: Dict[str,float],
             cycles: int = 4,
             gs_iters: int = 20):
    theta = dict(theta0)
    best_J, best_info = objective_fn(theta)
    history = [{"stage": "init", "J": best_J, "reached": bool(best_info.get("reached", False))}]

    keys = list(BOUNDS.keys())

    for c in range(1, cycles+1):
        for k in keys:
            lo, hi = BOUNDS[k]

            def f1(x):
                th = dict(theta)
                th[k] = x
                J, _ = objective_fn(th)
                return J

            xbest, _ = golden_section_minimize(f1, lo, hi, iters=gs_iters)
            theta[k] = xbest
            best_J, best_info = objective_fn(theta)
            history.append({"stage": f"cycle{c}:{k}", "J": best_J, "reached": bool(best_info.get("reached", False))})

    return best_J, theta, best_info, history
