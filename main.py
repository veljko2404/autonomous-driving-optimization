import argparse
from src.config import DEFAULT_THETA
from src.track import make_s_track, make_circle, make_random_track
from src.optimization.random_search import optimize as random_opt
from src.optimization.cd_golden_section import optimize as cd_opt
from src.optimization.nelder_mead import optimize as nm_opt
from src.optimization.cma_es import optimize as cma_opt
from src.utils import save_best, load_best, rollout
from src.visualization import run_replay

# Izbor staze na osnovu argumenta iz komandne linije
def get_track(name: str):
    """
    Vraca instancu staze na osnovu imena.
    Podrzane staze su S-oblik i kruzna staza.
    """
    name = name.lower()
    if name in ("s", "s_track", "strack"):
        return make_s_track()
    if name in ("circle", "c"):
        return make_circle()
    raise ValueError("Nepoznata staza. Koristi: s ili circle ili random")

# Omotac oko rollout funkcije da bi se fiksirala staza
def objective_fn(track):
    """
    Kreira objective funkciju J(theta) za zadatu stazu.
    Ova funkcija se prosledjuje optimizatorima.
    """
    def _obj(theta):
        return rollout(track, theta)
    return _obj

def main():
    """
    Glavna ulazna tacka programa.
    Omogucava pokretanje simulacije, optimizacije i replay-a
    preko argumenata komandne linije.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "random", "cd", "nm", "cma", "play"], default="sim")
    ap.add_argument("--track", default="s", help="s | circle | random")
    ap.add_argument("--control_points", type=int, default=8, help="Number of control points for random track")
    ap.add_argument("--radius", type=float, default=35.0, help="Average radius/length scale for random track")
    ap.add_argument("--jitter", type=float, default=10.0, help="Jitter magnitude for random track")
    ap.add_argument("--closed", type=lambda x: x.lower() in ("1", "true", "yes"), default=True, help="Whether random track is closed (True/False)")

    # Prilagodljivi parametri algoritama optimizacije
    ap.add_argument("--iters", type=int, default=200, help="random search iterations")
    ap.add_argument("--lr_alg", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=4, help="coordinate descent cycles")
    ap.add_argument("--gs_iters", type=int, default=20, help="golden section iterations per coordinate")
    ap.add_argument("--cma_iters", type=int, default=70, help="CMA-ES iterations")
    ap.add_argument("--cma_sigma", type=float, default=0.3, help="globalni step-size")

    ap.add_argument("--save", default="best.json")
    ap.add_argument("--load", default="best.json")
    args = ap.parse_args()

    # Kreiranje staze i objective funkcije
    # Ako je korisnik izabrao random stazu, iskoristi dodatne parametre; inace pozovi get_track
    if args.track.lower() in ("random", "rand"):
        track = make_random_track(
            n=420,
            scale=1.0,
            seed=args.seed,
            control_points=args.control_points,
            radius=args.radius,
            jitter=args.jitter,
            closed=args.closed
        )
    else:
        track = get_track(args.track)

    obj = objective_fn(track)

    # Rezim: samo simulacija sa default parametrima
    if args.mode == "sim":
        J, info = obj(DEFAULT_THETA)
        print("DEFAULT THETA:", DEFAULT_THETA)
        print("J:", J)
        print({k: info[k] for k in ("reached", "t", "mean_cte", "offroad_time", "steer_jerk")})

        # Cuvanje rezultata radi kasnijeg replay-a
        save_best(
            args.save,
            J,
            DEFAULT_THETA,
            info,
            history=[{"stage": "sim", "J": J, "reached": info["reached"]}]
        )
        print(f"Saved: {args.save}")

    # Rezim: Random Search optimizacija
    elif args.mode == "random":
        best_J, best_theta, best_info, history = random_opt(
            obj, iters=args.iters, seed=args.seed
        )
        print("BEST J:", best_J)
        print("BEST THETA:", best_theta)
        print({k: best_info[k] for k in ("reached", "t", "mean_cte", "offroad_time", "steer_jerk")})

        save_best(args.save, best_J, best_theta, best_info, history=history)
        print(f"Saved: {args.save}")

    # Rezim: Coordinate Descent + Golden Section optimizacija
    elif args.mode == "cd":
        best_J, best_theta, best_info, history = cd_opt(
            obj, DEFAULT_THETA, cycles=args.cycles, gs_iters=args.gs_iters
        )
        print("BEST J:", best_J)
        print("BEST THETA:", best_theta)
        print({k: best_info[k] for k in ("reached", "t", "mean_cte", "offroad_time", "steer_jerk")})

        save_best(args.save, best_J, best_theta, best_info, history=history)
        print(f"Saved: {args.save}")

    #Rezim: Nelder-Mead optimizacija
    elif args.mode == "nm":
        best_J, best_theta, best_info, history = nm_opt(
            obj, DEFAULT_THETA, max_iters=200
        )

        print("BEST J:", best_J)
        print("BEST THETA:", best_theta)
        print({k: best_info[k] for k in ("reached", "t", "mean_cte", "offroad_time", "steer_jerk")})

        save_best(args.save, best_J, best_theta, best_info, history=history)
        print(f"Saved: {args.save}")

    # Rezim: CMA-ES optimizacija (Covariance Matrix Adaptation Evolution Strategy)
    elif args.mode == "cma":
        best_J, best_theta, best_info, history = cma_opt(
            obj,
            DEFAULT_THETA,
            iters=args.cma_iters,
            seed=args.seed,
            sigma=args.cma_sigma
        )

        print("BEST J:", best_J)
        print("BEST THETA:", best_theta)
        print({k: best_info[k] for k in ("reached", "t", "mean_cte", "offroad_time", "steer_jerk")})

        save_best(args.save, best_J, best_theta, best_info, history=history)
        print(f"Saved: {args.save}")

    # Rezim: Replay sacuvanog najboljeg resenja
    elif args.mode == "play":
        data = load_best(args.load)

        # Priprema meta-podataka za prikaz
        meta = {}
        meta.update(data.get("best_info", {}))
        meta["best_J"] = data.get("best_J")

        path = data.get("path", [])
        goals = data.get("goals", None)

        run_replay(track, path, goals, meta=meta)

if __name__ == "__main__":
    # Omogucava da se skripta pokrene direktno
    main()

