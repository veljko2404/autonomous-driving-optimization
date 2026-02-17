import argparse
from src.config import DEFAULT_THETA
from src.track import make_s_track, make_circle
from src.optim.random_search import optimize as random_opt
from src.optim.coordinate_descent import optimize as cd_opt
from src.utils import save_best, load_best, rollout
from src.viz import run_replay

def get_track(name: str):
    name = name.lower()
    if name in ("s", "s_track", "strack"):
        return make_s_track()
    if name in ("circle", "c"):
        return make_circle()
    raise ValueError("Nepoznata staza. Koristi: s ili circle")

def objective_fn(track):
    def _obj(theta):
        return rollout(track, theta)
    return _obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim","random","cd","play"], default="sim")
    ap.add_argument("--track", default="s", help="s | circle")
    ap.add_argument("--iters", type=int, default=200, help="random search iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=4, help="coordinate descent cycles")
    ap.add_argument("--gs_iters", type=int, default=20, help="golden section iterations per coordinate")
    ap.add_argument("--save", default="best.json")
    ap.add_argument("--load", default="best.json")
    args = ap.parse_args()

    track = get_track(args.track)
    obj = objective_fn(track)

    if args.mode == "sim":
        J, info = obj(DEFAULT_THETA)
        print("DEFAULT THETA:", DEFAULT_THETA)
        print("J:", J)
        print({k: info[k] for k in ("reached","t","mean_cte","offroad_time","steer_jerk")})
        save_best(args.save, J, DEFAULT_THETA, info, history=[{"stage":"sim","J":J,"reached":info["reached"]}])
        print(f"Saved: {args.save}")

    elif args.mode == "random":
        best_J, best_theta, best_info, history = random_opt(obj, iters=args.iters, seed=args.seed)
        print("BEST J:", best_J)
        print("BEST THETA:", best_theta)
        print({k: best_info[k] for k in ("reached","t","mean_cte","offroad_time","steer_jerk")})
        save_best(args.save, best_J, best_theta, best_info, history=history)
        print(f"Saved: {args.save}")

    elif args.mode == "cd":
        best_J, best_theta, best_info, history = cd_opt(obj, DEFAULT_THETA, cycles=args.cycles, gs_iters=args.gs_iters)
        print("BEST J:", best_J)
        print("BEST THETA:", best_theta)
        print({k: best_info[k] for k in ("reached","t","mean_cte","offroad_time","steer_jerk")})
        save_best(args.save, best_J, best_theta, best_info, history=history)
        print(f"Saved: {args.save}")

    elif args.mode == "play":
        data = load_best(args.load)
        meta = {}
        meta.update(data.get("best_info", {}))
        meta["best_J"] = data.get("best_J")
        path = data.get("path", [])
        goals = data.get("goals", None)
        run_replay(track, path, goals, meta=meta)

if __name__ == "__main__":
    main()
