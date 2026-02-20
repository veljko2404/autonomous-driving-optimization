import time
import json
import os
from math import ceil, floor
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from src.config import DEFAULT_THETA
from src.track import make_s_track
from src.utils import rollout, save_best  # rollout vraća J, info
from src.optimization.random_search import optimize as random_opt
from src.optimization.cd_golden_section import optimize as cd_opt
from src.optimization.nelder_mead import optimize as nm_opt
from src.optimization.cma_es import optimize as cma_opt

# ---------- Wrapper za brojanje evaluacija i prikupljanje J vrednosti ----------
class CountingObjective:
    """
    Omota objective funkciju tako da:
    - broji svaku evaluaciju
    - pamti listu svih J vrednosti (redosled poziva)
    - omogućava pozivanje originalnog obj(dict)->(J,info)
    """
    def __init__(self, obj_fn):
        self.obj_fn = obj_fn
        self.eval_count = 0
        self.J_list = []
        self.info_list = []

    def __call__(self, theta_dict):
        J, info = self.obj_fn(theta_dict)
        self.eval_count += 1
        self.J_list.append(float(J))
        self.info_list.append(info)
        return J, info

    def reset(self):
        self.eval_count = 0
        self.J_list = []
        self.info_list = []

# ---------- Helper: build objective for a track ----------
def make_objective_for_track(track):
    def obj(theta):
        return rollout(track, theta)
    return obj

# ---------- Benchmark procedure ----------
def run_benchmark(track_name="s", eval_budget=480, cma_seed=0, gs_iters_default=20):
    track = make_s_track() if track_name.lower().startswith("s") else None
    obj = make_objective_for_track(track)
    cnt_obj = CountingObjective(obj)

    # Problem dimension
    n = len(DEFAULT_THETA)
    # CMA population
    lambda_cma = 4 + int(3 * np.log(n))
    # Map budget -> per-algo iteration params (approximate)
    params = {}

    # Random Search: one evaluation per iteration
    params['random'] = {'iters': eval_budget}

    # CMA-ES: evaluations per iter = lambda_cma
    params['cma'] = {'iters': max(1, int(round(eval_budget / lambda_cma)))}

    # Nelder-Mead: NM evaluates (n+1) points per NM-iteration (initial simplex included)
    params['nm'] = {'max_iters': max(1, int(floor(eval_budget / (n + 1))))}

    # Coordinate Descent + Golden Section: per cycle ~ n * gs_iters_default evaluations (approx)
    est_cycle_evals = n * gs_iters_default
    params['cd'] = {'cycles': max(1, int(floor(eval_budget / max(1, est_cycle_evals)))),
                    'gs_iters': gs_iters_default}

    print("Benchmark settings (approx.):")
    print(f" problem dim n = {n}")
    print(f" eval_budget = {eval_budget}")
    print(" mapped params:", params)

    results = {}
    curves = {}

    # --- Random Search ---
    cnt_obj.reset()
    start = time.time()
    best_J, best_theta, best_info, history = random_opt(cnt_obj, iters=params['random']['iters'], seed=0)
    t = time.time() - start
    results['random'] = {
        'best_J': float(best_J),
        'best_theta': best_theta,
        'best_info': best_info,
        'evals': cnt_obj.eval_count,
        'time_s': t
    }
    curves['random'] = cnt_obj.J_list.copy()
    print(f"[random] evals={cnt_obj.eval_count} time={t:.2f}s best_J={best_J}")

    # --- Coordinate Descent + Golden Section ---
    cnt_obj.reset()
    start = time.time()
    # cd_opt signature: optimize(obj, x0, cycles=..., gs_iters=...)
    best_J, best_theta, best_info, history = cd_opt(cnt_obj, DEFAULT_THETA,
                                                   cycles=params['cd']['cycles'],
                                                   gs_iters=params['cd']['gs_iters'])
    t = time.time() - start
    results['cd'] = {
        'best_J': float(best_J),
        'best_theta': best_theta,
        'best_info': best_info,
        'evals': cnt_obj.eval_count,
        'time_s': t
    }
    curves['cd'] = cnt_obj.J_list.copy()
    print(f"[cd] evals={cnt_obj.eval_count} time={t:.2f}s best_J={best_J}")

    # --- Nelder-Mead ---
    cnt_obj.reset()
    start = time.time()
    best_J, best_theta, best_info, history = nm_opt(cnt_obj, DEFAULT_THETA, max_iters=params['nm']['max_iters'])
    t = time.time() - start
    results['nm'] = {
        'best_J': float(best_J),
        'best_theta': best_theta,
        'best_info': best_info,
        'evals': cnt_obj.eval_count,
        'time_s': t
    }
    curves['nm'] = cnt_obj.J_list.copy()
    print(f"[nm] evals={cnt_obj.eval_count} time={t:.2f}s best_J={best_J}")

    # --- CMA-ES ---
    cnt_obj.reset()
    start = time.time()
    best_J, best_theta, best_info, history = cma_opt(cnt_obj, DEFAULT_THETA, iters=params['cma']['iters'], seed=cma_seed)
    t = time.time() - start
    results['cma'] = {
        'best_J': float(best_J),
        'best_theta': best_theta,
        'best_info': best_info,
        'evals': cnt_obj.eval_count,
        'time_s': t
    }
    curves['cma'] = cnt_obj.J_list.copy()
    print(f"[cma] evals={cnt_obj.eval_count} time={t:.2f}s best_J={best_J}")

    # ---------- Postprocess: build best-so-far curve per evaluation ----------
    # For plotting we want monotonic 'best-so-far' arrays vs eval index
    best_so_far = {}
    for k, js in curves.items():
        best = []
        cur_min = float('inf')
        for j in js:
            if j < cur_min: cur_min = j
            best.append(cur_min)
        best_so_far[k] = best

    # Save numeric results JSON
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    out_json = os.path.join(out_dir, f"benchmark_{ts}.json")
    with open(out_json, "w") as f:
        json.dump({'params': params, 'results': results, 'curves': curves}, f, indent=2)
    print("Saved numeric results to", out_json)

    # ---------- Create CSV summary table ----------
    import csv
    csv_path = os.path.join(out_dir, f"benchmark_{ts}_summary.csv")
    with open(csv_path, "w", newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(["algorithm", "final_best_J", "evals", "time_s", "reached"])
        for name, meta in results.items():
            reached_val = meta.get('best_info', {}).get('reached', None)
            writer.writerow([name, meta['best_J'], meta['evals'], meta['time_s'], reached_val])
    print("Saved CSV summary to", csv_path)

    # ---------- Plot 1: zoom (first N evals) ----------
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available:", e)
        return results, best_so_far, out_json, None

    zoom_cut = 150  # default; can be exposed as CLI arg if desired
    plt.figure(figsize=(9, 5))
    max_len = max(len(v) for v in best_so_far.values())
    for name, arr in best_so_far.items():
        arr_cut = arr[:zoom_cut]
        if len(arr_cut) == 0:
            continue
        plt.plot(np.arange(1, len(arr_cut) + 1), arr_cut, label=f"{name} (evals={len(curves[name])})")
    plt.xlabel("Function evaluations")
    plt.ylabel("Best J so far (lower = better)")
    plt.title(f"Benchmark (zoom first {zoom_cut} evals)")
    plt.legend()
    plt.grid(True)
    out_png_zoom = os.path.join(out_dir, f"benchmark_{ts}_zoom.png")
    plt.tight_layout()
    plt.savefig(out_png_zoom, dpi=150)
    plt.close()
    print("Saved zoom plot to", out_png_zoom)

    # ---------- Plot 2: full budget ----------
    plt.figure(figsize=(9, 5))
    max_len = max(len(v) for v in best_so_far.values())
    for name, arr in best_so_far.items():
        # pad last value to match lengths for clean plotting
        if len(arr) < max_len:
            arr = arr + [arr[-1]] * (max_len - len(arr))
        plt.plot(np.arange(1, len(arr) + 1), arr, label=f"{name} (evals={len(curves[name])})")
    plt.xlabel("Function evaluations")
    plt.ylabel("Best J so far (lower = better)")
    plt.title("Benchmark: convergence vs evaluations (full)")
    plt.legend()
    plt.grid(True)
    out_png_full = os.path.join(out_dir, f"benchmark_{ts}_full.png")
    plt.tight_layout()
    plt.savefig(out_png_full, dpi=150)
    plt.close()
    print("Saved full plot to", out_png_full)

    # ---------- Print a nice terminal table ----------
    try:
        from tabulate import tabulate
        table = []
        for name, meta in results.items():
            reached_val = meta.get('best_info', {}).get('reached', None)
            table.append([name, meta['best_J'], meta['evals'], f"{meta['time_s']:.2f}s", reached_val])
        print("\nSummary:")
        print(tabulate(table, headers=["algo", "final_best_J", "evals", "time", "reached"], tablefmt="github"))
    except Exception:
        # fallback plain print
        print("\nSummary (plain):")
        for name, meta in results.items():
            print(
                f"{name}: final_best_J={meta['best_J']}, evals={meta['evals']}, time={meta['time_s']:.2f}s, reached={meta.get('best_info', {}).get('reached', None)}")

    return results, best_so_far, out_json, out_png_full

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="s", choices=["s", "circle"])
    ap.add_argument("--budget", type=int, default=480)
    ap.add_argument("--cma_seed", type=int, default=0)
    args = ap.parse_args()

    run_benchmark(
        track_name=args.track,
        eval_budget=args.budget,
        cma_seed=args.cma_seed
    )