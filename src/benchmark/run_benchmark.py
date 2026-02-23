import time          # merenje vremena izvršavanja algoritama
import json          # čuvanje rezultata u JSON formatu
import os            # rad sa fajl sistemom (kreiranje foldera, putanje)
from math import ceil, floor
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# Konfiguracija početnih parametara sistema (npr. parametri kontrolera)
from src.config import DEFAULT_THETA

# Generisanje staze
from src.track import make_s_track

# rollout simulira vožnju i vraća:
#   J    -> vrednost funkcije cilja (npr. ukupna greška)
#   info -> dodatne informacije (npr. da li je staza završena)
from src.utils import rollout, save_best

# Uvoz različitih optimizacionih algoritama
from src.optimization.random_search import optimize as random_opt
from src.optimization.cd_golden_section import optimize as cd_opt
from src.optimization.nelder_mead import optimize as nm_opt
from src.optimization.cma_es import optimize as cma_opt

#  Wrapper za brojanje evaluacija funkcije cilja

class CountingObjective:
    """
    Ovaj wrapper služi za:

    1) brojanje koliko puta je funkcija cilja evaluirana
    2) čuvanje svih J vrednosti po redosledu evaluacija
    3) omogućavanje generisanja "best-so-far" krivih

    Ovo je ključno za benchmark jer:
    - različiti algoritmi imaju različit broj evaluacija po iteraciji
    - jedini fer način poređenja je po broju evaluacija
    """

    def __init__(self, obj_fn):
        self.obj_fn = obj_fn
        self.eval_count = 0      # broj poziva funkcije cilja
        self.J_list = []         # lista svih J vrednosti
        self.info_list = []      # lista svih info objekata

    def __call__(self, theta_dict):
        """
        Kada algoritam pozove objective,
        ovaj wrapper presreće poziv.
        """
        J, info = self.obj_fn(theta_dict)

        self.eval_count += 1
        self.J_list.append(float(J))
        self.info_list.append(info)

        return J, info

    def reset(self):
        """
        Resetuje stanje pre pokretanja sledećeg algoritma.
        """
        self.eval_count = 0
        self.J_list = []
        self.info_list = []

#  Kreiranje objective funkcije za konkretnu stazu

def make_objective_for_track(track):
    """
    Vraća funkciju obj(theta) koja:
    - prima parametre kontrolera
    - izvršava simulaciju (rollout)
    - vraća (J, info)

    Ovo omogućava algoritmima da rade generički,
    bez znanja o detaljima simulacije.
    """
    def obj(theta):
        return rollout(track, theta)
    return obj

#  Benchmark procedura

def run_benchmark(track_name="s", eval_budget=480, cma_seed=0, gs_iters_default=20):
    """
    Pokreće više optimizacionih algoritama nad istim problemom
    i poredi ih pod istim budžetom evaluacija.

    eval_budget = maksimalan broj dozvoljenih poziva objective funkcije.

    Ključna ideja:
    Različiti algoritmi imaju različitu strukturu iteracija,
    zato budžet mapiramo na njihove interne parametre.
    """

    # ------------------------------------------------------
    # Priprema problema
    # ------------------------------------------------------
    track = make_s_track() if track_name.lower().startswith("s") else None

    obj = make_objective_for_track(track)
    cnt_obj = CountingObjective(obj)

    # Broj parametara koje optimizujemo
    n = len(DEFAULT_THETA)

    # CMA-ES koristi populaciju λ = 4 + 3 log(n)
    lambda_cma = 4 + int(3 * np.log(n))

    # ------------------------------------------------------
    # Mapiranje budžeta evaluacija na parametre algoritama
    # ------------------------------------------------------
    params = {}

    # Random Search:
    # Jedna iteracija = jedna evaluacija
    params['random'] = {'iters': eval_budget}

    # CMA-ES:
    # Jedna generacija = lambda_cma evaluacija
    params['cma'] = {
        'iters': max(1, int(round(eval_budget / lambda_cma)))
    }

    # Nelder-Mead:
    # Jedna iteracija ~ n+1 evaluacija
    params['nm'] = {
        'max_iters': max(1, int(floor(eval_budget / (n + 1))))
    }

    # Coordinate Descent:
    # Jedan ciklus ≈ n * gs_iters evaluacija
    est_cycle_evals = n * gs_iters_default
    params['cd'] = {
        'cycles': max(1, int(floor(eval_budget / max(1, est_cycle_evals)))),
        'gs_iters': gs_iters_default
    }

    print("Benchmark settings (approx.):")
    print(f" problem dim n = {n}")
    print(f" eval_budget = {eval_budget}")
    print(" mapped params:", params)

    results = {}
    curves = {}

    # RANDOM SEARCH
    
    cnt_obj.reset()
    start = time.time()

    best_J, best_theta, best_info, history = random_opt(
        cnt_obj,
        iters=params['random']['iters'],
        seed=0
    )

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

    # COORDINATE DESCENT
    cnt_obj.reset()
    start = time.time()

    best_J, best_theta, best_info, history = cd_opt(
        cnt_obj,
        DEFAULT_THETA,
        cycles=params['cd']['cycles'],
        gs_iters=params['cd']['gs_iters']
    )

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

    # NELDER-MEAD
    cnt_obj.reset()
    start = time.time()

    best_J, best_theta, best_info, history = nm_opt(
        cnt_obj,
        DEFAULT_THETA,
        max_iters=params['nm']['max_iters']
    )

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

    # CMA-ES
    cnt_obj.reset()
    start = time.time()

    best_J, best_theta, best_info, history = cma_opt(
        cnt_obj,
        DEFAULT_THETA,
        iters=params['cma']['iters'],
        seed=cma_seed
    )

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

    # Post-processing: best-so-far krive
    best_so_far = {}

    for k, js in curves.items():
        best = []
        cur_min = float('inf')

        for j in js:
            if j < cur_min:
                cur_min = j
            best.append(cur_min)

        best_so_far[k] = best

    # Čuvanje rezultata
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")

    out_json = os.path.join(out_dir, f"benchmark_{ts}.json")

    with open(out_json, "w") as f:
        json.dump({
            'params': params,
            'results': results,
            'curves': curves
        }, f, indent=2)

    print("Saved numeric results to", out_json)

    # Plotovanje krivih konvergencije
    plt.figure()

    max_len = max(len(v) for v in best_so_far.values())

    for name, arr in best_so_far.items():
        if len(arr) < max_len:
            arr = arr + [arr[-1]] * (max_len - len(arr))

        plt.plot(
            np.arange(1, len(arr) + 1),
            arr,
            label=f"{name} (evals={len(curves[name])})"
        )

    plt.xlabel("Function evaluations")
    plt.ylabel("Best J so far (lower = better)")
    plt.title("Benchmark: convergence vs evaluations")
    plt.legend()
    plt.grid(True)

    out_png_full = os.path.join(out_dir, f"benchmark_{ts}_full.png")

    plt.tight_layout()
    plt.savefig(out_png_full, dpi=150)
    plt.close()

    print("Saved full plot to", out_png_full)

    # Plot 2: y-limit zoom (10 <= J <= 14)
    plt.figure()

    for name, arr in best_so_far.items():
        plt.plot(
            np.arange(1, len(arr) + 1),
            arr,
            label=f"{name} (evals={len(curves[name])})"
        )

    plt.xlabel("Function evaluations")
    plt.ylabel("Best J so far (lower = better)")
    plt.title("Benchmark (Y limited to [10, 14])")
    plt.legend()
    plt.grid(True)

    # SAMO ograničavamo y osu
    plt.ylim(10, 14)

    out_png_zoom_ylim = os.path.join(out_dir, f"benchmark_{ts}_ylim_10_14.png")

    plt.tight_layout()
    plt.savefig(out_png_zoom_ylim, dpi=150)
    plt.close()

    print("Saved ylim[10,20] plot to", out_png_zoom_ylim)

    return results, best_so_far, out_json, out_png_full


# CLI pokretanje iz terminala
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