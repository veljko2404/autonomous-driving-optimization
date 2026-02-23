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
    Omota objective funkciju kako bi:
      - brojao koliko puta je funkcija pozvana (eval_count),
      - čuvao listu svih dobijenih J vrednosti (J_list) u redosledu poziva,
      - čuvao pridružene info vrednosti (info_list),
      - prosleđivao pozivanje originalnoj obj_fn funkciji.

    Korisno za benchmark: tako dobijemo tačan broj evaluacija i krivulje
    "best so far" po pozivima funkcije.
    """
    def __init__(self, obj_fn):
        self.obj_fn = obj_fn
        self.eval_count = 0
        self.J_list = []
        self.info_list = []

    def __call__(self, theta_dict):
        # Pozovi originalnu funkciju cilja i zabeleži rezultat
        J, info = self.obj_fn(theta_dict)
        self.eval_count += 1
        self.J_list.append(float(J))
        self.info_list.append(info)
        return J, info

    def reset(self):
        # Resetuje brojilo i liste — koristan pre pokretanja svake metode
        self.eval_count = 0
        self.J_list = []
        self.info_list = []


# ---------- Helper: build objective for a track ----------
def make_objective_for_track(track):
    """
    Pravi funkciju obj(theta) koja enkapsulira rollout nad konkretnim stazom.
    rollout(track, theta) vraća (J, info) — tako možemo proslediti tu funkciju
    u optimizacione algoritme koji očekuju obj(dict)->(J,info).
    """
    def obj(theta):
        return rollout(track, theta)
    return obj


# ---------- Benchmark procedure ----------
def run_benchmark(track_name="s", eval_budget=480, cma_seed=0, gs_iters_default=20):
    """
    Pokreće benchmark nekoliko optimizatora nad istim problemom i
    mapira ukupni eval_budget na parametre pojedinih algoritama.

    Arguments:
      track_name: "s" ili "circle" (trenutno podržano samo "s" kroz make_s_track)
      eval_budget: ukupni broj dozvoljenih evaluacija funkcije cilja
      cma_seed: seed za reproducibilnost CMA-ES-a
      gs_iters_default: broj iteracija za Golden Section u CD+GS algoritmu
    Returns:
      results: dict sa metrikama (finalni best_J, evaluacije, vreme ...)
      best_so_far: dict sa listama "najbolje do sada" po evaluaciji (non-increasing)
      out_json: putanja do sačuvanog JSON fajla sa numeričkim rezultatima
      out_png_full: putanja do glavnog (full) PNG plot-a
    """

    # --- priprema problema i objective ---
    track = make_s_track() if track_name.lower().startswith("s") else None
    # Ako budu dodate druge staze, proširiti grananje iznad.
    obj = make_objective_for_track(track)
    cnt_obj = CountingObjective(obj)  # omotač za brojanje i cuvanje J-ova

    # Dimenzija problema (broj parametara iz DEFAULT_THETA dict-a)
    n = len(DEFAULT_THETA)

    # CMA populacija (isti izraz kao u implementaciji CMA-ES)
    lambda_cma = 4 + int(3 * np.log(n))

    # Mapiranje ukupnog budžeta na parametre pojedinih algoritama.
    # Cilj: približno fer poređenje (svaki algoritam troši ~eval_budget evaluacija)
    params = {}

    # Random Search: svaki iter poziva obj jednom -> iters = eval_budget
    params['random'] = {'iters': eval_budget}

    # CMA-ES: svaka iteracija (generacija) koristi lambda_cma evaluacija
    # broj generacija = eval_budget / lambda_cma
    params['cma'] = {'iters': max(1, int(round(eval_budget / lambda_cma)))}

    # Nelder-Mead: NM interno evaluira n+1 tačaka po iteraciji
    params['nm'] = {'max_iters': max(1, int(floor(eval_budget / (n + 1))))}

    # Coordinate Descent + Golden Section: procena da je jedan ciklus ~
    # n * gs_iters_default evaluacija (po koordinati gs_iters)
    est_cycle_evals = n * gs_iters_default
    params['cd'] = {
        'cycles': max(1, int(floor(eval_budget / max(1, est_cycle_evals)))),
        'gs_iters': gs_iters_default
    }

    # Ispis za korisnika (kratka provera mapiranja budžeta)
    print("Benchmark settings (approx.):")
    print(f" problem dim n = {n}")
    print(f" eval_budget = {eval_budget}")
    print(" mapped params:", params)

    results = {}  # meta-podaci o svakom algoritmu (vraćamo ovo)
    curves = {}   # sirovi nizovi J po pozivu za svaki algoritam

    # --- Random Search ---
    cnt_obj.reset()  # reset brojila pre svakog algoritma
    start = time.time()
    # random_opt signature: optimize(obj, iters=..., seed=...)
    best_J, best_theta, best_info, history = random_opt(cnt_obj, iters=params['random']['iters'], seed=0)
    t = time.time() - start

    # Skladištimo metrike za kasniju upotrebu/ispis
    results['random'] = {
        'best_J': float(best_J),
        'best_theta': best_theta,
        'best_info': best_info,
        'evals': cnt_obj.eval_count,
        'time_s': t
    }
    # Sačuvaj listu svih J vrednosti koje je CountingObjective zabeležio
    curves['random'] = cnt_obj.J_list.copy()
    print(f"[random] evals={cnt_obj.eval_count} time={t:.2f}s best_J={best_J}")

    # --- Coordinate Descent + Golden Section ---
    cnt_obj.reset()
    start = time.time()
    # cd_opt signature: optimize(obj, x0, cycles=..., gs_iters=...)
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

    # --- Nelder-Mead ---
    cnt_obj.reset()
    start = time.time()
    # nm_opt signature: optimize(obj, x0, max_iters=...)
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
    # cma_opt signature: optimize(obj, x0, iters=..., seed=...)
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
    # Mape: za svaki algoritam želimo niz koji na poziciji k sadrži vrednost
    # najboljeg J-a koji je ikada do tada iskazan — to olakšava poređenje.
    best_so_far = {}
    for k, js in curves.items():
        best = []
        cur_min = float('inf')
        for j in js:
            # formiramo monotono ne-povećavajuću listu: best_so_far
            if j < cur_min:
                cur_min = j
            best.append(cur_min)
        best_so_far[k] = best

    # ---------- Sačuvaj numeričke rezultate u JSON ----------
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_json = os.path.join(out_dir, f"benchmark_{ts}.json")
    with open(out_json, "w") as f:
        # Čuvamo parametre, rezultate i sirove krivulje J-ova
        json.dump({'params': params, 'results': results, 'curves': curves}, f, indent=2)
    print("Saved numeric results to", out_json)

    # ---------- Create CSV summary table ----------
    import csv
    csv_path = os.path.join(out_dir, f"benchmark_{ts}_summary.csv")
    with open(csv_path, "w", newline='') as cf:
        writer = csv.writer(cf)
        # zaglavlje CSV tabele
        writer.writerow(["algorithm", "final_best_J", "evals", "time_s", "reached"])
        for name, meta in results.items():
            # očekujemo da best_info eventualno sadrži ključeve poput 'reached'
            reached_val = meta.get('best_info', {}).get('reached', None)
            writer.writerow([name, meta['best_J'], meta['evals'], meta['time_s'], reached_val])
    print("Saved CSV summary to", csv_path)

    # ---------- Plot 1: zoom (by J-range) ----------
    # Opcija: fokus na vertikalni opseg J (zoom_j_range), i opciono fokus na x
    zoom_j_range = (0.0, 20.0)  # (ymin, ymax)  — postavi na (None, None) da isključiš clipping
    zoom_x_focus = True  # ako True, x osa se skraćuje samo na indeks gde kriva ulazi u J window

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available:", e)
        # Vraćamo rezultate, i None umesto slike
        return results, best_so_far, out_json, None

    # Kreiramo figure i plotujemo samo delove krivih koji spadaju u J-window
    plt.figure()
    plotted_any = False
    ymin, ymax = zoom_j_range

    for name, arr in best_so_far.items():
        if len(arr) == 0:
            # nema evaluacija za ovaj algoritam
            continue
        arr_np = np.array(arr)  # best-so-far (non-increasing sequence)
        x = np.arange(1, len(arr_np) + 1)

        # Odlučujemo koji deo krive crtamo:
        if ymin is None and ymax is None:
            # bez J clipping-a: plotujemo sve
            plot_x, plot_y = x, arr_np

        elif zoom_x_focus:
            # Fokusiramo x na interval gde kriva ulazi u J-window
            # idx_start: prvi indeks gde arr <= ymax (ako je ymax dat)
            if ymax is not None:
                idx_start = np.argmax(arr_np <= ymax) if np.any(arr_np <= ymax) else None
            else:
                idx_start = 0

            # idx_end: poslednji indeks gde arr >= ymin (ako je ymin dat).
            # arr je ne-povećavajuća, pa maske rade jednostavno.
            if ymin is not None:
                mask_ge = arr_np >= ymin
                idx_end = np.where(mask_ge)[0][-1] if np.any(mask_ge) else None
            else:
                idx_end = len(arr_np) - 1

            # Ako nema preseka intervala, preskačemo crtanje za ovaj algoritam
            if idx_start is None or idx_end is None or idx_start > idx_end:
                continue

            plot_x = x[idx_start: idx_end + 1]
            plot_y = arr_np[idx_start: idx_end + 1]
        else:
            # crtaj celu krivu, kasnije klipujemo y-osu
            plot_x, plot_y = x, arr_np

        plt.plot(plot_x, plot_y, label=f"{name} (evals={len(curves.get(name, []))})")
        plotted_any = True

    if not plotted_any:
        print("No curves entered the requested J window; skipping zoom plot.")
    else:
        # dodaj oznake i legendu
        plt.xlabel("Function evaluations")
        plt.ylabel("Best J so far (lower = better)")
        plt.title(f"Benchmark (zoom by J-range {ymin}..{ymax})")
        plt.legend()
        plt.grid(True)

        # Ako nismo fokusirali x, podešavamo y-limits po zadata dva broja
        if not zoom_x_focus:
            ylo = -np.inf if ymin is None else ymin
            yhi = np.inf if ymax is None else ymax
            plt.ylim(ylo, yhi)

        out_png_zoom = os.path.join(out_dir, f"benchmark_{ts}_zoom_byJ_{ymin}_{ymax}.png")
        plt.tight_layout()
        plt.savefig(out_png_zoom, dpi=150)
        plt.close()
        print("Saved zoom (by J-range) plot to", out_png_zoom)

    # ---------- Plot 2: full budget ----------
    # Plotujemo kompletne best-so-far krive, sve do maksimalnog broja evaluacija
    plt.figure()
    # odredi maksimalnu dužinu; neke krive mogu biti kraće — padujemo
    max_len = max(len(v) for v in best_so_far.values()) if best_so_far else 0
    for name, arr in best_so_far.items():
        if len(arr) < max_len:
            # padujemo poslednjom vrednošću — to čini linije poravnate i lepše za poređenje
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
        # Ako je tabulate instaliran, ispisujemo lepu tablicu
        from tabulate import tabulate
        table = []
        for name, meta in results.items():
            reached_val = meta.get('best_info', {}).get('reached', None)
            table.append([name, meta['best_J'], meta['evals'], f"{meta['time_s']:.2f}s", reached_val])
        print("\nSummary:")
        print(tabulate(table, headers=["algo", "final_best_J", "evals", "time", "reached"], tablefmt="github"))
    except Exception:
        # fallback: jednostavan tekstualni ispis
        print("\nSummary (plain):")
        for name, meta in results.items():
            print(
                f"{name}: final_best_J={meta['best_J']}, evals={meta['evals']}, time={meta['time_s']:.2f}s, reached={meta.get('best_info', {}).get('reached', None)}")

    return results, best_so_far, out_json, out_png_full


# ---------- CLI entrypoint ----------
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