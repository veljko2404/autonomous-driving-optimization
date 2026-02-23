import numpy as np


def optimize(obj, x0, max_iters=200, tol=1e-6):
    """
    Nelder-Mead simplex metoda optimizacije (bez gradijenata).

    Parameters
    ----------
    obj : function
        Funkcija cilja. Prima dict parametara (theta) i vraća tuple:
        (J, info) gde je:
            J    - vrednost funkcije cilja
            info - dodatne informacije (npr. metrike, debug podaci)

    x0 : dict
        Početne vrednosti parametara, npr. {"a": 1.0, "b": 2.0}

    max_iters : int
        Maksimalan broj iteracija algoritma.

    tol : float
        Tolerancija zaustavljanja (na osnovu standardne devijacije
        vrednosti funkcije cilja unutar simpleksa).

    Returns
    -------
    best_J : float
        Najmanja pronađena vrednost funkcije cilja.

    best_params : dict
        Parametri koji daju najbolju vrednost.

    best_info : any
        Dodatne informacije iz obj funkcije za najbolju tačku.

    history : list
        Istorija optimizacije (iteracija i najbolji J).
    """

    # Redosled ključeva parametara (bitno za mapiranje dict <-> vektor)
    keys = list(x0.keys())
    n = len(keys)

    # Konverzija dict -> numpy vektor
    def dict_to_vec(d):
        return np.array([d[k] for k in keys], dtype=float)

    # Konverzija numpy vektor -> dict
    def vec_to_dict(v):
        return {k: float(v[i]) for i, k in enumerate(keys)}

    # Početna tačka u vektorskoj formi
    x_start = dict_to_vec(x0)

    # ==========================
    # Konstrukcija početnog simpleksa
    # Simplex ima n+1 tačaka u n-dimenzionalnom prostoru
    # ==========================
    simplex = [x_start]

    for i in range(n):
        x = x_start.copy()
        # Mala perturbacija po i-toj koordinati
        x[i] += 0.05 * (abs(x[i]) + 1.0)
        simplex.append(x)

    simplex = np.array(simplex)

    # Standardni Nelder-Mead koeficijenti
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    history = []

    # ==========================
    # Glavna optimizaciona petlja
    # ==========================
    for iteration in range(max_iters):

        values = []
        infos = []

        # Evaluacija funkcije cilja za sve tačke simpleksa
        for x in simplex:
            J, info = obj(vec_to_dict(x))
            values.append(J)
            infos.append(info)

        values = np.array(values)

        # Sortiranje simpleksa po vrednosti funkcije (rastuce)
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]
        infos = [infos[i] for i in order]

        # Najbolja, najgora i druga najgora tačka
        best = simplex[0]
        worst = simplex[-1]
        second_worst = simplex[-2]

        # Čuvanje istorije
        history.append({
            "iter": iteration,
            "best_J": float(values[0])
        })

        # Kriterijum zaustavljanja:
        # Ako su vrednosti unutar simpleksa dovoljno bliske
        if np.std(values) < tol:
            break

        # Centroid svih tačaka osim najgore
        centroid = np.mean(simplex[:-1], axis=0)

        # ==========================
        # REFLECTION
        # ==========================
        xr = centroid + alpha * (centroid - worst)
        Jr, _ = obj(vec_to_dict(xr))

        # Ako je reflektovana tačka bolja od druge najgore,
        # ali ne bolja od najbolje → prihvati refleksiju
        if values[0] <= Jr < values[-2]:
            simplex[-1] = xr
            continue

        # ==========================
        # EXPANSION
        # ==========================
        # Ako je refleksija najbolja do sada → probaj ekspanziju
        if Jr < values[0]:
            xe = centroid + gamma * (xr - centroid)
            Je, _ = obj(vec_to_dict(xe))

            # Zadrži bolju između expansion i reflection
            simplex[-1] = xe if Je < Jr else xr
            continue

        # ==========================
        # CONTRACTION
        # ==========================
        xc = centroid + rho * (worst - centroid)
        Jc, _ = obj(vec_to_dict(xc))

        # Ako je kontrakcija bolja od najgore → prihvati
        if Jc < values[-1]:
            simplex[-1] = xc
            continue

        # ==========================
        # SHRINK
        # ==========================
        # Ako ništa nije uspelo, smanji ceo simplex
        best = simplex[0]
        new_simplex = [best]

        for x in simplex[1:]:
            xs = best + sigma * (x - best)
            new_simplex.append(xs)

        simplex = np.array(new_simplex)

    # Konačna najbolja tačka
    best_vec = simplex[0]
    best_J, best_info = obj(vec_to_dict(best_vec))

    return best_J, vec_to_dict(best_vec), best_info, history