import numpy as np


def optimize(obj, x0, iters=70, seed=0, sigma = 0.3):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

    Globalna, bezgradijentna optimizaciona metoda.
    Održava multivarijantnu normalnu distribuciju nad parametrima
    i iterativno adaptira:
        - srednju vrednost (x_mean)
        - kovarijacionu matricu (C)
        - globalni step size (sigma)

    Parameters
    ----------
    obj : function
        Funkcija cilja: theta(dict) -> (J, info)
        J    - vrednost funkcije cilja
        info - dodatne informacije (debug, metrike…)

    x0 : dict
        Početni parametri.

    iters : int
        Broj generacija (iteracija CMA-ES algoritma).

    seed : int
        Seed za reproduktivnost (kontrola random generatora).

    sigma : float
        Globalni step-size (početna širina distribucije)

    Returns
    -------
    best_J : float
        Najbolja pronađena vrednost funkcije cilja.

    best_theta : dict
        Parametri koji daju najbolju vrednost.

    best_info : any
        Dodatne informacije za najbolju tačku.

    history : list
        Istorija optimizacije (najbolji J po iteraciji).
    """

    # Random generator (stabilan i moderan NumPy API)
    rng = np.random.default_rng(seed)

    # Fiksiramo redosled parametara (bitno za mapiranje dict <-> vektor)
    keys = list(x0.keys())
    n = len(keys)

    # Konverzija dict -> numpy vektor
    def dict_to_vec(d):
        return np.array([d[k] for k in keys], dtype=float)

    # Konverzija numpy vektor -> dict
    def vec_to_dict(v):
        return {k: float(v[i]) for i, k in enumerate(keys)}

    # Početna srednja vrednost distribucije
    x_mean = dict_to_vec(x0)

    # ==========================================================
    # STRATEGIJSKI PARAMETRI (standardne CMA-ES formule)
    # ==========================================================

    # Veličina populacije (λ)
    lambda_ = 4 + int(3 * np.log(n))

    # Broj selektovanih najboljih jedinki (μ)
    mu = lambda_ // 2

    # Logaritamske težine (favorizuju najbolje)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)

    # Efektivni broj roditelja
    mu_eff = 1.0 / np.sum(weights**2)

    # ==========================================================
    # ADAPTACIONI KOEFICIJENTI
    # ==========================================================

    # Parametri za adaptaciju sigma
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1)/(n + 1)) - 1) + c_sigma

    # Parametri za adaptaciju kovarijacije
    c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
    c1 = 2 / ((n + 1.3)**2 + mu_eff)

    # Ograničenje da c1 + c_mu <= 1
    c_mu = min(
        1 - c1,
        2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff)
    )

    # ==========================================================
    # INICIJALIZACIJA STRUKTURA
    # ==========================================================

    # Kovarijaciona matrica (inicijalno identitet)
    C = np.eye(n)

    # Evolution path za sigma (kontroliše skaliranje)
    p_sigma = np.zeros(n)

    # Evolution path za kovarijaciju
    p_c = np.zeros(n)

    history = []

    # ==========================================================
    # GLAVNA PETLJA (po generacijama)
    # ==========================================================
    for iteration in range(iters):

        # ------------------------------------------------------
        # Eigendecomposition kovarijacione matrice
        # C = B D^2 B^T
        # ------------------------------------------------------
        eigvals, B = np.linalg.eigh(C)

        # D = sqrt(eigenvalues)
        D = np.sqrt(np.maximum(eigvals, 1e-12))

        # Inverzna kvadratna koren matrica (za whitening)
        inv_sqrt_C = B @ np.diag(1/D) @ B.T

        # ------------------------------------------------------
        # SAMPLE POPULACIJE
        # x ~ N(x_mean, sigma^2 C)
        # ------------------------------------------------------
        population = []
        zs = []

        for _ in range(lambda_):
            # z ~ N(0, I)
            z = rng.standard_normal(n)

            # Transformacija u korelisani prostor
            x = x_mean + sigma * (B @ (D * z)) # sigma je globalni step-size (početna širina distribucije)

            population.append(x)
            zs.append(z)

        # ------------------------------------------------------
        # Evaluacija funkcije cilja
        # ------------------------------------------------------
        fitness = []
        infos = []

        for x in population:
            J, info = obj(vec_to_dict(x))
            fitness.append(J)
            infos.append(info)

        fitness = np.array(fitness)

        # ------------------------------------------------------
        # Sortiranje po fitness-u (minimizacija)
        # ------------------------------------------------------
        idx = np.argsort(fitness)

        population = [population[i] for i in idx]
        zs = [zs[i] for i in idx]
        fitness = fitness[idx]
        infos = [infos[i] for i in idx]

        best_J = float(fitness[0])

        history.append({
            "iter": iteration,
            "best_J": best_J
        })

        # ------------------------------------------------------
        # SELEKCIJA TOP μ JEDINKI
        # ------------------------------------------------------
        x_old = x_mean.copy()

        # Težinska sredina novih roditelja
        pop_mat = np.vstack(population[:mu])  # (mu, n)
        x_mean = weights @ pop_mat           # nova sredina

        zs_mat = np.vstack(zs[:mu])
        z_mean = weights @ zs_mat

        # ------------------------------------------------------
        # UPDATE EVOLUTION PATH ZA SIGMA
        # ------------------------------------------------------
        p_sigma = (
            (1 - c_sigma) * p_sigma +
            np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) *
            (inv_sqrt_C @ (x_mean - x_old) / sigma)
        )

        # Adaptacija globalnog step-size
        norm_ps = np.linalg.norm(p_sigma)

        # Očekivana norma Gauss vektora
        expected_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / expected_norm - 1))

        # ------------------------------------------------------
        # UPDATE EVOLUTION PATH ZA C
        # ------------------------------------------------------
        h_sigma = int(
            norm_ps / np.sqrt(
                1 - (1 - c_sigma)**(2*(iteration+1))
            ) < (1.4 + 2/(n+1)) * expected_norm
        )

        p_c = (
            (1 - c_c) * p_c +
            h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) *
            (x_mean - x_old) / sigma
        )

        # ------------------------------------------------------
        # RANK-ONE UPDATE
        # (informacija iz evolution path-a)
        # ------------------------------------------------------
        rank_one = np.outer(p_c, p_c)

        # ------------------------------------------------------
        # RANK-μ UPDATE
        # (informacija iz najboljih μ jedinki)
        # ------------------------------------------------------
        rank_mu = np.zeros((n, n))

        for i in range(mu):
            y = (population[i] - x_old) / sigma
            rank_mu += weights[i] * np.outer(y, y)

        # ------------------------------------------------------
        # Ažuriranje kovarijacione matrice
        # ------------------------------------------------------
        C = (
            (1 - c1 - c_mu) * C +
            c1 * rank_one +
            c_mu * rank_mu
        )

    # ==========================================================
    # Konačno najbolje rešenje iz poslednje populacije
    # ==========================================================
    best_theta = vec_to_dict(population[0])
    best_J, best_info = obj(best_theta)

    return best_J, best_theta, best_info, history