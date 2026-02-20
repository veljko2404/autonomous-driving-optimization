import numpy as np


def optimize(obj, x0, iters=70, seed=0):
    """
    CMA-ES (ozbiljnija verzija, ali bez ekstremne komplikacije)
    obj: funkcija theta(dict) -> (J, info)
    x0: početni dict parametara
    """

    rng = np.random.default_rng(seed)

    keys = list(x0.keys())
    n = len(keys)

    def dict_to_vec(d):
        return np.array([d[k] for k in keys], dtype=float)

    def vec_to_dict(v):
        return {k: float(v[i]) for i, k in enumerate(keys)}

    x_mean = dict_to_vec(x0)

    # --- Strategijski parametri ---
    lambda_ = 4 + int(3 * np.log(n))
    mu = lambda_ // 2

    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mu_eff = 1.0 / np.sum(weights**2)

    sigma = 0.3

    # Adaptacioni parametri
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1)/(n + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
    c1 = 2 / ((n + 1.3)**2 + mu_eff)
    c_mu = min(
        1 - c1,
        2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff)
    )

    # Inicijalizacija
    C = np.eye(n)
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)

    history = []

    for iteration in range(iters):

        # Eigendecomposition
        eigvals, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(eigvals, 1e-12))
        inv_sqrt_C = B @ np.diag(1/D) @ B.T

        # Sample populacija
        population = []
        zs = []
        for _ in range(lambda_):
            z = rng.standard_normal(n)
            x = x_mean + sigma * (B @ (D * z))
            population.append(x)
            zs.append(z)

        # Evaluacija
        fitness = []
        infos = []
        for x in population:
            J, info = obj(vec_to_dict(x))
            fitness.append(J)
            infos.append(info)

        fitness = np.array(fitness)

        # Sortiranje
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

        # Selekcija top μ
        x_old = x_mean.copy()

        # --- ispravan, brz i ne-deprecated način za izračunavanje težinske sredine ---
        pop_mat = np.vstack(population[:mu])    # shape (mu, n)
        # weights shape (mu,), pop_mat shape (mu,n) -> result (n,)
        x_mean = weights @ pop_mat

        zs_mat = np.vstack(zs[:mu])             # shape (mu, n)
        z_mean = weights @ zs_mat

        # Update evolution path sigma
        p_sigma = (
            (1 - c_sigma) * p_sigma +
            np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) *
            (inv_sqrt_C @ (x_mean - x_old) / sigma)
        )

        # Adaptacija sigma
        norm_ps = np.linalg.norm(p_sigma)
        expected_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / expected_norm - 1))

        # Update evolution path C
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

        # Rank-one update
        rank_one = np.outer(p_c, p_c)

        # Rank-mu update
        rank_mu = np.zeros((n, n))
        for i in range(mu):
            y = (population[i] - x_old) / sigma
            rank_mu += weights[i] * np.outer(y, y)

        C = (
            (1 - c1 - c_mu) * C +
            c1 * rank_one +
            c_mu * rank_mu
        )

    best_theta = vec_to_dict(population[0])
    best_J, best_info = obj(best_theta)

    return best_J, best_theta, best_info, history