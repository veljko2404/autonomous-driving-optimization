import numpy as np


def optimize(obj, x0, max_iters=200, tol=1e-6):
    """
    Nelder-Mead simplex method (standard implementation).
    obj: funkcija koja prima dict theta i vraća (J, info)
    x0: početni dict parametara
    """

    keys = list(x0.keys())
    n = len(keys)

    def dict_to_vec(d):
        return np.array([d[k] for k in keys], dtype=float)

    def vec_to_dict(v):
        return {k: float(v[i]) for i, k in enumerate(keys)}

    x_start = dict_to_vec(x0)

    # inicijalni simplex
    simplex = [x_start]
    for i in range(n):
        x = x_start.copy()
        x[i] += 0.05 * (abs(x[i]) + 1.0)
        simplex.append(x)

    simplex = np.array(simplex)

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    history = []

    for iteration in range(max_iters):

        values = []
        infos = []
        for x in simplex:
            J, info = obj(vec_to_dict(x))
            values.append(J)
            infos.append(info)

        values = np.array(values)

        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]
        infos = [infos[i] for i in order]

        best = simplex[0]
        worst = simplex[-1]
        second_worst = simplex[-2]

        history.append({
            "iter": iteration,
            "best_J": float(values[0])
        })

        if np.std(values) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        # reflection
        xr = centroid + alpha * (centroid - worst)
        Jr, _ = obj(vec_to_dict(xr))

        if values[0] <= Jr < values[-2]:
            simplex[-1] = xr
            continue

        # expansion
        if Jr < values[0]:
            xe = centroid + gamma * (xr - centroid)
            Je, _ = obj(vec_to_dict(xe))
            simplex[-1] = xe if Je < Jr else xr
            continue

        # contraction
        xc = centroid + rho * (worst - centroid)
        Jc, _ = obj(vec_to_dict(xc))

        if Jc < values[-1]:
            simplex[-1] = xc
            continue

        # shrink
        best = simplex[0]
        new_simplex = [best]
        for x in simplex[1:]:
            xs = best + sigma * (x - best)
            new_simplex.append(xs)
        simplex = np.array(new_simplex)

    best_vec = simplex[0]
    best_J, best_info = obj(vec_to_dict(best_vec))

    return best_J, vec_to_dict(best_vec), best_info, history