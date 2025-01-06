import numpy as np
import numpy.typing as npt
import numba
from scipy.stats import binom


from utils import _Mhat, _rho
from params import RestrepoParams


@numba.njit
def calc_Mhat(cj: float, params: RestrepoParams) -> float:
    return _Mhat(_rho(cj, params), params.BCSQN)


@numba.njit
def ryr_gillespie(
    init: npt.NDArray, cs: float, cj: float, params: RestrepoParams, nstep: int
):
    t_out = np.zeros(nstep + 1)
    out = np.zeros((nstep + 1, 4), dtype=np.int64)

    state = np.copy(init)
    rates = np.zeros(8, dtype=float)
    cdf = np.zeros(8, dtype=float)
    out[0, :] = init

    Ku = params.Ku
    Kb = params.Kb
    k14 = calc_Mhat(cj, params)
    k21 = 1.0 / params.tau_c
    k23 = k14
    k41 = 1 / params.tau_u
    k34 = 1 / params.tau_c
    k32 = k41 * Ku / Kb

    kr = params.tau_ps * params.Jmax / params.vp
    N = float(init.sum())

    t = 0.0
    for i in range(nstep):
        Po = float(state[1] + state[2]) / N
        cp = (cs + Po * kr * cj) / (1 + kr * Po)
        k12 = Ku * cp**2
        k43 = Kb * cp**2

        rates[0] = k12 * state[0]
        rates[1] = k21 * state[1]
        rates[2] = k23 * state[1]
        rates[3] = k32 * state[2]
        rates[4] = k34 * state[2]
        rates[5] = k43 * state[3]
        rates[6] = k41 * state[3]
        rates[7] = k14 * state[0]

        cdf[:] = np.cumsum(rates)
        dt = -np.log(np.random.rand()) / cdf[-1]
        u = np.random.rand() * cdf[-1]
        if u < cdf[0]:
            state[0] -= 1
            state[1] += 1
        elif u < cdf[1]:
            state[1] -= 1
            state[0] += 1
        elif u < cdf[2]:
            state[1] -= 1
            state[2] += 1
        elif u < cdf[3]:
            state[2] -= 1
            state[1] += 1
        elif u < cdf[4]:
            state[2] -= 1
            state[3] += 1
        elif u < cdf[5]:
            state[3] -= 1
            state[2] += 1
        elif u < cdf[6]:
            state[3] -= 1
            state[0] += 1
        else:
            state[0] -= 1
            state[3] += 1

        t += dt
        t_out[i + 1] = t
        out[i + 1, :] = state
    return t_out, out


@numba.njit
def projection(state_sorted):
    l = 0.0
    sum_ = 0.0
    for i in range(4):
        sum_ += state_sorted[3 - i]
        l = (sum_ - 1.0) / (i + 1)
        if i == 3:
            break
        elif l >= state_sorted[2 - i]:
            break
    return l


@numba.njit
def projection_v2(state_sorted: npt.NDArray, D: npt.NDArray) -> float:
    mu = (state_sorted.sum() - 1.0) / D.sum()
    if state_sorted[0] - mu * D[0] >= 0:
        return mu
    mu = (state_sorted[1:].sum() - 1.0) / D[1:].sum()
    if state_sorted[1] - mu * D[1] >= 0:
        return mu
    mu = (state_sorted[2:].sum() - 1.0) / D[2:].sum()
    if state_sorted[2] - mu * D[2] >= 0:
        return mu
    return (state_sorted[-1] - 1.0) / D[-1]


@numba.njit
def projection_v3(state_sorted: npt.NDArray, D: npt.NDArray, a: float) -> float:
    mu = (state_sorted.sum() - a) / D.sum()
    if state_sorted[0] - mu * D[0] >= 0:
        return mu
    mu = (state_sorted[1:].sum() - a) / D[1:].sum()
    if state_sorted[1] - mu * D[1] >= 0:
        return mu
    return (state_sorted[-1] - a) / D[-1]


@numba.njit
def ryr_diffusion(
    init: npt.NDArray,
    cs: float,
    cj: float,
    eps: float,
    params: RestrepoParams,
    nstep: int,
    dt: float,
):
    t_out = np.zeros(nstep + 1)
    out = np.zeros((nstep + 1, 4))

    state = np.copy(init)
    state_sorted = np.copy(init)
    drift = np.zeros(4, dtype=float)
    sigma = np.zeros(4, dtype=float)
    out[0, :] = init

    Ku = params.Ku
    Kb = params.Kb
    k14 = calc_Mhat(cj, params)
    k21 = 1.0 / params.tau_c
    k23 = k14
    k41 = 1 / params.tau_u
    k34 = 1 / params.tau_c
    k32 = k41 * Ku / Kb

    kr = params.tau_ps * params.Jmax / params.vp
    sqrtdt = np.sqrt(dt)

    for i in range(nstep):
        Po = state[1] + state[2]
        cp = (cs + Po * kr * cj) / (1 + kr * Po)
        k12 = Ku * cp**2
        k43 = Kb * cp**2

        drift[0] = k21 * state[1] + k41 * state[3] - (k12 + k14) * state[0]
        drift[1] = k12 * state[0] + k32 * state[2] - (k23 + k21) * state[1]
        drift[2] = k23 * state[1] + k43 * state[3] - (k32 + k34) * state[2]
        drift[3] = k34 * state[2] + k14 * state[0] - (k43 + k41) * state[3]

        sigma[0] = eps * np.sqrt(k12 * state[0] + k21 * state[1])
        sigma[1] = eps * np.sqrt(k23 * state[1] + k32 * state[2])
        sigma[2] = eps * np.sqrt(k34 * state[2] + k43 * state[3])
        sigma[3] = eps * np.sqrt(k14 * state[0] + k41 * state[3])

        dW12 = sqrtdt * np.random.normal()
        dW23 = sqrtdt * np.random.normal()
        dW34 = sqrtdt * np.random.normal()
        dW14 = sqrtdt * np.random.normal()

        state[0] += dt * drift[0] + sigma[0] * dW12 + sigma[3] * dW14
        state[1] += dt * drift[1] - sigma[0] * dW12 + sigma[1] * dW23
        state[2] += dt * drift[2] - sigma[1] * dW23 + sigma[2] * dW34
        state[3] += dt * drift[3] - sigma[2] * dW34 - sigma[0] * dW14

        state_sorted[:] = np.sort(state)
        l = projection(state_sorted)

        state[0] = max(state[0] - l, 0.0)
        state[1] = max(state[1] - l, 0.0)
        state[2] = max(state[2] - l, 0.0)
        state[3] = max(state[3] - l, 0.0)

        t_out[i + 1] = t_out[i] + dt
        out[i + 1, :] = state

    return t_out, out


@numba.njit
def ryr_diffusion_v2(
    init: npt.NDArray,
    cs: float,
    cj: float,
    eps: float,
    params: RestrepoParams,
    nstep: int,
    dt: float,
):
    t_out = np.zeros(nstep + 1)
    out = np.zeros((nstep + 1, 4))

    state = np.copy(init)
    state_sorted = np.copy(init)
    weighted_state = np.copy(init)
    sort_idx = np.zeros(4, dtype=np.int64)

    drift = np.zeros(4, dtype=float)
    sigma = np.zeros(4, dtype=float)
    D = np.zeros(4, dtype=float)
    out[0, :] = init

    Ku = params.Ku
    Kb = params.Kb
    k14 = calc_Mhat(cj, params)
    k21 = 1.0 / params.tau_c
    k23 = k14
    k41 = 1 / params.tau_u
    k34 = 1 / params.tau_c
    k32 = k41 * Ku / Kb

    kr = params.tau_ps * params.Jmax / params.vp
    sqrtdt = np.sqrt(dt)

    for i in range(nstep):
        Po = state[1] + state[2]
        cp = (cs + Po * kr * cj) / (1 + kr * Po)
        k12 = Ku * cp**2
        k43 = Kb * cp**2

        drift[0] = k21 * state[1] + k41 * state[3] - (k12 + k14) * state[0]
        drift[1] = k12 * state[0] + k32 * state[2] - (k23 + k21) * state[1]
        drift[2] = k23 * state[1] + k43 * state[3] - (k32 + k34) * state[2]
        drift[3] = k34 * state[2] + k14 * state[0] - (k43 + k41) * state[3]

        sigma[0] = eps * np.sqrt(k12 * state[0] + k21 * state[1])
        sigma[1] = eps * np.sqrt(k23 * state[1] + k32 * state[2])
        sigma[2] = eps * np.sqrt(k34 * state[2] + k43 * state[3])
        sigma[3] = eps * np.sqrt(k14 * state[0] + k41 * state[3])

        dW12 = sqrtdt * np.random.normal()
        dW23 = sqrtdt * np.random.normal()
        dW34 = sqrtdt * np.random.normal()
        dW14 = sqrtdt * np.random.normal()

        state[0] += dt * drift[0] + sigma[0] * dW12 + sigma[3] * dW14
        state[1] += dt * drift[1] - sigma[0] * dW12 + sigma[1] * dW23
        state[2] += dt * drift[2] - sigma[1] * dW23 + sigma[2] * dW34
        state[3] += dt * drift[3] - sigma[2] * dW34 - sigma[0] * dW14

        D[0] = sigma[0] ** 2 + sigma[3] ** 2  # sigma_12^2 + sigma_14^2
        D[1] = sigma[0] ** 2 + sigma[1] ** 2  # sigma_12^2 + sigma_23^2
        D[2] = sigma[1] ** 2 + sigma[2] ** 2  # sigma_23^2 + sigma_34^2
        D[3] = sigma[2] ** 2 + sigma[3] ** 2  # sigma_23^2 + sigma_14^2

        weighted_state[0] = state[0] / D[0] if D[0] > 0 else 0.0
        weighted_state[1] = state[1] / D[1] if D[1] > 0 else 0.0
        weighted_state[2] = state[2] / D[2] if D[2] > 0 else 0.0
        weighted_state[3] = state[3] / D[3] if D[3] > 0 else 0.0
        sort_idx[:] = np.argsort(weighted_state)
        state_sorted[:] = state[sort_idx]

        mu = projection_v2(state_sorted, D[sort_idx])

        state[0] = max(state[0] - D[0] * mu, 0.0)
        state[1] = max(state[1] - D[1] * mu, 0.0)
        state[2] = max(state[2] - D[2] * mu, 0.0)
        state[3] = max(state[3] - D[3] * mu, 0.0)

        t_out[i + 1] = t_out[i] + dt
        out[i + 1, :] = state

    return t_out, out


@numba.njit
def ryr_diffusion_v3(
    init: npt.NDArray,
    cs: float,
    cj: float,
    eps: float,
    params: RestrepoParams,
    nstep: int,
    dt: float,
):
    t_out = np.zeros(nstep + 1)
    out = np.zeros((nstep + 1, 4))

    state = np.copy(init)
    state_sorted = np.zeros(3, dtype=float)
    weighted_state = np.zeros(3, dtype=float)
    sort_idx = np.zeros(3, dtype=np.int64)

    drift = np.zeros(4, dtype=float)
    sigma = np.zeros(4, dtype=float)
    D = np.zeros(3, dtype=float)
    out[0, :] = init

    Ku = params.Ku
    Kb = params.Kb
    k14 = calc_Mhat(cj, params)
    k21 = 1.0 / params.tau_c
    k23 = k14
    k41 = 1 / params.tau_u
    k34 = 1 / params.tau_c
    k32 = k41 * Ku / Kb

    kr = params.tau_ps * params.Jmax / params.vp
    sqrtdt = np.sqrt(dt)

    for i in range(nstep):
        Po = state[1] + state[2]
        cp = (cs + Po * kr * cj) / (1 + kr * Po)
        k12 = Ku * cp**2
        k43 = Kb * cp**2

        drift[0] = k21 * state[1] + k41 * state[3] - (k12 + k14) * state[0]
        drift[1] = k12 * state[0] + k32 * state[2] - (k23 + k21) * state[1]
        drift[2] = k23 * state[1] + k43 * state[3] - (k32 + k34) * state[2]
        drift[3] = k34 * state[2] + k14 * state[0] - (k43 + k41) * state[3]

        sigma[0] = eps * np.sqrt(k12 * state[0] + k21 * state[1])
        sigma[1] = eps * np.sqrt(k23 * state[1] + k32 * state[2])
        sigma[2] = eps * np.sqrt(k34 * state[2] + k43 * state[3])
        sigma[3] = eps * np.sqrt(k14 * state[0] + k41 * state[3])

        dW12 = sqrtdt * np.random.normal()
        dW23 = sqrtdt * np.random.normal()
        dW34 = sqrtdt * np.random.normal()
        dW14 = sqrtdt * np.random.normal()

        state[0] += dt * drift[0] + sigma[0] * dW12 + sigma[3] * dW14
        state[1] += dt * drift[1] - sigma[0] * dW12 + sigma[1] * dW23
        state[2] += dt * drift[2] - sigma[1] * dW23 + sigma[2] * dW34
        # state[3] += dt * drift[3] - sigma[2] * dW34 - sigma[0] * dW14

        D[0] = sigma[0] ** 2 + sigma[3] ** 2  # sigma_12^2 + sigma_14^2
        D[1] = sigma[0] ** 2 + sigma[1] ** 2  # sigma_12^2 + sigma_23^2
        D[2] = sigma[1] ** 2 + sigma[2] ** 2  # sigma_23^2 + sigma_34^2
        # D[3] = sigma[2] ** 2 + sigma[3] ** 2  # sigma_23^2 + sigma_14^2

        weighted_state[0] = state[0] / D[0] if D[0] > 0 else 0.0
        weighted_state[1] = state[1] / D[1] if D[1] > 0 else 0.0
        weighted_state[2] = state[2] / D[2] if D[2] > 0 else 0.0
        # weighted_state[3] = state[3] / D[3] if D[3] > 0 else 0.0
        sort_idx[:] = np.argsort(weighted_state)
        state_sorted[:] = state[:-1][sort_idx]

        a = 1.0 if state_sorted.sum() > 1.0 else state_sorted.sum()
        mu = projection_v3(state_sorted, D[sort_idx], a)

        state[0] = max(state[0] - D[0] * mu, 0.0)
        state[1] = max(state[1] - D[1] * mu, 0.0)
        state[2] = max(state[2] - D[2] * mu, 0.0)
        state[3] = 1.0 - (state[0] + state[1] + state[2])

        t_out[i + 1] = t_out[i] + dt
        out[i + 1, :] = state

    return t_out, out


@numba.njit
def ryr_diffusion_relaxed_bdy(
    init: npt.NDArray,
    cs: float,
    cj: float,
    eps: float,
    params: RestrepoParams,
    nstep: int,
    dt: float,
):
    t_out = np.zeros(nstep + 1)
    out = np.zeros((nstep + 1, 4))

    state = np.copy(init)
    drift = np.zeros(3, dtype=float)
    sigma = np.zeros(4, dtype=float)
    out[0, :] = init

    Ku = params.Ku
    Kb = params.Kb
    k14 = calc_Mhat(cj, params)
    k21 = 1.0 / params.tau_c
    k23 = k14
    k41 = 1 / params.tau_u
    k34 = 1 / params.tau_c
    k32 = k41 * Ku / Kb

    kr = params.tau_ps * params.Jmax / params.vp
    sqrtdt = np.sqrt(dt)

    for i in range(nstep):
        Po = state[1] + state[2]
        cp = (cs + Po * kr * cj) / (1 + kr * Po)
        k12 = Ku * cp**2
        k43 = Kb * cp**2

        drift[0] = k21 * state[1] + k41 * state[3] - (k12 + k14) * state[0]
        drift[1] = k12 * state[0] + k32 * state[2] - (k23 + k21) * state[1]
        drift[2] = k23 * state[1] + k43 * state[3] - (k32 + k34) * state[2]
        # drift[3] = k34 * state[2] + k14 * state[0] - (k43 + k41) * state[3]

        sigma[0] = eps * np.sqrt(max(k12 * state[0] + k21 * state[1], 0))
        sigma[1] = eps * np.sqrt(max(k23 * state[1] + k32 * state[2], 0))
        sigma[2] = eps * np.sqrt(max(k34 * state[2] + k43 * state[3], 0))
        sigma[3] = eps * np.sqrt(max(k14 * state[0] + k41 * state[3], 0))

        dW12 = sqrtdt * np.random.normal()
        dW23 = sqrtdt * np.random.normal()
        dW34 = sqrtdt * np.random.normal()
        dW14 = sqrtdt * np.random.normal()

        state[0] += dt * drift[0] + sigma[0] * dW12 + sigma[3] * dW14
        state[1] += dt * drift[1] - sigma[0] * dW12 + sigma[1] * dW23
        state[2] += dt * drift[2] - sigma[1] * dW23 + sigma[2] * dW34

        # Orthogonal projection onto the subspace R2 + R3 = 0
        if state[1] + state[2] < 0:
            Po = state[1] + state[2]
            state[1] -= 0.5 * Po
            state[2] -= 0.5 * Po

        state[3] = 1 - (state[0] + state[1] + state[2])

        t_out[i + 1] = t_out[i] + dt
        out[i + 1, :] = state

    return t_out, out


@numba.njit
def ryr_diffusion_no_bdy(
    init: npt.NDArray,
    cs: float,
    cj: float,
    eps: float,
    params: RestrepoParams,
    nstep: int,
    dt: float,
):
    t_out = np.zeros(nstep + 1)
    out = np.zeros((nstep + 1, 4))

    state = np.copy(init)
    drift = np.zeros(3, dtype=float)
    sigma = np.zeros(4, dtype=float)
    out[0, :] = init

    Ku = params.Ku
    Kb = params.Kb
    k14 = calc_Mhat(cj, params)
    k21 = 1.0 / params.tau_c
    k23 = k14
    k41 = 1 / params.tau_u
    k34 = 1 / params.tau_c
    k32 = k41 * Ku / Kb

    kr = params.tau_ps * params.Jmax / params.vp
    sqrtdt = np.sqrt(dt)

    for i in range(nstep):
        Po = max(state[1] + state[2], 0)
        cp = (cs + Po * kr * cj) / (1 + kr * Po)
        k12 = Ku * cp**2
        k43 = Kb * cp**2

        drift[0] = k21 * state[1] + k41 * state[3] - (k12 + k14) * state[0]
        drift[1] = k12 * state[0] + k32 * state[2] - (k23 + k21) * state[1]
        drift[2] = k23 * state[1] + k43 * state[3] - (k32 + k34) * state[2]
        # drift[3] = k34 * state[2] + k14 * state[0] - (k43 + k41) * state[3]

        sigma[0] = eps * np.sqrt(max(k12 * state[0] + k21 * state[1], 0))
        sigma[1] = eps * np.sqrt(max(k23 * state[1] + k32 * state[2], 0))
        sigma[2] = eps * np.sqrt(max(k34 * state[2] + k43 * state[3], 0))
        sigma[3] = eps * np.sqrt(max(k14 * state[0] + k41 * state[3], 0))

        dW12 = sqrtdt * np.random.normal()
        dW23 = sqrtdt * np.random.normal()
        dW34 = sqrtdt * np.random.normal()
        dW14 = sqrtdt * np.random.normal()

        state[0] += dt * drift[0] + sigma[0] * dW12 + sigma[3] * dW14
        state[1] += dt * drift[1] - sigma[0] * dW12 + sigma[1] * dW23
        state[2] += dt * drift[2] - sigma[1] * dW23 + sigma[2] * dW34
        state[3] = 1 - (state[0] + state[1] + state[2])

        t_out[i + 1] = t_out[i] + dt
        out[i + 1, :] = state

    return t_out, out


@numba.njit
def ryr_diffusion_penalised_qv(
    init: npt.NDArray,
    cs: float,
    cj: float,
    eps: float,
    params: RestrepoParams,
    nstep: int,
    dt: float,
):
    t_out = np.zeros(nstep + 1)
    out = np.zeros((nstep + 1, 4))

    state = np.copy(init)
    state_sorted = np.copy(init)
    drift = np.zeros(4, dtype=float)
    sigma = np.zeros(4, dtype=float)
    penalty = np.zeros(4, dtype=float)
    out[0, :] = init

    Ku = params.Ku
    Kb = params.Kb
    k14 = calc_Mhat(cj, params)
    k21 = 1.0 / params.tau_c
    k23 = k14
    k41 = 1 / params.tau_u
    k34 = 1 / params.tau_c
    k32 = k41 * Ku / Kb

    kr = params.tau_ps * params.Jmax / params.vp
    sqrtdt = np.sqrt(dt)

    for i in range(nstep):
        Po = state[1] + state[2]
        cp = (cs + Po * kr * cj) / (1 + kr * Po)
        k12 = Ku * cp**2
        k43 = Kb * cp**2

        drift[0] = k21 * state[1] + k41 * state[3] - (k12 + k14) * state[0]
        drift[1] = k12 * state[0] + k32 * state[2] - (k23 + k21) * state[1]
        drift[2] = k23 * state[1] + k43 * state[3] - (k32 + k34) * state[2]
        drift[3] = k34 * state[2] + k14 * state[0] - (k43 + k41) * state[3]

        sigma[0] = eps * np.sqrt(k12 * state[0] + k21 * state[1])
        sigma[1] = eps * np.sqrt(k23 * state[1] + k32 * state[2])
        sigma[2] = eps * np.sqrt(k34 * state[2] + k43 * state[3])
        sigma[3] = eps * np.sqrt(k14 * state[0] + k41 * state[3])

        penalty[0] = 0.25 * eps**2 * (k12 + k14 - (k21 + k41))
        penalty[1] = 0.25 * eps**2 * (k21 + k23 - (k12 + k32))
        penalty[2] = 0.25 * eps**2 * (k32 + k34 - (k23 + k43))
        penalty[3] = 0.25 * eps**2 * (k41 + k43 - (k14 + k34))

        dW12 = sqrtdt * np.random.normal()
        dW23 = sqrtdt * np.random.normal()
        dW34 = sqrtdt * np.random.normal()
        dW14 = sqrtdt * np.random.normal()

        state[0] += dt * (drift[0] - penalty[0]) + sigma[0] * dW12 + sigma[3] * dW14
        state[1] += dt * (drift[1] - penalty[1]) - sigma[0] * dW12 + sigma[1] * dW23
        state[2] += dt * (drift[2] - penalty[2]) - sigma[1] * dW23 + sigma[2] * dW34
        state[3] += dt * (drift[3] - penalty[3]) - sigma[2] * dW34 - sigma[0] * dW14

        state_sorted[:] = np.sort(state)
        l = projection(state_sorted)

        state[0] = max(state[0] - l, 0.0)
        state[1] = max(state[1] - l, 0.0)
        state[2] = max(state[2] - l, 0.0)
        state[3] = max(state[3] - l, 0.0)

        t_out[i + 1] = t_out[i] + dt
        out[i + 1, :] = state

    return t_out, out


def Po_stationary_logp(
    N: int, cs: float, cj: float, params: RestrepoParams, truncate: int | None = None
) -> npt.NDArray:

    if truncate is None:
        N_ = N
    else:
        N_ = truncate

    log_probs = np.zeros(N_ + 1)
    Mhat = calc_Mhat(cj, params)
    kr = params.tau_ps * params.Jmax / params.vp

    log_t1 = np.log(
        params.tau_b * params.Ku + params.tau_u * Mhat * params.Kb
    ) - np.log(params.tau_b + params.tau_u * Mhat)

    log_N_fact = np.sum(np.log(np.arange(1, N + 1)))
    log_n_fact = 0.0
    log_N_minus_n_fact = log_N_fact

    log_t2 = 0.0

    for i in range(N_):
        n = i + 1
        log_n_fact += np.log(n)
        log_N_minus_n_fact -= np.log(N - i)

        log_binom_coeff = log_N_fact - log_n_fact - log_N_minus_n_fact

        log_t2 += np.log(cs + kr * cj * (n - 1) / N) - np.log(1.0 + kr * (n - 1) / N)
        log_probs[n] = log_binom_coeff + n * log_t1 + 2 * log_t2

    # log sum exp to get normalisation
    m = np.max(log_probs)
    lse = np.log(np.sum(np.exp(log_probs - m))) + m
    return log_probs - lse


def Po_R1_log_joint(
    n: int, r1: int, log_Po_probs: npt.NDArray, cj: float, params: RestrepoParams
) -> float:
    Mhat = _Mhat(_rho(cj, params), params.BCSQN)
    p = params.tau_b / (params.tau_u * Mhat + params.tau_b)
    return log_Po_probs[n] + binom(len(log_Po_probs) - 1, p).logpmf(r1)


def Po_fixed_point(cs: float, cj: float, params: RestrepoParams):
    kr = float(params.tau_ps) * float(params.Jmax) / float(params.vp)
    Mhat = float(calc_Mhat(cj, params))
    beta = (params.tau_b + Mhat * params.tau_u) / (
        params.tau_b * params.Ku + params.tau_u * Mhat * params.Kb
    )

    a = kr**2 * (cj**2 + beta)
    b = kr * (2 * (cs * cj + beta) - kr * cj**2)
    c = cs**2 + beta - 2 * kr * cj * cs
    d = -(cs**2)
    return np.roots([a, b, c, d])


def jacobian(
    ryr: npt.NDArray,
    cj: float,
    cs: float,
    Mhat: float,
    params: RestrepoParams,
):
    Po = ryr[1] + ryr[2]
    kr = params.tau_ps * params.Jmax / params.vp
    cp = (cs + kr * Po * cj) / (1 + kr * Po)
    cp2 = cp**2
    dcp = 2 * kr * cp * (cj - cs) / (1 + kr * Po) ** 2
    x4 = 1 - ryr.sum()

    out = np.zeros((3, 3))

    out[0, 0] = -(params.Ku * cp2 + Mhat / params.tau_b + 1 / params.tau_u)
    out[0, 1] = 1 - (1 / params.tau_u) - params.Ku * dcp * ryr[0]
    out[0, 2] = -params.Ku * dcp * ryr[0]

    out[1, 0] = params.Ku * cp2
    out[1, 1] = params.Ku * dcp * ryr[0] - (1 + Mhat / params.tau_b)
    out[1, 2] = params.Ku * dcp * ryr[0] + params.Ku / (params.Kb * params.tau_u)

    out[2, 0] = -params.Kb * cp2
    out[2, 1] = Mhat / params.tau_b - params.Kb * cp2 + params.Kb * dcp * x4
    out[2, 2] = params.Kb * (dcp * x4 - cp2) - (
        params.Ku / (params.Kb * params.tau_u) + 1
    )
    return out


def diffusion_mat(
    ryr: npt.NDArray, cs: float, cj: float, eps: float, params: RestrepoParams
):
    Po = ryr[1] + ryr[2]
    kr = params.tau_ps * params.Jmax / params.vp
    cp2 = ((cs + kr * cj * Po) / (1 + kr * Po)) ** 2

    k12 = params.Ku * cp2
    k43 = params.Kb * cp2
    k14 = calc_Mhat(cj, params)
    k21 = 1.0 / params.tau_c
    k23 = k14
    k41 = 1 / params.tau_u
    k34 = 1 / params.tau_c
    k32 = k41 * params.Ku / params.Kb

    sigma12 = eps * np.sqrt(k12 * ryr[0] + k21 * ryr[1])
    sigma23 = eps * np.sqrt(k23 * ryr[1] + k32 * ryr[2])
    sigma34 = eps * np.sqrt(k34 * ryr[2] + k43 * (1 - ryr.sum()))
    sigma14 = eps * np.sqrt(k14 * ryr[0] + k41 * (1 - ryr.sum()))

    return np.array(
        [[sigma12, 0, 0, sigma14], [-sigma12, sigma23, 0, 0], [0, -sigma23, sigma34, 0]]
    )


def fixed_points_and_jac(cs: float, cj: float, eps: float, params: RestrepoParams):
    Pos = np.sort(Po_fixed_point(cs, cj, params))
    Mhat = calc_Mhat(cj, params)

    fp1 = np.array(
        [
            (1 - Pos[0]) * params.tau_b / (params.tau_b + params.tau_u * Mhat),
            Pos[0]
            * (params.Ku * params.tau_b)
            / (params.Ku * params.tau_b + params.Kb * params.tau_u * Mhat),
            Pos[0]
            * (params.Kb * params.tau_u * Mhat)
            / (params.Ku * params.tau_b + params.Kb * params.tau_u * Mhat),
        ]
    )

    fp2 = np.array(
        [
            (1 - Pos[1]) * params.tau_b / (params.tau_b + params.tau_u * Mhat),
            Pos[1]
            * (params.Ku * params.tau_b)
            / (params.Ku * params.tau_b + params.Kb * params.tau_u * Mhat),
            Pos[1]
            * (params.Kb * params.tau_u * Mhat)
            / (params.Ku * params.tau_b + params.Kb * params.tau_u * Mhat),
        ]
    )

    fp3 = np.array(
        [
            (1 - Pos[2]) * params.tau_b / (params.tau_b + params.tau_u * Mhat),
            Pos[2]
            * (params.Ku * params.tau_b)
            / (params.Ku * params.tau_b + params.Kb * params.tau_u * Mhat),
            Pos[2]
            * (params.Kb * params.tau_u * Mhat)
            / (params.Ku * params.tau_b + params.Kb * params.tau_u * Mhat),
        ]
    )
    return {
        1: {
            "fp": fp1,
            "jacobian": jacobian(fp1, cj, cs, Mhat, params),
            "Sigma": diffusion_mat(fp1, cs, cj, eps, params),
        },
        2: {
            "fp": fp2,
            "jacobian": jacobian(fp2, cj, cs, Mhat, params),
            "Sigma": diffusion_mat(fp2, cs, cj, eps, params),
        },
        3: {
            "fp": fp3,
            "jacobian": jacobian(fp3, cj, cs, Mhat, params),
            "Sigma": diffusion_mat(fp3, cs, cj, eps, params),
        },
    }


def solve_riccati(A: npt.NDArray, Sigma: npt.NDArray):
    D = Sigma @ Sigma.T

    M = np.array(
        [
            [2 * A[0, 0], 2 * A[0, 1], 2 * A[0, 2], 0, 0, 0],
            [A[1, 0], A[0, 0] + A[0, 1], A[1, 2], A[0, 1], A[0, 2], 0],
            [A[2, 0], A[2, 1], A[0, 0] + A[0, 2], 0, A[0, 1], A[0, 2]],
            [0, 2 * A[1, 0], 0, 2 * A[1, 1], 2 * A[1, 2], 0],
            [0, A[2, 0], A[1, 0], A[2, 1], A[1, 1] + A[2, 2], A[1, 2]],
            [0, 0, 2 * A[2, 0], 0, 2 * A[2, 1], 2 * A[2, 2]],
        ]
    )
    d = np.array([D[0, 0], D[0, 1], D[0, 2], D[1, 1], D[1, 2], D[2, 2]])

    b = np.linalg.solve(M, d)

    B = np.zeros((3, 3))
    B[0, 0] = b[0]
    B[0, 1] = B[1, 0] = b[1]
    B[0, 2] = B[2, 0] = b[2]
    B[1, 1] = b[3]
    B[1, 2] = B[2, 1] = b[4]
    B[2, 2] = b[5]
    return B
