import numpy as np
import numpy.typing as npt
import numba


from utils import _Mhat, _rho
from params import RestrepoParams


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
    k14 = _Mhat(_rho(cj, params), params.BCSQN)
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
def ryr_gillespie_and_martingale(
    init: npt.NDArray, cs: float, cj: float, params: RestrepoParams, nstep: int
):
    t_out = np.zeros(nstep + 1)
    out = np.zeros((nstep + 1, 4), dtype=np.int64)
    integral = np.zeros((nstep + 1, 4), dtype=float)

    state = np.copy(init)
    rates = np.zeros(8, dtype=float)
    cdf = np.zeros(8, dtype=float)
    out[0, :] = init

    Ku = params.Ku
    Kb = params.Kb
    k14 = _Mhat(_rho(cj, params), params.BCSQN)
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
        integral[i + 1, 0] = integral[i, 0] + dt * (
            rates[1] + rates[6] - (rates[0] + rates[7])
        )
        integral[i + 1, 1] = integral[i, 1] + dt * (
            rates[0] + rates[3] - (rates[1] + rates[2])
        )
        integral[i + 1, 2] = integral[i, 2] + dt * (
            rates[2] + rates[5] - (rates[4] + rates[3])
        )
        integral[i + 1, 3] = integral[i, 3] + dt * (
            rates[7] + rates[4] - (rates[6] + rates[5])
        )

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
    return t_out, out, (out - integral) / N


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
    k14 = _Mhat(_rho(cj, params), params.BCSQN)
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
    k14 = _Mhat(_rho(cj, params), params.BCSQN)
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


def Po_stationary_logp(N: int, cs: float, cj: float, params: RestrepoParams):
    log_probs = np.zeros(N + 1)
    Mhat = _Mhat(_rho(cj, params), params.BCSQN)
    kr = params.tau_ps * params.Jmax / params.vp

    log_t1 = np.log(
        params.tau_b * params.Ku + params.tau_u * Mhat * params.Kb
    ) - np.log(params.tau_b + params.tau_u * Mhat)

    log_N_fact = np.sum(np.log(np.arange(1, N + 1)))
    log_n_fact = 0.0
    log_N_minus_n_fact = log_N_fact

    log_t2 = 0.0

    for i in range(N):
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


@numba.njit
def ryr_gillespie_homog(
    init: npt.NDArray,
    cs: float,
    cj: float,
    cp_init: float,
    params: RestrepoParams,
    nstep: int,
    time_scale_param: float = 1.0,
    dt_min: float = 1e-3,
):
    t_out = np.zeros(nstep + 1)
    ryr_out = np.zeros((nstep + 1, 4), dtype=np.int64)
    cp_out = np.zeros((nstep + 1,), dtype=float)

    state = np.copy(init)
    cp = cp_init
    rates = np.zeros(8, dtype=float)
    cdf = np.zeros(8, dtype=float)
    ryr_out[0, :] = init
    cp_out[0] = cp_init

    Ku = params.Ku
    Kb = params.Kb
    k14 = _Mhat(_rho(cj, params), params.BCSQN)
    k21 = 1.0 / params.tau_c
    k23 = k14
    k41 = 1 / params.tau_u
    k34 = 1 / params.tau_c
    k32 = k41 * Ku / Kb

    tau_ps = params.tau_ps
    Jmax = params.Jmax
    vp = params.vp
    N = float(init.sum())

    t = 0.0
    for i in range(nstep):
        Po = float(state[1] + state[2]) / N
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
        if dt < dt_min:
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
        else:
            dt = dt_min
        t += dt
        cp += dt * (Po * Jmax * (cj - cp) / vp + (cs - cp) / tau_ps) / time_scale_param
        t_out[i + 1] = t
        ryr_out[i + 1, :] = state
        cp_out[i + 1] = cp
    return t_out, ryr_out, cp_out
