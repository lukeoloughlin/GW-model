import numpy as np
import numba


@numba.njit
def mean_reverting_gbm(
    x0: float, mu: float, sigma: float, r: float, dt: float, nstep: int
):
    out = np.zeros(nstep + 1)
    ts = np.zeros(nstep + 1)
    out[0] = x0

    sqrtdt = np.sqrt(dt)
    x = x0
    for i in range(nstep):
        dW = sqrtdt * np.random.normal()
        drift = r * (mu - x)
        diff = sigma * x
        milst = 0.5 * sigma * diff

        x += dt * drift + dW * diff + milst * (dW * dW - dt)

        out[i + 1] = x
        ts[i + 1] = ts[i] + dt

    return ts, out


@numba.njit
def sample_stable(alpha: float) -> float:
    """See wikipedia entry for stable distributions"""
    U = (np.random.rand() - 0.5) * np.pi
    W = np.random.exponential(1.0)

    sin_term = np.sin(alpha * U)
    cos_term1 = np.cos(U) ** (1.0 / alpha)
    cos_term2 = (np.cos((1.0 - alpha) * U) / W) ** ((1.0 - alpha) / alpha)
    return sin_term * cos_term2 / cos_term1


@numba.njit
def levy_ou_process(
    x0: float, mu: float, sigma: float, r: float, alpha: float, dt: float, nstep: int
):
    out = np.zeros(nstep + 1)
    ts = np.zeros(nstep + 1)
    out[0] = x0

    scale = dt ** (1.0 / alpha)
    x = x0
    for i in range(nstep):
        dL = scale * sample_stable(alpha)
        drift = r * (mu - x)

        x += dt * drift + sigma * dL

        out[i + 1] = x
        ts[i + 1] = ts[i] + dt

    return ts, out
