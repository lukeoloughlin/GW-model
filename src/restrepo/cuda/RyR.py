import math

import numpy as np
import numpy.typing as npt
from numba import float32
import numba.cuda as cuda

from params import RestrepoParams
from src.restrepo.cuda.utils import (
    square,
    bubble_sort_ryr,
    calculate_rho,
    calculate_Mhat,
)


@cuda.jit(device=True, inline=True)
def update_RyR_rates(
    RyR_rates: npt.NDArray,
    RyR: npt.NDArray,
    cp: npt.NDArray,
    cjsr: npt.NDArray,
    params: RestrepoParams,
    # Ku,
    # Kb,
    # tau_u,
    # tau_b,
    # tau_c,
    # BCSQN,
    # rho_inf,
    # K,
    x: int,
    y: int,
):
    """Device func to update RyR rates at position x, y"""
    Mhat = calculate_Mhat(
        calculate_rho(cjsr[x, y], params.K[0], params.rho_inf[0]), params.BCSQN[0]
    )

    k12 = params.Ku[0] * square(cp[x, y])  # k12
    k23 = Mhat * cp[x, y] / params.tau_b[0]  # k23

    k43 = params.Kb[0] * square(cp[x, y])  # k43
    k32 = k12 / (k43 * params.tau_u[0])  # k32 = k41 * k12 / k43

    RyR_rates[x, y, 0] = k12 * RyR[x, y, 0]  # 1 -> 2
    RyR_rates[x, y, 1] = RyR[x, y, 1] / params.tau_c[0]  # 2 -> 1; k21 = _1_tau_c
    RyR_rates[x, y, 2] = k23 * RyR[x, y, 1]  # 2 -> 3
    RyR_rates[x, y, 3] = k32 * RyR[x, y, 2]  # 3 -> 2
    RyR_rates[x, y, 4] = RyR[x, y, 2] / params.tau_c[0]  # 3 -> 4; k34 = _1_tau_c
    RyR_rates[x, y, 5] = k43 * RyR[x, y, 3]  # 4 -> 3
    RyR_rates[x, y, 6] = RyR[x, y, 3] / params.tau_u[0]  # 4 -> 1; k41 = _1_tau_u
    RyR_rates[x, y, 7] = k23 * RyR[x, y, 0]  # 1-> 4; k14 = k23


@cuda.jit(device=True, inline=True)
def RyR_orth_proj_simplex(RyR, RyR_sorted, x, y):
    """Device func to perform orthogonal projection of RyR values onto simplex after Euler Maruyama update"""
    # Copy the RyR values into preallocated array and use bubble sort

    bubble_sort_ryr(RyR, RyR_sorted, x, y)

    lambda_ = float32(0.0)
    sum_ = float32(1.0)
    for i in range(4):
        if sum_ - (float32(4.0 - i)) * RyR_sorted[x, y, i] < float32(1.0):
            lambda_ = (sum_ - 1.0) / (float32(4.0 - i))
            break
        else:
            sum_ -= RyR_sorted[x, y, i]

    RyR[x, y, 0] = max(RyR[x, y, 0] - lambda_, float32(0.0))
    RyR[x, y, 1] = max(RyR[x, y, 1] - lambda_, float32(0.0))
    RyR[x, y, 2] = max(RyR[x, y, 2] - lambda_, float32(0.0))
    RyR[x, y, 3] = max(RyR[x, y, 3] - lambda_, float32(0.0))


@cuda.jit(device=True, inline=True)
def update_RyR_diffusion(RyR, RyR_sorted, RyR_rates, dW, dt, x, y):
    """Euler Maruyama step for RyR model with reflecting boundary conditions."""
    drift1 = (
        RyR_rates[x, y, 1]
        + RyR_rates[x, y, 6]
        - (RyR_rates[x, y, 0] + RyR_rates[x, y, 7])
    )  # q21 + q41 - (q12 + q14)
    drift2 = (
        RyR_rates[x, y, 0]
        + RyR_rates[x, y, 3]
        - (RyR_rates[x, y, 1] + RyR_rates[x, y, 2])
    )  # q12 + q32 - (q21 + q23)
    drift3 = (
        RyR_rates[x, y, 2]
        + RyR_rates[x, y, 5]
        - (RyR_rates[x, y, 3] + RyR_rates[x, y, 4])
    )  # q23 + q43 - (q32 + q34)

    sigma12 = float32(0.1) * math.sqrt(
        RyR_rates[x, y, 0] + RyR_rates[x, y, 1]
    )  # q12 + q21
    sigma23 = float32(0.1) * math.sqrt(
        RyR_rates[x, y, 2] + RyR_rates[x, y, 3]
    )  # q23 + q32
    sigma34 = float32(0.1) * math.sqrt(
        RyR_rates[x, y, 4] + RyR_rates[x, y, 5]
    )  # q34 + q43
    sigma14 = float32(0.1) * math.sqrt(
        RyR_rates[x, y, 6] + RyR_rates[x, y, 7]
    )  # q41 + q14

    RyR[x, y, 0] += dt * drift1 + sigma12 * dW[x, y, 0] + sigma14 * dW[x, y, 3]
    RyR[x, y, 1] += dt * drift2 - sigma12 * dW[x, y, 0] + sigma23 * dW[x, y, 1]
    RyR[x, y, 2] += dt * drift3 - sigma23 * dW[x, y, 1] + sigma34 * dW[x, y, 2]
    RyR[x, y, 3] = float32(1.0) - (RyR[x, y, 0] + RyR[x, y, 1] + RyR[x, y, 2])

    RyR_orth_proj_simplex(RyR, RyR_sorted, x, y)


# @cuda.jit
# def RyR_kernel(
#    RyR,
#    RyR_rates,
#    RyR_tmp,
#    cp,
#    cjsr,
#    dW,
#    eps,
#    dt,
#    sqrtdt,
#    Ku,
#    Kb,
#    _1_tau_u,
#    _1_tau_b,
#    _1_tau_c,
#    BCSQN,
#    rho_inf,
#    K,
#    rng_states,
# ):
#    x, y = cuda.grid(2)

#    N1, N2, _ = RyR.shape

#    tid = y * N1 + x

#    if x < N1 and y < N2:

#        update_RyR_rates(
#            RyR_rates,
#            RyR,
#            cp,
#            cjsr,
#            Ku,
#            Kb,
#            _1_tau_u,
#            _1_tau_b,
#            _1_tau_c,
#            BCSQN,
#            rho_inf,
#            K,
#            x,
#            y,
#        )

# update normals
#        for i in range(4):
#            dW[x, y, i] = sqrtdt * xoroshiro128p_normal_float32(rng_states, tid)

#        update_RyR_diffusion(RyR, RyR_tmp, RyR_rates, dW, eps, dt, x, y)


def test_RyR_cpu(
    RyR: npt.NDArray, RyR_rates: npt.NDArray, dW: npt.NDArray, dt: float
) -> npt.NDArray:
    drift1 = (
        RyR_rates[..., 1] + RyR_rates[..., 6] - (RyR_rates[..., 0] + RyR_rates[..., 7])
    )  # q21 + q41 - (q12 + q14)
    drift2 = (
        RyR_rates[..., 0] + RyR_rates[..., 3] - (RyR_rates[..., 1] + RyR_rates[..., 2])
    )  # q12 + q32 - (q21 + q23)
    drift3 = (
        RyR_rates[..., 2] + RyR_rates[..., 5] - (RyR_rates[..., 3] + RyR_rates[..., 4])
    )  # q23 + q43 - (q32 + q34)

    sigma12 = 0.1 * np.sqrt(RyR_rates[..., 0] + RyR_rates[..., 1])  # q12 + q21
    sigma23 = 0.1 * np.sqrt(RyR_rates[..., 2] + RyR_rates[..., 3])  # q23 + q32
    sigma34 = 0.1 * np.sqrt(RyR_rates[..., 4] + RyR_rates[..., 5])  # q34 + q43
    sigma14 = 0.1 * np.sqrt(RyR_rates[..., 6] + RyR_rates[..., 7])  # q41 + q14

    RyR_out = np.copy(RyR)

    RyR_out[..., 0] += dt * drift1 + sigma12 * dW[..., 0] + sigma14 * dW[..., 3]
    RyR_out[..., 1] += dt * drift2 - sigma12 * dW[..., 0] + sigma23 * dW[..., 1]
    RyR_out[..., 2] += dt * drift3 - sigma23 * dW[..., 1] + sigma34 * dW[..., 2]
    RyR_out[..., 3] = 1.0 - (RyR_out[..., 0] + RyR_out[..., 1] + RyR_out[..., 2])

    RyR_sorted = np.sort(RyR_out, axis=-1)

    for x in range(RyR.shape[0]):
        for y in range(RyR.shape[0]):
            lambda_ = 0.0
            sum_ = 1.0
            for i in range(4):
                if sum_ - (4.0 - i) * RyR_sorted[x, y, i] < 1:
                    lambda_ = (sum_ - 1.0) / (4 - i)
                    break
                else:
                    sum_ -= RyR_sorted[x, y, i]
            RyR_out[x, y, 0] = max(RyR_out[x, y, 0] - lambda_, 0.0)
            RyR_out[x, y, 1] = max(RyR_out[x, y, 1] - lambda_, 0.0)
            RyR_out[x, y, 2] = max(RyR_out[x, y, 2] - lambda_, 0.0)
            RyR_out[x, y, 3] = max(RyR_out[x, y, 3] - lambda_, 0.0)

    return RyR_out
