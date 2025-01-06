import math

import numpy as np
import numpy.typing as npt
from numba import float32
import numba.cuda as cuda

from params import RestrepoParams
from .utils import (
    square,
    bubble_sort_ryr,
    bubble_sort_ryr_3d,
    calculate_rho,
    calculate_Mhat,
    sample_poisson,
    sample_binomial,
)

f32 = np.float32
i32 = np.int32


@cuda.jit(device=True, inline=True)
def update_RyR_rates(
    RyR_rates: npt.NDArray[f32],
    RyR: npt.NDArray[f32],
    cp: npt.NDArray[f32],
    cjsr: npt.NDArray[f32],
    params: RestrepoParams,
    x: i32,
    y: i32,
) -> None:
    """Device func to update RyR rates at position x, y"""
    Mhat = calculate_Mhat(
        calculate_rho(cjsr[x, y], params.K[0], params.rho_inf[0], params.h[0]),
        params.BCSQN[0],
    )

    k12 = params.Ku[0] * square(cp[x, y])  # k12
    k23 = Mhat / params.tau_b[0]  # k23

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
def update_RyR_rates_3d(
    RyR_rates: npt.NDArray[f32],
    RyR: npt.NDArray[f32],
    # corrections: npt.NDArray[f32],
    cp: f32,
    cjsr: f32,
    params: RestrepoParams,
    x: i32,
    y: i32,
    z: i32,
) -> None:
    """Device func to update RyR rates at position x, y"""
    Mhat = calculate_Mhat(
        calculate_rho(cjsr, params.K[0], params.rho_inf[0], params.h[0]),
        params.BCSQN[0],
    )
    cp2 = square(cp)

    k12 = params.Ku[0] * cp2  # k12
    k21 = float32(1) / params.tau_c[0]
    k23 = Mhat / params.tau_b[0]  # k23
    k32 = params.Ku[0] / (params.Kb[0] * params.tau_u[0])
    k34 = float32(1) / params.tau_c[0]
    k43 = params.Kb[0] * cp2  # k43
    k41 = float32(1) / params.tau_u[0]
    k14 = k23

    RyR_rates[x, y, z, 0] = k12 * RyR[x, y, z, 0]  # 1 -> 2
    RyR_rates[x, y, z, 1] = k21 * RyR[x, y, z, 1]  # 2 -> 1
    RyR_rates[x, y, z, 2] = k23 * RyR[x, y, z, 1]  # 2 -> 3
    RyR_rates[x, y, z, 3] = k32 * RyR[x, y, z, 2]  # 3 -> 2
    RyR_rates[x, y, z, 4] = k34 * RyR[x, y, z, 2]  # 3 -> 4
    RyR_rates[x, y, z, 5] = k43 * RyR[x, y, z, 3]  # 4 -> 3
    RyR_rates[x, y, z, 6] = k41 * RyR[x, y, z, 3]  # 4 -> 1
    RyR_rates[x, y, z, 7] = k14 * RyR[x, y, z, 0]  # 1 -> 4; k14 = k23


@cuda.jit(device=True, inline=True)
def update_RyR_rates_3d_tau_leap(
    RyR_rates: npt.NDArray[f32],
    RyR: npt.NDArray[i32],
    cp: f32,
    cjsr: f32,
    params: RestrepoParams,
    x: i32,
    y: i32,
    z: i32,
) -> None:
    """Device func to update RyR rates at position x, y"""
    Mhat = calculate_Mhat(
        calculate_rho(cjsr, params.K[0], params.rho_inf[0], params.h[0]),
        params.BCSQN[0],
    )
    cp2 = square(cp)

    k12 = params.Ku[0] * cp2  # k12
    k21 = float32(1) / params.tau_c[0]
    k23 = Mhat / params.tau_b[0]  # k23
    k32 = params.Ku[0] / (params.Kb[0] * params.tau_u[0])
    k34 = float32(1) / params.tau_c[0]
    k43 = params.Kb[0] * cp2  # k43
    k41 = float32(1) / params.tau_u[0]
    k14 = k23

    RyR_rates[x, y, z, 0] = k12 * RyR[x, y, z, 0]  # 1 -> 2
    RyR_rates[x, y, z, 1] = k21 * RyR[x, y, z, 1]  # 2 -> 1
    RyR_rates[x, y, z, 2] = k23 * RyR[x, y, z, 1]  # 2 -> 3
    RyR_rates[x, y, z, 3] = k32 * RyR[x, y, z, 2]  # 3 -> 2
    RyR_rates[x, y, z, 4] = k34 * RyR[x, y, z, 2]  # 3 -> 4
    RyR_rates[x, y, z, 5] = k43 * RyR[x, y, z, 3]  # 4 -> 3
    RyR_rates[x, y, z, 6] = k41 * RyR[x, y, z, 3]  # 4 -> 1
    RyR_rates[x, y, z, 7] = k14 * RyR[x, y, z, 0]  # 1-> 4; k14 = k23


@cuda.jit(device=True, inline=True)
def RyR_orth_proj_simplex(
    RyR: npt.NDArray[f32], RyR_sorted: npt.NDArray[f32], x: i32, y: i32
) -> None:
    """Device func to perform orthogonal projection of RyR values onto simplex after Euler Maruyama update"""
    # Copy the RyR values into preallocated array and use bubble sort

    bubble_sort_ryr(RyR, RyR_sorted, x, y)

    t = f32(0.0)
    sum_ = f32(0.0)
    for i in range(4):
        sum_ += RyR_sorted[x, y, 3 - i]
        t = (sum_ - f32(1)) / f32(i + 1)
        if i == 3:
            break
        elif t >= RyR_sorted[x, y, 2 - i]:
            break

    RyR[x, y, 0] = max(RyR[x, y, 0] - t, float32(0))
    RyR[x, y, 1] = max(RyR[x, y, 1] - t, float32(0))
    RyR[x, y, 2] = max(RyR[x, y, 2] - t, float32(0))
    RyR[x, y, 3] = max(RyR[x, y, 3] - t, float32(0))


@cuda.jit(device=True, inline=True)
def RyR_orth_proj_simplex_3d(
    RyR: npt.NDArray[f32], RyR_sorted: npt.NDArray[f32], x: i32, y: i32, z: i32
) -> None:
    """Device func to perform orthogonal projection of RyR values onto simplex after Euler Maruyama update"""
    # Copy the RyR values into preallocated array and use bubble sort

    bubble_sort_ryr_3d(RyR, RyR_sorted, x, y, z)

    t = f32(0.0)
    sum_ = f32(0.0)
    for i in range(4):
        sum_ += RyR_sorted[x, y, z, 3 - i]
        t = (sum_ - f32(1)) / f32(i + 1)
        if i == 3:
            break
        elif t >= RyR_sorted[x, y, z, 2 - i]:
            break

    RyR[x, y, z, 0] = max(RyR[x, y, z, 0] - t, f32(0))
    RyR[x, y, z, 1] = max(RyR[x, y, z, 1] - t, f32(0))
    RyR[x, y, z, 2] = max(RyR[x, y, z, 2] - t, f32(0))
    RyR[x, y, z, 3] = max(RyR[x, y, z, 3] - t, f32(0))


@cuda.jit(device=True, inline=True)
def update_RyR_diffusion(
    RyR: npt.NDArray[f32],
    RyR_sorted: npt.NDArray[f32],
    RyR_rates: npt.NDArray[f32],
    dW: npt.NDArray[f32],
    dt: f32,
    x: i32,
    y: i32,
) -> None:
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


@cuda.jit(device=True, inline=True)
def update_RyR_diffusion_3d(
    RyR: npt.NDArray[f32],
    # RyR_sorted: npt.NDArray[f32],
    RyR_rates: npt.NDArray[f32],
    # corrections: npt.NDArray[f32],
    dW: npt.NDArray[f32],
    dt: f32,
    x: i32,
    y: i32,
    z: i32,
) -> None:
    """Euler Maruyama step for RyR model with reflecting boundary conditions."""
    drift1 = (
        RyR_rates[x, y, z, 1]
        + RyR_rates[x, y, z, 6]
        - (RyR_rates[x, y, z, 0] + RyR_rates[x, y, z, 7])
    )  # q21 + q41 - (q12 + q14)
    drift2 = (
        RyR_rates[x, y, z, 0]
        + RyR_rates[x, y, z, 3]
        - (RyR_rates[x, y, z, 1] + RyR_rates[x, y, z, 2])
    )  # q12 + q32 - (q21 + q23)
    drift3 = (
        RyR_rates[x, y, z, 2]
        + RyR_rates[x, y, z, 5]
        - (RyR_rates[x, y, z, 3] + RyR_rates[x, y, z, 4])
    )  # q23 + q43 - (q32 + q34)
    drift4 = (
        RyR_rates[x, y, z, 4]
        + RyR_rates[x, y, z, 7]
        - (RyR_rates[x, y, z, 5] + RyR_rates[x, y, z, 6])
    )

    sigma12 = float32(0.1) * math.sqrt(
        max(RyR_rates[x, y, z, 0] + RyR_rates[x, y, z, 1], f32(0))
    )  # q12 + q21
    sigma23 = float32(0.1) * math.sqrt(
        max(RyR_rates[x, y, z, 2] + RyR_rates[x, y, z, 3], f32(0))
    )  # q23 + q32
    sigma34 = float32(0.1) * math.sqrt(
        max(RyR_rates[x, y, z, 4] + RyR_rates[x, y, z, 5], f32(0))
    )  # q34 + q43
    sigma14 = float32(0.1) * math.sqrt(
        max(RyR_rates[x, y, z, 6] + RyR_rates[x, y, z, 7], f32(0))
    )  # q41 + q14

    RyR[x, y, z, 0] += dt * drift1 + sigma12 * dW[x, y, z, 0] + sigma14 * dW[x, y, z, 3]
    RyR[x, y, z, 1] += dt * drift2 - sigma12 * dW[x, y, z, 0] + sigma23 * dW[x, y, z, 1]
    RyR[x, y, z, 2] += dt * drift3 - sigma23 * dW[x, y, z, 1] + sigma34 * dW[x, y, z, 2]
    RyR[x, y, z, 3] += dt * drift4 - sigma14 * dW[x, y, z, 3] - sigma34 * dW[x, y, z, 2]
    # RyR[x, y, z, 3] = float32(1.0) - (
    #    RyR[x, y, z, 0] + RyR[x, y, z, 1] + RyR[x, y, z, 2]
    # )

    # RyR_orth_proj_simplex_3d(RyR, RyR_sorted, x, y, z)


@cuda.jit(device=True, inline=True)
def update_RyR_tau_leap_3d(
    RyR: npt.NDArray[i32],
    RyR_rates: npt.NDArray[f32],
    dt: f32,
    rng_states,
    x: int,
    y: int,
    z: int,
    rng_idx: int,
) -> None:
    """Euler Maruyama step for RyR model with reflecting boundary conditions."""

    # Outgoings from state 1
    lambda_ = RyR_rates[x, y, z, 0] + RyR_rates[x, y, z, 7]
    Nout1 = (
        min(sample_poisson(dt * lambda_, rng_states, rng_idx), RyR[x, y, z, 0])
        if RyR[x, y, z, 0] > 0
        else i32(0)
    )
    p = RyR_rates[x, y, z, 0] / lambda_
    N12 = sample_binomial(Nout1, p, rng_states, rng_idx) if Nout1 > 0 else i32(0)
    N14 = Nout1 - N12

    # Outgoings from state 2
    lambda_ = RyR_rates[x, y, z, 1] + RyR_rates[x, y, z, 2]
    Nout2 = (
        min(sample_poisson(dt * lambda_, rng_states, rng_idx), RyR[x, y, z, 1])
        if RyR[x, y, z, 1] > 0
        else i32(0)
    )
    p = RyR_rates[x, y, z, 1] / lambda_
    N21 = sample_binomial(Nout2, p, rng_states, rng_idx) if Nout2 > 0 else i32(0)
    N23 = Nout2 - N21

    # Outgoings from state 3
    lambda_ = RyR_rates[x, y, z, 3] + RyR_rates[x, y, z, 4]
    Nout3 = (
        min(sample_poisson(dt * lambda_, rng_states, rng_idx), RyR[x, y, z, 2])
        if RyR[x, y, z, 2] > 0
        else 0
    )
    p = RyR_rates[x, y, z, 3] / lambda_
    N32 = sample_binomial(Nout3, p, rng_states, rng_idx) if Nout3 > 0 else i32(0)
    N34 = Nout3 - N32

    # Outgoings from state 4
    lambda_ = RyR_rates[x, y, z, 5] + RyR_rates[x, y, z, 6]
    Nout4 = (
        min(sample_poisson(dt * lambda_, rng_states, rng_idx), RyR[x, y, z, 3])
        if RyR[x, y, z, 3] > 0
        else 0
    )
    p = RyR_rates[x, y, z, 5] / lambda_
    N43 = sample_binomial(Nout4, p, rng_states, rng_idx) if Nout4 > 0 else i32(0)
    N41 = Nout4 - N43

    RyR[x, y, z, 0] += N21 + N41 - N12 - N14
    RyR[x, y, z, 1] += N12 + N32 - N21 - N23
    RyR[x, y, z, 2] += N23 + N43 - N32 - N34
    RyR[x, y, z, 3] += N14 + N34 - N43 - N41


def RyR_stationary(
    cp: npt.NDArray, params: RestrepoParams, single_precision: bool = True
) -> npt.NDArray:
    log_cp_K = np.log(cp) - np.log(params.K)
    hill_fn = params.rho_inf / (1.0 + np.exp(23.0 * log_cp_K))
    Mhat = (np.sqrt(1.0 + 8.0 * hill_fn * params.BCSQN) - 1.0) / (
        4.0 * hill_fn * params.BCSQN
    )

    k12 = params.Ku * cp**2  # k12
    k21 = 1 / params.tau_c
    k23 = Mhat / params.tau_b  # k23
    k34 = 1 / params.tau_c
    k43 = params.Kb * cp**2  # k43
    k32 = k12 / (params.tau_u * k43)  # k32 = k41 * k12 / k43

    pi2_un = k12 / k21
    pi3_un = (k23 / k32) * pi2_un
    pi4_un = (k34 / k43) * pi3_un

    norm = 1.0 + pi2_un + pi3_un + pi4_un

    RyR = np.zeros((*cp.shape, 4))
    RyR[..., 0] = 1.0 / norm
    RyR[..., 1] = pi2_un / norm
    RyR[..., 2] = pi3_un / norm
    RyR[..., 3] = pi4_un / norm

    if single_precision:
        return RyR.astype(f32)
    else:
        return RyR
