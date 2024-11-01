import math

import numpy as np
from numba import void, f4, float32, size_t  # f4 = float32
import numba.cuda as cuda
from numba.cuda.random import (
    xoroshiro128p_normal_float32,
    create_xoroshiro128p_states,
    xoroshiro128p_type,
)


@cuda.jit(
    void(
        f4[:, :, :],
        f4[:, :, :],
        f4[:, :],
        f4,
        f4,
        f4,
        f4,
        f4,
        f4,
        f4,
        f4,
        size_t,
        size_t,
    ),
    device=True,
    inline=True,
)
def update_RyR_rates(
    RyR_rates,
    RyR,
    cp,
    Ku,
    Kb,
    _1_tau_u,
    _1_tau_b,
    _1_tau_c,
    BCSQN,
    rho_inf,
    K,
    x,
    y,
):
    """Device func to update RyR rates at position x, y"""
    log_cp_K = math.log(cp[x, y]) - math.log(K)
    hill_fn = rho_inf / (float32(1.0) + math.exp(float32(23.0) * log_cp_K))
    Mhat = (math.sqrt(float32(1.0) + float32(8.0) * hill_fn * BCSQN) - float32(1.0)) / (
        float32(4.0) * hill_fn * BCSQN
    )

    k12 = Ku * cp[x, y] * cp[x, y]  # k12
    k23 = Mhat * _1_tau_b  # k23

    k43 = Kb * cp[x, y] * cp[x, y]  # k43
    k32 = _1_tau_u * k12 / k43  # k32 = k41 * k12 / k43

    RyR_rates[x, y, 0] = k12 * RyR[x, y, 0]  # 1 -> 2
    RyR_rates[x, y, 1] = _1_tau_c * RyR[x, y, 1]  # 2 -> 1; k21 = _1_tau_c
    RyR_rates[x, y, 2] = k23 * RyR[x, y, 1]  # 2 -> 3
    RyR_rates[x, y, 3] = k32 * RyR[x, y, 2]  # 3 -> 2
    RyR_rates[x, y, 4] = _1_tau_c * RyR[x, y, 2]  # 3 -> 4; k34 = _1_tau_c
    RyR_rates[x, y, 5] = k43 * RyR[x, y, 3]  # 4 -> 3
    RyR_rates[x, y, 6] = _1_tau_u * RyR[x, y, 3]  # 4 -> 1; k41 = _1_tau_u
    RyR_rates[x, y, 7] = k23 * RyR[x, y, 0]  # 1-> 4; k14 = k23


@cuda.jit(void(f4[:, :, :], f4[:, :, :], size_t, size_t), device=True, inline=True)
def bubble_sort(RyR, sorted_arr, x, y):
    sorted_arr[x, y, 0] = RyR[x, y, 0]
    sorted_arr[x, y, 1] = RyR[x, y, 1]
    sorted_arr[x, y, 2] = RyR[x, y, 2]
    sorted_arr[x, y, 3] = RyR[x, y, 3]

    swapped = False
    tmp = float32(0)
    for i in range(3):
        swapped = False
        for j in range(3 - i):
            if sorted_arr[x, y, j] > sorted_arr[x, y, j + 1]:
                tmp = sorted_arr[x, y, j]
                sorted_arr[x, y, j] = sorted_arr[x, y, j + 1]
                sorted_arr[x, y, j + 1] = tmp
                swapped = True
        # If no two elements were swapped, then break
        if not swapped:
            break


@cuda.jit(
    void(f4[:, :, :], f4[:, :, :], size_t, size_t),
    device=True,
    inline=True,
)
def RyR_orth_proj_simplex(RyR, sorted_arr, x, y):
    """Device func to perform orthogonal projection of RyR values onto simplex after Euler Maruyama update"""
    # Copy the RyR values into preallocated array and use bubble sort

    bubble_sort(RyR, sorted_arr, x, y)

    lambda_ = float32(0.0)
    sum_ = float32(1.0)
    for i in range(4):
        if sum_ - (float32(4.0) - i) * sorted_arr[x, y, i] < float32(1):
            lambda_ = (sum_ - 1.0) / (4 - i)
            break
        else:
            sum_ -= sorted_arr[x, y, i]

    RyR[x, y, 0] = max(RyR[x, y, 0] - lambda_, float32(0.0))
    RyR[x, y, 1] = max(RyR[x, y, 1] - lambda_, float32(0.0))
    RyR[x, y, 2] = max(RyR[x, y, 2] - lambda_, float32(0.0))
    RyR[x, y, 3] = max(RyR[x, y, 3] - lambda_, float32(0.0))


@cuda.jit(
    void(f4[:, :, :], f4[:, :, :], f4[:, :, :], f4[:, :, :], f4, f4, size_t, size_t),
    device=True,
    inline=True,
)
def update_RyR_diffusion(RyR, RyR_tmp, RyR_rates, dW, eps, dt, x, y):
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

    sigma12 = eps * math.sqrt(RyR_rates[x, y, 0] + RyR_rates[x, y, 1])  # q12 + q21
    sigma23 = eps * math.sqrt(RyR_rates[x, y, 2] + RyR_rates[x, y, 3])  # q23 + q32
    sigma34 = eps * math.sqrt(RyR_rates[x, y, 4] + RyR_rates[x, y, 5])  # q34 + q43
    sigma14 = eps * math.sqrt(RyR_rates[x, y, 6] + RyR_rates[x, y, 7])  # q41 + q14

    RyR[x, y, 0] += dt * drift1 + sigma12 * dW[x, y, 0] + sigma14 * dW[x, y, 3]
    RyR[x, y, 1] += dt * drift2 - sigma12 * dW[x, y, 0] + sigma23 * dW[x, y, 1]
    RyR[x, y, 2] += dt * drift3 - sigma23 * dW[x, y, 1] + sigma34 * dW[x, y, 2]
    RyR[x, y, 3] = float32(1.0) - (RyR[x, y, 0] + RyR[x, y, 1] + RyR[x, y, 2])

    RyR_orth_proj_simplex(RyR, RyR_tmp, x, y)


@cuda.jit(
    void(
        f4[:, :, :],
        f4[:, :, :],
        f4[:, :, :],
        f4[:, :],
        f4[:, :, :],
        f4,
        f4,
        f4,
        f4,
        f4,
        f4,
        f4,
        f4,
        f4,
        f4,
        f4,
        xoroshiro128p_type[:],
    )
)
def RyR_kernel(
    RyR,
    RyR_rates,
    RyR_tmp,
    cp,
    dW,
    eps,
    dt,
    sqrtdt,
    Ku,
    Kb,
    _1_tau_u,
    _1_tau_b,
    _1_tau_c,
    BCSQN,
    rho_inf,
    K,
    rng_states,
):
    x, y = cuda.grid(2)

    N1, N2, _ = RyR.shape

    tid = y * N1 + x

    if x < N1 and y < N2:

        update_RyR_rates(
            RyR_rates,
            RyR,
            cp,
            Ku,
            Kb,
            _1_tau_u,
            _1_tau_b,
            _1_tau_c,
            BCSQN,
            rho_inf,
            K,
            x,
            y,
        )

        # update normals
        for i in range(4):
            dW[x, y, i] = sqrtdt * xoroshiro128p_normal_float32(rng_states, tid)

        update_RyR_diffusion(RyR, RyR_tmp, RyR_rates, dW, eps, dt, x, y)


def call_RyR_kernel(RyR, cp, eps, dt, nstep=1, threadsperblock=256, seed=0):
    x, y, _ = RyR.shape

    RyR_dev = cuda.to_device(RyR)
    cp_dev = cuda.to_device(cp)

    RyR_tmp = cuda.device_array((x, y, 4), dtype=np.float32)
    RyR_rates = cuda.device_array((x, y, 8), dtype=np.float32)
    dW = cuda.device_array((x, y, 4), dtype=np.float32)

    rng_states = create_xoroshiro128p_states(x * y, seed)

    tpb = (threadsperblock, threadsperblock)
    blockspergrid_x = math.ceil(x / threadsperblock)
    blockspergrid_y = math.ceil(y / threadsperblock)
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    RyR_fwd_kernel[blockspergrid, tpb](
        RyR_dev,
        RyR_rates,
        RyR_tmp,
        cp_dev,
        dW,
        np.float32(eps),
        np.float32(dt),
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        10.0,
        rng_states,
        nstep,
    )
    cuda.synchronize()
    RyR_dev.copy_to_host(RyR)
