from typing import Any

import numpy as np
import numpy.typing as npt
from numba import float32, int32, njit
import numba.cuda as cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

from params import RestrepoParams
from .utils import cube, pow4

# This does the LCC stuff. This should be called in a separate kernel because it only needs to work on the boundary of the domain

# 1 = C2
# 2 = C1
# 3 = I1Ca
# 4 = I2Ca
# 5 = I1Ba
# 6 = I2Ba
# 7 = O

f32 = np.float32
i32 = np.int32
RNG_state = Any

LCC_params = tuple[
    np.floating,
    np.floating,
    np.floating,
    np.floating,
    np.floating,
    np.floating,
    np.floating,
    np.floating,
]


@cuda.jit(device=True, inline=True)
def update_LCC_probs(
    LCC_probs: npt.NDArray[f32],
    LCC: npt.NDArray[i32],
    cp: f32,
    dt: f32,
    alpha: f32,
    beta: f32,
    k3: f32,
    k3_: f32,
    k5_: f32,
    k6_: f32,
    Pr: f32,
    Ps: f32,
    R: f32,
    V: f32,
    params: RestrepoParams,
    idx: i32,
):
    """Does euler step for Kolmogorov equations. Assumed that the LCCs are stored in a 1d array and the indexing semantics are dealt with elsewhere"""
    cptilde_cp3 = cube(params.cp_tilde[0] / cp)
    TCa = (
        float32(78.0329) + float32(0.1) * pow4(float32(1.0) + cp / params.cp_bar[0])
    ) / (float32(1.0) + pow4(cp / params.cp_bar[0]))
    tauCa = (R - TCa) * Pr + TCa

    s1 = float32(0.02) / (float32(1.0) + cptilde_cp3)
    k1 = float32(0.03) / (float32(1.0) + cptilde_cp3)
    if V < float32(-40.0):
        k5 = k5_
        k6 = k6_
    else:
        k5 = (float32(1.0) - Ps) / tauCa
        k6 = Ps / (tauCa * (float32(1.0) + cptilde_cp3))

    s2 = s1 * params.k2[0] * params.r1[0] / (k1 * params.r2[0])
    s2_ = params.s1_[0] * params.k2_[0] * params.r1[0] / (params.k1_[0] * params.r2[0])
    k4 = k3 * (alpha / beta) * (k1 / params.k2[0]) * (k5 / k6)
    k4_ = k3_ * (alpha / beta) * (params.k1_[0] / params.k2_[0]) * (k5_ / k6_)

    for j in range(4):
        if LCC[idx, j] == 1:
            LCC_probs[idx, j, 0] = float32(1.0) - dt * (k6 + k6_ + alpha)  # C2
            LCC_probs[idx, j, 1] = dt * alpha  # C1
            LCC_probs[idx, j, 2] = float32(0.0)  # I1Ca
            LCC_probs[idx, j, 3] = dt * k6  # I2Ca
            LCC_probs[idx, j, 4] = float32(0.0)  # I1Ba
            LCC_probs[idx, j, 5] = dt * k6_  # I2Ba
            LCC_probs[idx, j, 6] = float32(0.0)  # O
        elif LCC[idx, j] == 2:
            LCC_probs[idx, j, 0] = dt * beta  # C2
            LCC_probs[idx, j, 1] = float32(1.0) - dt * (
                params.r1[0] + beta + k1 + params.k1_[0]
            )  # C1
            LCC_probs[idx, j, 2] = dt * k1  # I1Ca
            LCC_probs[idx, j, 3] = float32(0.0)  # I2Ca
            LCC_probs[idx, j, 4] = dt * params.k1_[0]  # I1Ba
            LCC_probs[idx, j, 5] = float32(0.0)  # I2Ba
            LCC_probs[idx, j, 6] = dt * params.r1[0]  # O
        elif LCC[idx, j] == 3:
            LCC_probs[idx, j, 0] = float32(0.0)  # C2
            LCC_probs[idx, j, 1] = dt * params.k2[0]  # C1
            LCC_probs[idx, j, 2] = float32(1.0) - dt * (params.k2[0] + k3 + s2)  # I1Ca
            LCC_probs[idx, j, 3] = dt * k3  # I2Ca
            LCC_probs[idx, j, 4] = float32(0.0)  # I1Ba
            LCC_probs[idx, j, 5] = float32(0.0)  # I2Ba
            LCC_probs[idx, j, 6] = dt * s2  # O
        elif LCC[idx, j] == 4:
            LCC_probs[idx, j, 0] = dt * k5  # C2
            LCC_probs[idx, j, 1] = float32(0.0)  # C1
            LCC_probs[idx, j, 2] = dt * k4  # I1Ca
            LCC_probs[idx, j, 3] = float32(1.0) - dt * (k4 + k5)  # I2Ca
            LCC_probs[idx, j, 4] = float32(0.0)  # I1Ba
            LCC_probs[idx, j, 5] = float32(0.0)  # I2Ba
            LCC_probs[idx, j, 6] = float32(0.0)  # O
        elif LCC[idx, j] == 5:
            LCC_probs[idx, j, 0] = float32(0.0)  # C2
            LCC_probs[idx, j, 1] = dt * params.k2_[0]  # C1
            LCC_probs[idx, j, 2] = float32(0.0)  # I1Ca
            LCC_probs[idx, j, 3] = float32(0.0)  # I2Ca
            LCC_probs[idx, j, 4] = float32(1.0) - dt * (
                params.k2_[0] + k3_ + s2_
            )  # I1Ba
            LCC_probs[idx, j, 5] = dt * k3_  # I2Ba
            LCC_probs[idx, j, 6] = dt * s2_  # O
        elif LCC[idx, j] == 6:
            LCC_probs[idx, j, 0] = dt * k5_  # C2
            LCC_probs[idx, j, 1] = float32(0.0)  # C1
            LCC_probs[idx, j, 2] = float32(0.0)  # I1Ca
            LCC_probs[idx, j, 3] = float32(0.0)  # I2Ca
            LCC_probs[idx, j, 4] = dt * k4_  # I1Ba
            LCC_probs[idx, j, 5] = float32(1.0) - dt * (k5_ + k4_)  # I2Ba
            LCC_probs[idx, j, 6] = float32(0.0)  # O
        else:
            LCC_probs[idx, j, 0] = float32(0.0)  # C2
            LCC_probs[idx, j, 1] = dt * params.r2[0]  # C1
            LCC_probs[idx, j, 2] = dt * s1  # I1Ca
            LCC_probs[idx, j, 3] = float32(0.0)  # I2Ca
            LCC_probs[idx, j, 4] = dt * params.s1_[0]  # I1Ba
            LCC_probs[idx, j, 5] = float32(0.0)  # I2Ba
            LCC_probs[idx, j, 6] = float32(1.0) - dt * (
                params.r2[0] + s1 + params.s1_[0]
            )  # O


@cuda.jit(device=True, inline=True)
def update_LCC_probs_3d(
    LCC_probs: npt.NDArray[f32],
    LCC: npt.NDArray[i32],
    cp: f32,
    dt: f32,
    alpha: f32,
    beta: f32,
    k3: f32,
    k3_: f32,
    k5_: f32,
    k6_: f32,
    Pr: f32,
    Ps: f32,
    R: f32,
    V: f32,
    params: RestrepoParams,
    x: i32,
    y: i32,
    z: i32,
):
    """Does euler step for Kolmogorov equations. Assumed that the LCCs are stored in a 1d array and the indexing semantics are dealt with elsewhere"""
    cptilde_cp3 = cube(params.cp_tilde[0] / cp)
    TCa = (
        float32(78.0329) + float32(0.1) * pow4(float32(1.0) + cp / params.cp_bar[0])
    ) / (float32(1.0) + pow4(cp / params.cp_bar[0]))
    tauCa = (R - TCa) * Pr + TCa

    s1 = float32(0.02) / (float32(1.0) + cptilde_cp3)
    k1 = float32(0.03) / (float32(1.0) + cptilde_cp3)
    if V < float32(-40.0):
        k5 = k5_
        k6 = k6_
    else:
        k5 = (float32(1.0) - Ps) / tauCa
        k6 = Ps / (tauCa * (float32(1.0) + cptilde_cp3))

    s2 = s1 * params.k2[0] * params.r1[0] / (k1 * params.r2[0])
    s2_ = params.s1_[0] * params.k2_[0] * params.r1[0] / (params.k1_[0] * params.r2[0])
    k4 = k3 * (alpha / beta) * (k1 / params.k2[0]) * (k5 / k6)
    k4_ = k3_ * (alpha / beta) * (params.k1_[0] / params.k2_[0]) * (k5_ / k6_)

    for j in range(4):
        if LCC[x, y, z, j] == 1:
            LCC_probs[x, y, z, j, 0] = float32(1.0) - dt * (k6 + k6_ + alpha)  # C2
            LCC_probs[x, y, z, j, 1] = dt * alpha  # C1
            LCC_probs[x, y, z, j, 2] = float32(0.0)  # I1Ca
            LCC_probs[x, y, z, j, 3] = dt * k6  # I2Ca
            LCC_probs[x, y, z, j, 4] = float32(0.0)  # I1Ba
            LCC_probs[x, y, z, j, 5] = dt * k6_  # I2Ba
            LCC_probs[x, y, z, j, 6] = float32(0.0)  # O
        elif LCC[x, y, z, j] == 2:
            LCC_probs[x, y, z, j, 0] = dt * beta  # C2
            LCC_probs[x, y, z, j, 1] = float32(1.0) - dt * (
                params.r1[0] + beta + k1 + params.k1_[0]
            )  # C1
            LCC_probs[x, y, z, j, 2] = dt * k1  # I1Ca
            LCC_probs[x, y, z, j, 3] = float32(0.0)  # I2Ca
            LCC_probs[x, y, z, j, 4] = dt * params.k1_[0]  # I1Ba
            LCC_probs[x, y, z, j, 5] = float32(0.0)  # I2Ba
            LCC_probs[x, y, z, j, 6] = dt * params.r1[0]  # O
        elif LCC[x, y, z, j] == 3:
            LCC_probs[x, y, z, j, 0] = float32(0.0)  # C2
            LCC_probs[x, y, z, j, 1] = dt * params.k2[0]  # C1
            LCC_probs[x, y, z, j, 2] = float32(1.0) - dt * (
                params.k2[0] + k3 + s2
            )  # I1Ca
            LCC_probs[x, y, z, j, 3] = dt * k3  # I2Ca
            LCC_probs[x, y, z, j, 4] = float32(0.0)  # I1Ba
            LCC_probs[x, y, z, j, 5] = float32(0.0)  # I2Ba
            LCC_probs[x, y, z, j, 6] = dt * s2  # O
        elif LCC[x, y, z, j] == 4:
            LCC_probs[x, y, z, j, 0] = dt * k5  # C2
            LCC_probs[x, y, z, j, 1] = float32(0.0)  # C1
            LCC_probs[x, y, z, j, 2] = dt * k4  # I1Ca
            LCC_probs[x, y, z, j, 3] = float32(1.0) - dt * (k4 + k5)  # I2Ca
            LCC_probs[x, y, z, j, 4] = float32(0.0)  # I1Ba
            LCC_probs[x, y, z, j, 5] = float32(0.0)  # I2Ba
            LCC_probs[x, y, z, j, 6] = float32(0.0)  # O
        elif LCC[x, y, z, j] == 5:
            LCC_probs[x, y, z, j, 0] = float32(0.0)  # C2
            LCC_probs[x, y, z, j, 1] = dt * params.k2_[0]  # C1
            LCC_probs[x, y, z, j, 2] = float32(0.0)  # I1Ca
            LCC_probs[x, y, z, j, 3] = float32(0.0)  # I2Ca
            LCC_probs[x, y, z, j, 4] = float32(1.0) - dt * (
                params.k2_[0] + k3_ + s2_
            )  # I1Ba
            LCC_probs[x, y, z, j, 5] = dt * k3_  # I2Ba
            LCC_probs[x, y, z, j, 6] = dt * s2_  # O
        elif LCC[x, y, z, j] == 6:
            LCC_probs[x, y, z, j, 0] = dt * k5_  # C2
            LCC_probs[x, y, z, j, 1] = float32(0.0)  # C1
            LCC_probs[x, y, z, j, 2] = float32(0.0)  # I1Ca
            LCC_probs[x, y, z, j, 3] = float32(0.0)  # I2Ca
            LCC_probs[x, y, z, j, 4] = dt * k4_  # I1Ba
            LCC_probs[x, y, z, j, 5] = float32(1.0) - dt * (k5_ + k4_)  # I2Ba
            LCC_probs[x, y, z, j, 6] = float32(0.0)  # O
        elif LCC[x, y, z, j] == 7:
            LCC_probs[x, y, z, j, 0] = float32(0.0)  # C2
            LCC_probs[x, y, z, j, 1] = dt * params.r2[0]  # C1
            LCC_probs[x, y, z, j, 2] = dt * s1  # I1Ca
            LCC_probs[x, y, z, j, 3] = float32(0.0)  # I2Ca
            LCC_probs[x, y, z, j, 4] = dt * params.s1_[0]  # I1Ba
            LCC_probs[x, y, z, j, 5] = float32(0.0)  # I2Ba
            LCC_probs[x, y, z, j, 6] = float32(1.0) - dt * (
                params.r2[0] + s1 + params.s1_[0]
            )  # O


@cuda.jit(device=True, inline=True)
def sample_LCC_icdf(
    LCC_probs: npt.NDArray[f32], rng_states: RNG_state, i: i32, j: i32
) -> i32:
    """
    Sample LCC state using inverse cdf method
    Args:
        LCC_probs (npt.NDArray[f32]): The LCC probabilities
        rng_states (_type_): The xorshiro rng states for sampling
        i (i32): CRU position index
        j (i32): LCC index within CRU i

    Returns:
        i32: The sampled state
    """
    u = xoroshiro128p_uniform_float32(rng_states, i)
    cdf = float32(0.0)
    for k in range(7):
        cdf += LCC_probs[i, j, k]
        if u < cdf:
            return int32(k + 1)  # state starts at 1 so increment
    return int32(7)


@cuda.jit(device=True, inline=True)
def sample_LCC_icdf_3d(
    LCC_probs: npt.NDArray[f32],
    rng_states: RNG_state,
    x: i32,
    y: i32,
    z: i32,
    lcc_num: i32,
    junctional: bool,
    tid: i32,
) -> i32:
    """
    Sample LCC state using inverse cdf method
    Args:
        LCC_probs (npt.NDArray[f32]): The LCC probabilities
        rng_states (_type_): The xorshiro rng states for sampling
        i (i32): CRU position index
        j (i32): LCC index within CRU i

    Returns:
        i32: The sampled state
    """
    if junctional:
        u = xoroshiro128p_uniform_float32(rng_states, tid)
        cdf = float32(0.0)
        for k in range(7):
            cdf += LCC_probs[x, y, z, lcc_num, k]
            if u < cdf:
                return int32(k + 1)  # state starts at 1 so increment
        return int32(7)
    else:
        return int32(0)


def calculate_V_dep_LCC_params(
    V: float | f32, params: RestrepoParams, single_precision: bool = True
) -> LCC_params:
    po_inf = 1.0 / (1.0 + np.exp(-V / 8))
    Pr = 1.0 / (1.0 + np.exp(-(V + 40.0) / 4.0))
    Ps = 1.0 / (1.0 + np.exp(-(V + 40.0) / 11.32))
    R = 10.0 + 4954.0 * np.exp(V / 15.6)
    tauBa = (R - params.TBa) * Pr + params.TBa

    alpha = po_inf / params.tau_po
    beta = (1.0 - po_inf) / params.tau_po
    k3 = np.exp(-(V + 40.0) / 3.0) / (3.0 * (1.0 + np.exp(-(V + 40.0) / 3.0)))
    k5_ = (1.0 - Ps) / tauBa
    k6_ = Ps / tauBa

    if single_precision:
        return (
            f32(alpha),
            f32(beta),
            f32(k3),
            f32(k5_),
            f32(k6_),
            f32(Pr),
            f32(Ps),
            f32(R),
        )
    else:
        return alpha, beta, k3, k5_, k6_, Pr, Ps, R


def LCC_stationary(
    cp: npt.NDArray, V: float, params: RestrepoParams, single_precision: bool = True
) -> npt.NDArray:
    alpha, beta, k3, k5_, k6_, Pr, Ps, R = calculate_V_dep_LCC_params(
        V, params, single_precision=single_precision
    )

    TCa = (78.0329 + 0.1 * (1 + cp / params.cp_bar) ** 4) / (
        1.0 + (cp / params.cp_bar) ** 4
    )

    k1 = 0.03 / (1.0 + (params.cp_tilde / cp) ** 3)
    tauCa = (R - TCa) * Pr + TCa
    k5 = (1.0 - Ps) / tauCa
    k6 = Ps / (tauCa * (1.0 + (params.cp_bar / params.cp_tilde) ** 3))

    piC1_un = alpha / beta
    piI2Ca_un = k6 / k5
    piI2Ba_un = k6_ / k5_
    piI1Ca_un = (k1 / params.k2) * piC1_un
    piI1Ba_un = (params.k1_ / params.k2_) * piC1_un
    piO_un = (params.r1 / params.r2) * piC1_un

    pi_un = np.zeros((*cp.shape, 7))
    pi_un[..., 0] = piC1_un
    pi_un[..., 1] = 1.0
    pi_un[..., 2] = piI1Ca_un
    pi_un[..., 3] = piI2Ca_un
    pi_un[..., 4] = piI1Ba_un
    pi_un[..., 5] = piI2Ba_un
    pi_un[..., 6] = piO_un
    out = np.zeros((*cp.shape, 4), dtype=np.int32)

    ind = [1, 2, 3, 4, 5, 6, 7]
    for i in range(cp.shape[0]):
        out[0, i, :] = np.random.choice(
            ind, p=(pi_un[0, i, :] / pi_un[0, i, :].sum()), size=(4,)
        )
        out[-1, i, :] = np.random.choice(
            ind, p=(pi_un[-1, i, :] / pi_un[-1, i, :].sum()), size=(4,)
        )

    for i in range(1, cp.shape[1] - 1):
        out[i, 0, :] = np.random.choice(
            ind, p=(pi_un[i, 0, :] / pi_un[i, 0, :].sum()), size=(4,)
        )
        out[i, -1, :] = np.random.choice(
            ind, p=(pi_un[i, -1, :] / pi_un[i, -1, :].sum()), size=(4,)
        )

    if single_precision:
        return out.astype(f32)
    else:
        return out.astype(f32)


@njit
def LCC_gillespie_test(
    V: float,
    cp: float,
    params: RestrepoParams,
    init_state: int = 1,
    nlcc: int = 1000,
    nstep: int = 10_000,
):
    state = np.zeros(7, dtype=np.int64)
    state[init_state - 1] = nlcc

    po_inf = 1.0 / (1.0 + np.exp(-V / 8))
    Pr = 1.0 / (1.0 + np.exp(-(V + 40.0) / 4.0))
    Ps = 1.0 / (1.0 + np.exp(-(V + 40.0) / 11.32))
    R = 10.0 + 4954.0 * np.exp(V / 15.6)
    tauBa = (R - params.TBa) * Pr + params.TBa

    alpha = po_inf / params.tau_po
    beta = (1.0 - po_inf) / params.tau_po
    k3 = np.exp(-(V + 40.0) / 3.0) / (3.0 * (1.0 + np.exp(-(V + 40.0) / 3.0)))
    k5_ = (1.0 - Ps) / tauBa
    k6_ = Ps / tauBa

    cptilde_cp3 = (params.cp_tilde / cp) ** 3
    TCa = (78.0329 + 0.1 * (1.0 + cp / params.cp_bar) ** 4) / (
        float32(1.0) + (cp / params.cp_bar) ** 4
    )
    tauCa = (R - TCa) * Pr + TCa

    s1 = 0.02 / (1.0 + cptilde_cp3)
    k1 = 0.03 / (1.0 + cptilde_cp3)
    if V < -40.0:
        k5 = k5_
        k6 = k6_
    else:
        k5 = (1.0 - Ps) / tauCa
        k6 = Ps / (tauCa * (1.0 + cptilde_cp3))

    s2 = s1 * params.k2 * params.r1 / (k1 * params.r2)
    s2_ = params.s1_ * params.k2_ * params.r1 / (params.k1_ * params.r2)
    k4 = k3 * (alpha / beta) * (k1 / params.k2) * (k5 / k6)
    k4_ = k3 * (alpha / beta) * (params.k1_ / params.k2_) * (k5_ / k6_)

    out_state = np.zeros((nstep + 1, 7), dtype=np.int64)
    t_out = np.zeros(nstep + 1)
    rates = np.zeros(20)
    cdf = np.zeros(20)

    out_state[0, :] = np.copy(state)
    t = 0.0

    for i in range(nstep):
        rates[0] = alpha * state[0]
        rates[1] = k6 * state[0]
        rates[2] = k6_ * state[0]

        rates[3] = beta * state[1]
        rates[4] = k1 * state[1]
        rates[5] = params.k1_ * state[1]
        rates[6] = params.r1 * state[1]

        rates[7] = params.k2 * state[2]
        rates[8] = k3 * state[2]
        rates[9] = s2 * state[2]

        rates[10] = k4 * state[3]
        rates[11] = k5 * state[3]

        rates[12] = params.k2_ * state[4]
        rates[13] = k3 * state[4]
        rates[14] = s2_ * state[4]

        rates[15] = k4_ * state[5]
        rates[16] = k5_ * state[5]

        rates[17] = params.r2 * state[6]
        rates[18] = s1 * state[6]
        rates[19] = params.s1_ * state[6]

        cdf[:] = np.cumsum(rates)
        cum_rate = cdf[-1]
        cdf /= cum_rate

        u = np.random.rand()

        if u < cdf[0]:
            state[0] -= 1
            state[1] += 1
        elif u < cdf[1]:
            state[0] -= 1
            state[3] += 1
        elif u < cdf[2]:
            state[0] -= 1
            state[5] += 1
        elif u < cdf[3]:
            state[1] -= 1
            state[0] += 1
        elif u < cdf[4]:
            state[1] -= 1
            state[2] += 1
        elif u < cdf[5]:
            state[1] -= 1
            state[4] += 1
        elif u < cdf[6]:
            state[1] -= 1
            state[6] += 1
        elif u < cdf[7]:
            state[2] -= 1
            state[1] += 1
        elif u < cdf[8]:
            state[2] -= 1
            state[3] += 1
        elif u < cdf[9]:
            state[2] -= 1
            state[6] += 1
        elif u < cdf[10]:
            state[3] -= 1
            state[2] += 1
        elif u < cdf[11]:
            state[3] -= 1
            state[0] += 1
        elif u < cdf[12]:
            state[4] -= 1
            state[1] += 1
        elif u < cdf[13]:
            state[4] -= 1
            state[5] += 1
        elif u < cdf[14]:
            state[4] -= 1
            state[6] += 1
        elif u < cdf[15]:
            state[5] -= 1
            state[4] += 1
        elif u < cdf[16]:
            state[5] -= 1
            state[0] += 1
        elif u < cdf[17]:
            state[6] -= 1
            state[1] += 1
        elif u < cdf[18]:
            state[6] -= 1
            state[2] += 1
        else:
            state[6] -= 1
            state[4] += 1

        dt = np.random.exponential(1 / cum_rate)
        t += dt

        t_out[i + 1] = t
        out_state[i + 1, :] = np.copy(state)

    return t_out, out_state


@njit
def LCC_kolmogorov_test(
    V: float,
    cp: float,
    params: RestrepoParams,
    init_state: int = 1,
    nlcc: int = 1000,
    nstep: int = 10_000,
    dt: float = 1e-3,
):
    states = np.ones(nlcc, dtype=np.int64) * init_state
    probs = np.zeros((nlcc, 7))

    po_inf = 1.0 / (1.0 + np.exp(-V / 8))
    Pr = 1.0 / (1.0 + np.exp(-(V + 40.0) / 4.0))
    Ps = 1.0 / (1.0 + np.exp(-(V + 40.0) / 11.32))
    R = 10.0 + 4954.0 * np.exp(V / 15.6)
    tauBa = (R - params.TBa) * Pr + params.TBa

    alpha = po_inf / params.tau_po
    beta = (1.0 - po_inf) / params.tau_po
    k3 = np.exp(-(V + 40.0) / 3.0) / (3.0 * (1.0 + np.exp(-(V + 40.0) / 3.0)))
    k5_ = (1.0 - Ps) / tauBa
    k6_ = Ps / tauBa

    cptilde_cp3 = (params.cp_tilde / cp) ** 3
    TCa = (78.0329 + 0.1 * (1.0 + cp / params.cp_bar) ** 4) / (
        float32(1.0) + (cp / params.cp_bar) ** 4
    )
    tauCa = (R - TCa) * Pr + TCa

    s1 = 0.0182688 / (1.0 + cptilde_cp3)
    k1 = 0.024168 / (1.0 + cptilde_cp3)
    if V < -40.0:
        k5 = k5_
        k6 = k6_
    else:
        k5 = (1.0 - Ps) / tauCa
        k6 = Ps / (tauCa * (1.0 + cptilde_cp3))

    s2 = s1 * params.k2 * params.r1 / (k1 * params.r2)
    s2_ = params.s1_ * params.k2_ * params.r1 / (params.k1_ * params.r2)
    k4 = k3 * (alpha / beta) * (k1 / params.k2) * (k5 / k6)
    k4_ = k3 * (alpha / beta) * (params.k1_ / params.k2_) * (k5_ / k6_)

    out_state = np.zeros((nstep + 1, 7))
    t_out = np.zeros(nstep + 1)

    for k in range(7):
        out_state[0, k] = np.mean(states == (k + 1))

    for i in range(nstep):
        for j in range(nlcc):
            if states[j] == 1:
                probs[j, 0] = 1.0 - dt * (k6 + k6_ + alpha)  # C2
                probs[j, 1] = dt * alpha  # C1
                probs[j, 2] = 0  # I1Ca
                probs[j, 3] = dt * k6  # I2Ca
                probs[j, 4] = 0  # I1Ba
                probs[j, 5] = dt * k6_  # I2Ba
                probs[j, 6] = 0  # O
            elif states[j] == 2:
                probs[j, 0] = dt * beta  # C2
                probs[j, 1] = 1.0 - dt * (params.r1 + beta + k1 + params.k1_)  # C1
                probs[j, 2] = dt * k1  # I1Ca
                probs[j, 3] = 0  # I2Ca
                probs[j, 4] = dt * params.k1_  # I1Ba
                probs[j, 5] = 0  # I2Ba
                probs[j, 6] = dt * params.r1  # O
            elif states[j] == 3:
                probs[j, 0] = 0  # C2
                probs[j, 1] = dt * params.k2  # C1
                probs[j, 2] = 1.0 - dt * (params.k2 + k3 + s2)  # I1Ca
                probs[j, 3] = dt * k3  # I2Ca
                probs[j, 4] = 0  # I1Ba
                probs[j, 5] = 0  # I2Ba
                probs[j, 6] = dt * s2  # O
            elif states[j] == 4:
                probs[j, 0] = dt * k5  # C2
                probs[j, 1] = 0  # C1
                probs[j, 2] = dt * k4  # I1Ca
                probs[j, 3] = 1.0 - dt * (k4 + k5)  # I2Ca
                probs[j, 4] = 0  # I1Ba
                probs[j, 5] = 0  # I2Ba
                probs[j, 6] = 0  # O
            elif states[j] == 5:
                probs[j, 0] = 0  # C2
                probs[j, 1] = dt * params.k2_  # C1
                probs[j, 2] = 0  # I1Ca
                probs[j, 3] = 0  # I2Ca
                probs[j, 4] = 1.0 - dt * (params.k2_ + k3 + s2_)  # I1Ba
                probs[j, 5] = dt * k3  # I2Ba
                probs[j, 6] = dt * s2_  # O
            elif states[j] == 6:
                probs[j, 0] = dt * k5_  # C2
                probs[j, 1] = 0  # C1
                probs[j, 2] = 0  # I1Ca
                probs[j, 3] = 0  # I2Ca
                probs[j, 4] = dt * k4_  # I1Ba
                probs[j, 5] = 1.0 - dt * (k5_ + k4_)  # I2Ba
                probs[j, 6] = 0  # O
            else:
                probs[j, 0] = 0  # C2
                probs[j, 1] = dt * params.r2  # C1
                probs[j, 2] = dt * s1  # I1Ca
                probs[j, 3] = 0  # I2Ca
                probs[j, 4] = dt * params.s1_  # I1Ba
                probs[j, 5] = 0  # I2Ba
                probs[j, 6] = 1.0 - dt * (params.r2 + s1 + params.s1_)  # O

            cdf = 0.0
            u = np.random.rand()
            for k in range(7):
                cdf += probs[j, k]
                if u < cdf:
                    states[j] = k + 1
                    break

        t_out[i + 1] = t_out[i] + dt
        for k in range(7):
            out_state[i + 1, k] = np.mean(states == (k + 1))

    return t_out, out_state
