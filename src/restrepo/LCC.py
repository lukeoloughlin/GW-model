import math

import numpy as np
from numba import void, f4, i4, float32, size_t  # f4 = float32
import numba.cuda as cuda
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type

# This does the LCC stuff. This should be called in a separate kernel because it only needs to work on the boundary of the domain

# 1 = C1
# 2 = C2
# 3 = I1Ca
# 4 = I2Ca
# 5 = I1Ba
# 6 = I2Ba
# 7 = O


@cuda.jit(
    device=True,
    inline=True,
)
def update_LCC_probs(
    LCC_probs,
    LCC,
    cp,
    dt,
    alpha,
    beta,
    r1,
    r2,
    s1_,
    cp_bar,
    cp_tilde,
    k1_,
    k2,
    k2_,
    k3,
    k3_,
    k5_,
    k6_,
    Pr,
    Ps,
    R,
    idx,
):
    """Does euler step for Kolmogorov equations. Assumed that the LCCs are stored in a 1d array and the indexing semantics are dealt with elsewhere"""
    cptilde_cp3 = (cp_tilde / cp) * (cp_tilde / cp) * (cp_tilde / cp)
    cp_cpbar4 = (cp / cp_bar) * (cp / cp_bar) * (cp / cp_bar) * (cp / cp_bar)
    _1pcp_cpbar4 = (
        (float32(1.0) + cp / cp_bar)
        * (float32(1.0) + cp / cp_bar)
        * (float32(1.0) + cp / cp_bar)
        * (float32(1.0) + cp / cp_bar)
    )
    TCa = (float32(78.0329) + float32(0.1) * _1pcp_cpbar4) / (float32(1.0) + cp_cpbar4)
    tauCa = (R - TCa) * Pr + TCa

    s1 = float32(0.02) / (float32(1.0) + cptilde_cp3)
    k1 = float32(0.03) / (float32(1.0) + cptilde_cp3)
    k5 = (float32(1.0) - Ps) / tauCa
    k6 = Ps / (tauCa * (float32(1.0) + cptilde_cp3))

    s2 = s1 * k2 * r1 / (k1 * r2)
    s2_ = s1_ * k2_ * r1 / (k1_ * r2)
    k4 = k3 * (alpha / beta) * (k1 / k2) * (k5 / k6)
    k4_ = k3_ * (alpha / beta) * (k1_ / k2_) * (k5_ / k6_)

    for j in range(4):
        if LCC[idx, j] == 1:
            LCC_probs[idx, j, 0] = float32(1) - dt * (r1 + beta + k1 + k1_)
            LCC_probs[idx, j, 1] = dt * beta
            LCC_probs[idx, j, 2] = dt * k1
            LCC_probs[idx, j, 3] = float32(0)
            LCC_probs[idx, j, 4] = dt * k1_
            LCC_probs[idx, j, 5] = float32(0)
            LCC_probs[idx, j, 6] = dt * r1
        elif LCC[idx, j] == 2:
            LCC_probs[idx, j, 0] = dt * alpha
            LCC_probs[idx, j, 1] = float32(1) - dt * (k6 + k6_ + alpha)
            LCC_probs[idx, j, 2] = float32(0)
            LCC_probs[idx, j, 3] = dt * k6
            LCC_probs[idx, j, 4] = float32(0)
            LCC_probs[idx, j, 5] = dt * k6_
            LCC_probs[idx, j, 6] = float32(0)
        elif LCC[idx, j] == 3:
            LCC_probs[idx, j, 0] = dt * k2
            LCC_probs[idx, j, 1] = float32(0)
            LCC_probs[idx, j, 2] = float32(1) - dt * (k2 + k3 + s2)
            LCC_probs[idx, j, 3] = dt * k3
            LCC_probs[idx, j, 4] = float32(0)
            LCC_probs[idx, j, 5] = float32(0)
            LCC_probs[idx, j, 6] = dt * s2
        elif LCC[idx, j] == 4:
            LCC_probs[idx, j, 0] = float32(0)
            LCC_probs[idx, j, 1] = dt * k5
            LCC_probs[idx, j, 2] = dt * k4
            LCC_probs[idx, j, 3] = float32(1) - dt * (k4 + k5)
            LCC_probs[idx, j, 4] = float32(0)
            LCC_probs[idx, j, 5] = float32(0)
            LCC_probs[idx, j, 6] = float32(0)
        elif LCC[idx, j] == 5:
            LCC_probs[idx, j, 0] = dt * k2_
            LCC_probs[idx, j, 1] = float32(0)
            LCC_probs[idx, j, 2] = float32(0)
            LCC_probs[idx, j, 3] = float32(0)
            LCC_probs[idx, j, 4] = float32(1) - dt * (k2_ + k3_ + s2_)
            LCC_probs[idx, j, 5] = dt * k3_
            LCC_probs[idx, j, 6] = dt * s2_
        elif LCC[idx, j] == 6:
            LCC_probs[idx, j, 0] = float32(0)
            LCC_probs[idx, j, 1] = dt * k5_
            LCC_probs[idx, j, 2] = float32(0)
            LCC_probs[idx, j, 3] = float32(0)
            LCC_probs[idx, j, 4] = dt * k4_
            LCC_probs[idx, j, 5] = float32(1) - dt * (k4_ + k5_)
            LCC_probs[idx, j, 6] = float32(0)
        else:
            LCC_probs[idx, j, 0] = dt * r2
            LCC_probs[idx, j, 1] = float32(0)
            LCC_probs[idx, j, 2] = dt * s1
            LCC_probs[idx, j, 3] = float32(0)
            LCC_probs[idx, j, 4] = dt * s1_
            LCC_probs[idx, j, 5] = float32(0)
            LCC_probs[idx, j, 6] = float32(1) - dt * (r1 + s1 + s1_)


@cuda.jit(device=True, inline=True)
def sample_icdf(LCC_probs, rng_states, i, k):
    u = xoroshiro128p_uniform_float32(rng_states, i)
    cdf = float32(0)
    for j in range(7):
        cdf += LCC_probs[i, j, k]
        if u < cdf:
            return j + 1  # state starts at 1 so increment
    return 7


@cuda.jit
def LCC_kernel(
    LCC,
    LCC_probs,
    cp,
    rng_states,
    dt,
    alpha,
    beta,
    r1,
    r2,
    s1_,
    cp_bar,
    cp_tilde,
    k1_,
    k2,
    k2_,
    k3,
    k3_,
    k5_,
    k6_,
    Pr,
    Ps,
    R,
):
    idx = cuda.grid(1)
    if idx < LCC.shape[0]:
        cp_ = get_boundary_values(cp, cp.shape[0], cp.shape[1], idx)
        update_LCC_probs(
            LCC_probs,
            LCC,
            cp_,
            dt,
            alpha,
            beta,
            r1,
            r2,
            s1_,
            cp_bar,
            cp_tilde,
            k1_,
            k2,
            k2_,
            k3,
            k3_,
            k5_,
            k6_,
            Pr,
            Ps,
            R,
            idx,
        )
        for j in range(4):
            LCC[idx, j] = sample_icdf(LCC_probs, rng_states, idx, j)
